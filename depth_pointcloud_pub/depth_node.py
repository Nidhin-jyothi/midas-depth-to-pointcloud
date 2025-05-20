import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField
import cv2
import torch
import numpy as np
import open3d as o3d

class DepthPublisher(Node):
    def __init__(self):
        super().__init__('depth_pointcloud_node')
        self.publisher_ = self.create_publisher(PointCloud2, 'depth/pointcloud', 10)

        # Load MiDaS
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.midas.to('cpu').eval()
        transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = transforms.small_transform

        # Intrinsics (for 640x480)
        self.fx, self.fy = 500, 500
        self.cx, self.cy = 320, 240

        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("No frame from camera")
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to('cpu')

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        h, w = depth_map.shape
        points = []
        colors = []

        for v in range(h):
            for u in range(w):
                Z = depth_map[v, u]
                if Z == 0:
                    continue
                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
                points.append([X, Y, Z])
                colors.append(img[v, u] / 255.0)

        # Convert to ROS PointCloud2
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_link"

        cloud_data = []
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            rgb = (r << 16) | (g << 8) | b
            cloud_data.append([x, y, z, rgb])

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
            ]

        pointcloud_msg = point_cloud2.create_cloud(header, fields, cloud_data)


        self.publisher_.publish(pointcloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

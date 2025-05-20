import cv2
import torch
import numpy as np
import open3d as o3d

# Load MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu').eval()

# Load transform
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Camera intrinsics (approximate)
fx, fy = 500, 500
cx, cy = 320, 240  # for 640x480 resolution

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to('cpu')

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Visualize depth map
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)  # ← FIXED TYPO HERE
        depth_coloured = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        cv2.imshow('RGB', frame)
        cv2.imshow('Depth Map', depth_coloured)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            print("[INFO] Saving point cloud...")

            h, w = depth_map.shape
            points = []
            colors = []

            for v in range(h):
                for u in range(w):
                    Z = depth_map[v, u]
                    if Z == 0:
                        continue
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    points.append([X, Y, Z])
                    colors.append(img[v, u] / 255.0)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.array(points))
            pc.colors = o3d.utility.Vector3dVector(np.array(colors))

            o3d.io.write_point_cloud("pointcloud.ply", pc)
            print("[INFO] Point cloud saved as pointcloud.ply")

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

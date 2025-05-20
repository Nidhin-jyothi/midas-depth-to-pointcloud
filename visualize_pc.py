import open3d as o3d

# Load saved point cloud
pcd = o3d.io.read_point_cloud("pointcloud.ply")

# Estimate normals to help 3D shading
pcd.estimate_normals()

# Coordinate frame for reference
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# Launch 3D viewer
o3d.visualization.draw_geometries(
    [pcd, axis],
    window_name='3D Point Cloud',
    width=800,
    height=600,
    mesh_show_back_face=True
)

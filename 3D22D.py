import itertools
import numpy as np
import torch
import cv2
import trimesh
import open3d as o3d
import mediapipe as mp

from DentAlign.utils.norm import normalize_points

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# Load your image
image_path = "/home/waleed/Documents/3DLearning/DDS/DentAlign/dataset/LS3D-W-balanced-20-03-2017/new_dataset/7153.jpg"
obj_file = "/home/waleed/Documents/3DLearning/3DModels/dataset/data_part_3/upper/7G52QKXB/7G52QKXB_upper.obj"
mesh = trimesh.load(obj_file)

# Convert to PyTorch tensors
vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32)
print(vertices_tensor.shape)
print(torch.max(vertices_tensor), torch.min(vertices_tensor))
vertices_array = np.array(mesh.vertices)
faces_array = np.array(mesh.faces)

image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width, _ = image.shape

# Process the image and extract the landmarks
result = face_mesh.process(rgb_image)

for face_landmarks in result.multi_face_landmarks:
    print("Land Mark: ")

    # Extract the landmark coordinates
    landmarks = []
    print(len(face_landmarks.landmark))

    for landmark in face_landmarks.landmark:
        # Collect x, y, z normalized coordinates
        landmarks.append([landmark.x, landmark.y, landmark.z])

LIP_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))

# Extract only the landmarks corresponding to the teeth
teeth_landmarks = []
for idx in LIP_INDEXES:
    landmark = face_landmarks.landmark[idx]
    teeth_landmarks.append([landmark.x, landmark.y, landmark.z])
    
    # Convert normalized coordinates to pixel coordinates
    x = int(landmark.x * width)
    y = int(landmark.y * height)

    # Draw a small circle on each tooth landmark
    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green circles for landmarks

# Convert the list to a NumPy array
teeth_landmarks_array = np.array(teeth_landmarks)

# Optionally, convert to a PyTorch tensor for use in neural networks
teeth_landmarks_tensor = torch.tensor(teeth_landmarks_array, dtype=torch.float32)

print(teeth_landmarks_tensor.shape)

# Convert landmarks into a NumPy array
landmarks_array = np.array(landmarks)

# Convert the NumPy array to a PyTorch tensor
landmarks_tensor = torch.tensor(landmarks_array, dtype=torch.float32)

vertices_array = normalize_points(vertices_array)
teeth_landmarks_array = normalize_points(teeth_landmarks_array)
print(teeth_landmarks_array.min(), teeth_landmarks_array.max())
print(vertices_array.min(), vertices_array.max())

# Convert normalized landmarks and obj points to Open3D point clouds
landmarks_pcd = o3d.geometry.PointCloud()
obj_pcd = o3d.geometry.PointCloud()

landmarks_pcd.points = o3d.utility.Vector3dVector(teeth_landmarks_array)
obj_pcd.points = o3d.utility.Vector3dVector(vertices_array)

# Perform ICP alignment
threshold = 0.01  # Distance threshold for ICP
transformation_matrix = o3d.pipelines.registration.registration_icp(
    landmarks_pcd, obj_pcd, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
).transformation

# Compute normals for the target point cloud (the 3D mesh vertices)
obj_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

# Compute normals for the source point cloud (the teeth landmarks)
landmarks_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

# You can now run ICP using point-to-plane estimation
icp_result = o3d.pipelines.registration.registration_icp(
    landmarks_pcd, obj_pcd, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
).transformation


print("Transformation Matrix:")
print(icp_result)

obj_pcd.transform(icp_result) # .transformation
obj_points = np.asarray(obj_pcd.points)
print(obj_points.min(), obj_points.max())

scale_factor = 1.0  # Adjust based on mesh size
obj_points *= scale_factor
translation_vector = np.array([0, 0, 0])  # Adjust to shift the mesh
obj_points += translation_vector


f = 1.0  # Focal length (adjust as needed)
c_x, c_y = width / 2, height / 2  # Camera center offsets

obj_vertices_2d = np.zeros((obj_points.shape[0], 2))
obj_vertices_2d[:, 0] = f * obj_points[:, 0] / obj_points[:, 2] + c_x
obj_vertices_2d[:, 1] = f * obj_points[:, 1] / obj_points[:, 2] + c_y

image_cc = image.copy()
cv2.imshow('3D Mesh Projection on Image', image_cc)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()  # Close the window when done

# Iterate through faces in the .obj file
for face in faces_array:
    pts = np.array([obj_vertices_2d[face[0]], obj_vertices_2d[face[1]], obj_vertices_2d[face[2]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image_cc, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

# Display the image with the 3D obj mesh overlaid
cv2.imshow('3D Mesh Projection on Image', image_cc)
cv2.waitKey(5000)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()  # Close the window when done
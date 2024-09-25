import itertools
import torch
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# Load and process the image
image_path = "/home/waleed/Documents/3DLearning/DDS/DentAlign/dataset/LS3D-W-balanced-20-03-2017/new_dataset/7153.jpg"
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape

# Process the image and extract landmarks
result = face_mesh.process(rgb_image)

if result.multi_face_landmarks:
    face_landmarks = result.multi_face_landmarks[0]

    # Extract only the landmarks corresponding to the teeth (lips landmarks)
    LIP_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
    teeth_landmarks = [[face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y, face_landmarks.landmark[idx].z]
                       for idx in LIP_INDEXES]

    # Convert to tensors for scaling
    scale = torch.tensor([height, width, 1.0])
    arr = torch.tensor(teeth_landmarks)

    # Calculate the min and max coordinates for the mask
    mini = torch.min(arr, dim=0)[0]
    maxi = torch.max(arr, dim=0)[0]
    min_xy = (mini * scale)[:2]  # Scale x and y
    max_xy = (maxi * scale)[:2]

    # Create a coordinate grid
    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    x_coords = x_coords.float()
    y_coords = y_coords.float()

    # Generate the mask
    mask = (x_coords >= min_xy[0]) & (x_coords <= max_xy[0]) & \
           (y_coords >= min_xy[1]) & (y_coords <= max_xy[1])
    print(mask)
    print(mask.unsqueeze(-1).shape, mask.shape)
    # Convert image to tensor and apply the mask
    ten_im = torch.tensor(rgb_image, dtype=torch.float32)
    masked_image = ten_im * (1.0 - mask.unsqueeze(-1).float())

    # Visualize the mask and masked image
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")

    plt.subplot(1, 2, 2)
    plt.imshow(masked_image.int())
    plt.title("Masked Image")

    plt.show()
else:
    print("No face landmarks detected.")

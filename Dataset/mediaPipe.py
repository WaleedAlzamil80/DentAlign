from PIL import Image
import pandas as pd
import os
import torch
import mediapipe as mp
import numpy as np
import itertools

def process_single_image(rgb_image, face_mesh):
    width, height = rgb_image.size
    result = face_mesh.process(np.array(rgb_image))

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]

        # Extract the landmarks for lips/teeth region
        LIP_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_LIPS)))
        teeth_landmarks = [[face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y, face_landmarks.landmark[idx].z]
                           for idx in LIP_INDEXES]

        # Scale landmarks to image dimensions
        scale = torch.tensor([width, height, 1.0])
        arr = torch.tensor(teeth_landmarks)

        # Get min/max xy coordinates for the bounding box
        mini = torch.min(arr, dim=0)[0]
        maxi = torch.max(arr, dim=0)[0]
        min_xy = (mini * scale)[:2]
        max_xy = (maxi * scale)[:2]

        # Create a meshgrid of image coordinates
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        x_coords = x_coords.float()
        y_coords = y_coords.float()

        # Create a binary mask within the bounding box
        mask = (x_coords >= min_xy[0]) & (x_coords <= max_xy[0]) & \
               (y_coords >= min_xy[1]) & (y_coords <= max_xy[1])

        return mask.numpy()
    else:
        # Return an empty mask if no face landmarks were detected
        return np.zeros((height, width), dtype=np.float32)

def createMask(root_dir, output_path, partition='train'):
    """
    Args:
        root_dir (string): Directory with all the CSV files.
        output_path (string): Directory where the masks will be saved.
        partition (string): One of 'train' or 'test' to specify which subset to use.
    """
    # Initialize MediaPipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    # Load the CSV files
    attr_data = pd.read_csv(os.path.join(root_dir, 'list_attr_celeba.csv'))
    eval_partition = pd.read_csv(os.path.join(root_dir, 'list_eval_partition.csv'))

    # Filter data based on the partition (0: train, 1: validation, 2: test)
    if partition == 'train':
        partition_data = eval_partition[(eval_partition['partition'] == 0) | (eval_partition['partition'] == 1)]
    elif partition == 'test':
        partition_data = eval_partition[eval_partition['partition'] == 2]
    else:
        raise ValueError("Partition must be one of 'train' or 'test'")

    # Select only images with smiling and slightly open mouth attributes
    smiling_data = attr_data[(attr_data['Smiling'] == 1) & (attr_data['Mouth_Slightly_Open'] == 1)]
    partition_data = partition_data[partition_data['image_id'].isin(smiling_data['image_id'])]

    # Image directory
    img_dir = os.path.join(root_dir, 'img_align_celeba', 'img_align_celeba')

    # Ensure output directories exist
    output_subfolder = os.path.join(output_path, partition)
    os.makedirs(output_subfolder, exist_ok=True)

    # Process each image
    for idx in range(len(partition_data)):
        img_name = partition_data.iloc[idx, 0]
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Process the image to get the mask
        mask = process_single_image(image, face_mesh)

        # Convert the mask to a PIL image (0-255 grayscale)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        # Save the mask with the same name as the image
        mask_output_path = os.path.join(output_subfolder, img_name)
        mask_img.save(mask_output_path)

    face_mesh.close()

root_directory = "/home/waleed/Documents/3DLearning/DDS/SmileShift/archive"
output_directory = "/home/waleed/Documents/3DLearning/DDS/SmileShift/output"
createMask(root_directory, output_directory, partition='train')
createMask(root_directory, output_directory, partition='test')
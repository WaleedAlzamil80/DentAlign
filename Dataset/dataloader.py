import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import mediapipe as mp
import itertools
import torchvision.transforms as T

class CelebADataset_Faces(Dataset):
    def __init__(self, root_dir, img_dir, partition='train', transform=None, filter_teeth_visible=True,):
        """
        Args:
            root_dir (string): Directory with all the CSV files.
            img_dir (string): Directory with all the images.
            partition (string): One of 'train', 'val', 'test' to specify which subset to use.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.transform = transform
        self.filter_teeth_visible = filter_teeth_visible
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

        # Load the CSV files
        self.attr_data = pd.read_csv(os.path.join(root_dir, 'list_attr_celeba.csv'))
        self.eval_partition = pd.read_csv(os.path.join(root_dir, 'list_eval_partition.csv'))
        self.bbox_data = pd.read_csv(os.path.join(root_dir, 'list_bbox_celeba.csv'))
        self.landmarks_data = pd.read_csv(os.path.join(root_dir, 'list_landmarks_align_celeba.csv'))

        # Filter the data based on the partition (0: train, 1: validation, 2: test)
        if partition == 'train':
            self.partition_data = self.eval_partition[self.eval_partition['partition'] == 0]
        elif partition == 'val':
            self.partition_data = self.eval_partition[self.eval_partition['partition'] == 1]
        elif partition == 'test':
            self.partition_data = self.eval_partition[self.eval_partition['partition'] == 2]
        else:
            raise ValueError("Partition must be one of 'train', 'val', 'test'")

        # Filter the partition data based on smiling images if requested
        if self.filter_teeth_visible:
            smiling_data = self.attr_data[(self.attr_data['Smiling'] == 1) & (self.attr_data['Mouth_Slightly_Open'] == 1)]
            self.partition_data = self.partition_data[self.partition_data['image_id'].isin(smiling_data['image_id'])]

        # We will only use the images and attributes that are part of the selected partition
        self.attr_data = self.attr_data.loc[self.partition_data.index]

    def process_single_image(self, rgb_image):
        width, height = rgb_image.size
        result = self.face_mesh.process(np.array(rgb_image))

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]

            LIP_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LIPS)))
            teeth_landmarks = [[face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y, face_landmarks.landmark[idx].z]
                               for idx in LIP_INDEXES]

            scale = torch.tensor([width, height, 1.0])
            arr = torch.tensor(teeth_landmarks)

            mini = torch.min(arr, dim=0)[0]
            maxi = torch.max(arr, dim=0)[0]
            min_xy = (mini * scale)[:2]
            max_xy = (maxi * scale)[:2]

            y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
            x_coords = x_coords.float()
            y_coords = y_coords.float()

            mask = (x_coords >= min_xy[0]) & (x_coords <= max_xy[0]) & \
                   (y_coords >= min_xy[1]) & (y_coords <= max_xy[1])

            return mask
        else:
            return torch.zeros((height, width), dtype=torch.float32)

    def __len__(self):
        return len(self.partition_data)

    def __getitem__(self, idx):
        # Get the image ID from the partition data
        img_name = os.path.join(self.img_dir, self.partition_data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")

        mask = self.process_single_image(image)

        image_tensor = T.ToTensor()(image)
        mask = mask.unsqueeze(0).float()
        
        masked_tensor = image_tensor * (1-mask)

        if self.transform:
            image_tensor, masked_tensor = self.transform(image_tensor, masked_tensor)

        return image_tensor, masked_tensor


def get_dataLoader(root_dir, batch_size, num_workers, transform, test_transform):
    img_dir = os.path.join(root_dir, 'img_align_celeba', 'img_align_celeba')

    # For training data
    train_dataset = CelebADataset_Faces(root_dir=root_dir, img_dir=img_dir, partition='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # For validation data
    val_dataset = CelebADataset_Faces(root_dir=root_dir, img_dir=img_dir, partition='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # For testing data
    test_dataset = CelebADataset_Faces(root_dir=root_dir, img_dir=img_dir, partition='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader
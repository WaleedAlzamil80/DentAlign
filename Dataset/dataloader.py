import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CelebADataset_Faces(Dataset):
    def __init__(self, root_dir, mask_dir, partition='train', transform=None, filter_teeth_visible=True):
        """
        Args:
            root_dir (string): Directory with all the images and CSV files.
            masked_dir (string): Directory with all the images.
            partition (string): One of 'train', 'val', 'test' to specify which subset to use.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'img_align_celeba', 'img_align_celeba')
        self.mask_dir = os.path.join(mask_dir, partition)
        self.transform = transform
        self.filter_teeth_visible = filter_teeth_visible

        # Load the CSV files
        self.attr_data = pd.read_csv(os.path.join(root_dir, 'list_attr_celeba.csv'))
        self.eval_partition = pd.read_csv(os.path.join(root_dir, 'list_eval_partition.csv'))
        self.bbox_data = pd.read_csv(os.path.join(root_dir, 'list_bbox_celeba.csv'))
        self.landmarks_data = pd.read_csv(os.path.join(root_dir, 'list_landmarks_align_celeba.csv'))

        if partition == 'train':
            self.partition_data = self.eval_partition[(self.eval_partition['partition'] == 0) | (self.eval_partition['partition'] == 1)]
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

    def __len__(self):
        return len(self.partition_data)

    def __getitem__(self, idx):
        # Get the image ID from the partition data
        img_name = os.path.join(self.img_dir, self.partition_data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        mask_name = os.path.join(self.mask_dir, self.partition_data.iloc[idx, 0])
        mask = Image.open(mask_name)

        if self.transform:
            image_tensor, mask_tensor = self.transform(image, mask)

        masked_tensor = image_tensor * (1 - mask_tensor)

        return image_tensor, masked_tensor
 

def get_dataLoader(args, transform, test_transform):

    # For training data
    train_dataset = CelebADataset_Faces(root_dir=args.path, mask_dir=args.mask_dir, partition='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    # For testing data
    test_dataset = CelebADataset_Faces(root_dir=args.path, mask_dir=args.mask_dir, partition='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    return train_loader, test_loader
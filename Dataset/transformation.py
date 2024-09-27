import random
import torch
import torchvision.transforms as T

class JointTransform:
    def __init__(self, 
                 resize=None, 
                 horizontal_flip_prob=0.5, 
                 vertical_flip_prob=0.5, 
                 rotation_range=None, 
                 color_jitter_params=None, 
                 crop_size=None,
                 normalize_params=None):
        """
        Args:
            resize (tuple): Resize image and mask to the specified size (width, height).
            horizontal_flip_prob (float): Probability of applying horizontal flip.
            vertical_flip_prob (float): Probability of applying vertical flip.
            rotation_range (tuple): Tuple (min_degree, max_degree) for random rotation.
            color_jitter_params (tuple): Tuple (brightness, contrast, saturation, hue) for random color jitter.
            crop_size (tuple): Size (width, height) for random crop.
            normalize_params (tuple): Tuple (mean, std) to normalize the images (applied to image only).
        """
        self.resize = resize
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_range = rotation_range
        self.color_jitter_params = color_jitter_params
        self.crop_size = crop_size
        self.normalize_params = normalize_params
        
        self.to_tensor = T.ToTensor()

        # Define color jitter transformation if parameters are provided
        if self.color_jitter_params:
            self.color_jitter = T.ColorJitter(*self.color_jitter_params)

        # Define normalization transformation if parameters are provided
        if self.normalize_params:
            self.normalize = T.Normalize(*self.normalize_params)

    def __call__(self, image, mask):
        # Resize transformation (applied to both image and mask)
        if self.resize:
            image = T.Resize(self.resize)(image)
            mask = T.Resize(self.resize)(mask)

        # Random horizontal flip (applied to both image and mask)
        if random.random() < self.horizontal_flip_prob:
            image = T.functional.hflip(image)
            mask = T.functional.hflip(mask)

        # Random vertical flip (applied to both image and mask)
        if random.random() < self.vertical_flip_prob:
            image = T.functional.vflip(image)
            mask = T.functional.vflip(mask)

        # Random rotation (applied to both image and mask)
        if self.rotation_range:
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            image = T.functional.rotate(image, angle)
            mask = T.functional.rotate(mask, angle)

        # Random crop (applied to both image and mask)
        if self.crop_size:
            i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
            image = T.functional.crop(image, i, j, h, w)
            mask = T.functional.crop(mask, i, j, h, w)

        # Color jitter (applied to the image only)
        if self.color_jitter_params:
            image = self.color_jitter(image)

        # Convert to tensor (applied to both image and mask)
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        # Normalize (applied to image only)
        if self.normalize_params:
            image = self.normalize(image)

        return image, mask
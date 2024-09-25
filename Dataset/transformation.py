import torchvision.transforms as T
import random

# Joint transform class to apply to both image and mask
class JointTransform:
    def __init__(self, resize=None, horizontal_flip_prob=0.5, rotation_range=None):
        """
        Args:
            resize (tuple): Resize image and mask to the specified size.
            horizontal_flip_prob (float): Probability of applying horizontal flip.
            rotation_range (tuple): Tuple (min_degree, max_degree) for random rotation.
        """
        self.resize = resize
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_range = rotation_range
        self.to_tensor = T.ToTensor()

    def __call__(self, image, mask):
        # Resize transformation
        if self.resize:
            image = T.Resize(self.resize)(image)
            mask = T.Resize(self.resize)(mask)

        # Random horizontal flip (applied to both image and mask)
        if random.random() < self.horizontal_flip_prob:
            image = T.functional.hflip(image)
            mask = T.functional.hflip(mask)

        # Random rotation (applied to both image and mask)
        if self.rotation_range:
            angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            image = T.functional.rotate(image, angle)
            mask = T.functional.rotate(mask, angle)

        return image, mask
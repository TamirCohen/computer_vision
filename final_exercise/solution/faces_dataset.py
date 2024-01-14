"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def get_image(self, index: int) -> Image:
        """Get an image from the dataset."""
        if index < len(self.real_image_names):
            image_path = os.path.join(self.root_path, 'real', self.real_image_names[index])
        else:
            image_path = os.path.join(self.root_path, 'fake', self.fake_image_names[index - len(self.real_image_names)])
        image = Image.open(image_path)
        return image
    
    def get_label(self, index: int) -> int:
        """Get a label from the dataset."""
        if index < len(self.real_image_names):
            return 1
        else:
            return 0

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        
        image = self.get_image(index)
        label = self.get_label(index)
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.real_image_names) + len(self.fake_image_names) 

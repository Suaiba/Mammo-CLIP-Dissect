import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
class CSAWDataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        split: str = 'train',
        transform_config=None,
        mean=0.0,
        std=1.0,
    ):
        """
        CSAW Dataset Loader.
        Args:
            root_folder (str): Path to the root folder containing the CSV and images.
            annotation_csv (str): Name of the CSV file with annotations.
            imagefolder_path (str): Path to the folder containing images.
            split (str): Dataset split ('train', 'test', 'val').
            transform (callable, optional): Transformations to apply to the images.
            mean (float): Mean for normalization.
            std (float): Standard deviation for normalization.
        """
        super().__init__()
        assert split in ['train', 'test', 'val'], 'split must be "train", "test", or "val"'
        # Load and filter the annotation CSV
        annotation_csv = pd.read_csv(os.path.join(root_folder, annotation_csv))
        self.annotation_csv = annotation_csv.loc[annotation_csv['split'] == split]
        print("number of images in the dataset: ", len(self.annotation_csv))
        self.imagefolder_path = imagefolder_path
        self.transforms = transform_config  # Load transforms
        self.mean = mean
        self.std = std
    def __len__(self):
        return len(self.annotation_csv)
    def __getitem__(self, idx):
        # Get the data row
        data_line = self.annotation_csv.iloc[idx]
        # Extract image ID and label
        image_id = data_line["anon_filename"][:-4]  # Remove '.dcm' extension
        cancer = data_line["Cancer_visible"]
        # Load the image
        image_path = os.path.join(self.imagefolder_path, image_id + '.png')
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        # Convert grayscale to 3-channel RGB
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        # Apply transformations
        if self.transforms is not None:
            image = self.transforms(image)
        """
        # Normalize the image
        image = image.astype('float32')
        image -= image.min()
        image /= image.max()
        image = (image - self.mean) / self.std
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        """
        # Convert label to tensor
        #label = torch.tensor(cancer, dtype=torch.long)
        label=cancer

        return image,label

class CSAWDataset_all_splits(Dataset):
    def __init__(
        self,
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        transform_config=None,
        mean=0.0,
        std=1.0,
    ):
        """
        CSAW Dataset Loader.
        Args:
            root_folder (str): Path to the root folder containing the CSV and images.
            annotation_csv (str): Name of the CSV file with annotations.
            imagefolder_path (str): Path to the folder containing images.
            split (str): Dataset split ('train', 'test', 'val').
            transform (callable, optional): Transformations to apply to the images.
            mean (float): Mean for normalization.
            std (float): Standard deviation for normalization.
        """
        super().__init__()
        # Load and filter the annotation CSV
        self.annotation_csv = pd.read_csv(os.path.join(root_folder, annotation_csv))
        print("number of images in the dataset: ", len(self.annotation_csv))
        self.imagefolder_path = imagefolder_path
        self.transforms = transform_config  # Load transforms
        self.mean = mean
        self.std = std
    def __len__(self):
        return len(self.annotation_csv)
    def __getitem__(self, idx):
        # Get the data row
        data_line = self.annotation_csv.iloc[idx]
        # Extract image ID and label
        image_id = data_line["anon_filename"][:-4]  # Remove '.dcm' extension
        cancer = data_line["Cancer_visible"]
        # Load the image
        image_path = os.path.join(self.imagefolder_path, image_id + '.png')
        image = Image.open(image_path)
        image = np.array(image, dtype=np.float32)
        # Convert grayscale to 3-channel RGB
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        # Apply transformations
        if self.transforms is not None:
            image = self.transforms(image)
        """
        # Normalize the image
        image = image.astype('float32')
        image -= image.min()
        image /= image.max()
        image = (image - self.mean) / self.std
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        """
        # Convert label to tensor
        #label = torch.tensor(cancer, dtype=torch.long)
        label=cancer

        return image,label
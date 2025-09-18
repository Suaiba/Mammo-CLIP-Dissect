import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
class CustomImageDatasetFromTxt(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        Args:
            txt_file (str): Path to the .txt file containing image paths and labels.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.txt_file = txt_file
        self.transform = transform
        self.samples = []  # List to store (image_path, label) tuples
        # Load the image paths and labels from the .txt file
        self._load_samples()
    def _load_samples(self):
        """Loads the image paths and labels from the .txt file."""
        with open(self.txt_file, "r") as f:
            for line in f:
                img_path, label = line.strip().split(" ")
                self.samples.append((img_path, int(label)))
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, label) where image is the transformed image tensor and label is the class index.
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # Open the image and convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, label

class EMBED_Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (str): Directory containing the PNG images.
            csv_file (str): Path to the CSV file containing filenames and labels.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        # Ensure the CSV has the required columns
        if 'filename' not in self.data.columns or 'Implant_type' not in self.data.columns:
            raise ValueError("CSV file must contain 'filename' and 'Implant_type' columns.")
        # Map 'Implant_type' to labels (0 for 'non-implant', 1 for 'implant')
        self.data['label'] = self.data['Implant_type'].map({'non-implant': 0, 'implant': 1})
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Get the filename and label from the CSV
        row = self.data.iloc[idx]
        img_name = row['filename']
        label = row['label']
        # Construct the full path to the image
        img_path = os.path.join(self.root_dir, img_name)
        # Load the image
        image = Image.open(img_path).convert("RGB")
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        return image, label

class EMBED_marker_Dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (str): Directory containing the PNG images.
            csv_file (str): Path to the CSV file containing filenames and labels.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        # Ensure the CSV has the required columns
        if 'filename' not in self.data.columns or 'Marker' not in self.data.columns:
            raise ValueError("CSV file must contain 'filename' and 'Marker' columns.")
        # Map 'Implant_type' to labels (0 for 'non-implant', 1 for 'implant')
        self.data['label'] = self.data['Marker'].map({'No': 0, 'Yes': 1})
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Get the filename and label from the CSV
        row = self.data.iloc[idx]
        img_name = row['filename']
        label = row['label']
        # Construct the full path to the image
        img_path = os.path.join(self.root_dir, img_name)
        # Load the image
        image = Image.open(img_path).convert("RGB")
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        return image, label
class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        """
        Args:
            dataset1 (Dataset): The first dataset.
            dataset2 (Dataset): The second dataset.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset1_len = len(dataset1)
        self.dataset2_len = len(dataset2)
        self.total_len = self.dataset1_len + self.dataset2_len
    def __len__(self):
        """Returns the total length of the combined dataset."""
        return self.total_len
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, label) from the appropriate dataset.
        """

        if idx < self.dataset1_len:
            image, label = self.dataset1[idx]

        else:
            vindr_data = self.dataset2[idx - self.dataset1_len]
            image = vindr_data['images']  # Use 'images' as the image tensor
            label = vindr_data['mass']  # Use 'mass' as the label (or choose another key)
        return image, label
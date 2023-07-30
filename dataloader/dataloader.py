import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class AnomalyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (pd.DataFrame): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        current_sample = self.csv_file.iloc[idx, 0]
        label = self.csv_file.iloc[idx, 1]
        label_path = os.path.join(self.root_dir, label)
        label = io.imread(label_path)
        images = []
        for item in current_sample:
          img_name = os.path.join(self.root_dir, item)
          image = io.imread(img_name)
          image = transform.resize(image, (224, 224))
          # (batch_size, channels, frames, height, width)
          image = rearrange(image, 'h w c -> c h w')
          images.append(image)

        label = transform.resize(label, (224, 224))
        label = rearrange(label, 'h w c -> c h w')
        x = np.stack(images[:4])
        x = rearrange(x, 't c h w -> c t h w')
        return x, label, images
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = torch.tensor([0, 1], dtype=torch.float)
        if isinstance(self.data.iloc[idx, 0], str):
            label = torch.tensor([1, 0], dtype=torch.float)
   
        if isinstance(self.data.iloc[idx, 0], str):
            img_path = self.data.iloc[idx, 0]
        else:
            img_path = self.data.iloc[idx, 1]

        img_path = 'data/' + img_path
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

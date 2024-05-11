import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Class that load the dataset and return item"""

    def __init__(self, csv_file: str, base_img_path: str = '', transform: list = None) -> None:
        """Method to initialize the class

        :param csv_file: the path to the csv file
        :type csv_file: str
        :param base_img_path: the base path to the images, defaults to ''
        :type base_img_path: str, optional
        :param transform: a list of transformation that will be applied to the image, defaults to None
        :type transform: list, optional
        """
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self._base_img_path = base_img_path

    def __len__(self) -> int:
        """Method to return the length of the dataset

        :return: the length of the dataset
        :rtype: int
        """
        
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Method to return an item from the dataset

        :param idx: the index of the item to return
        :type idx: int
        :return: a tuple containing the image and the label
        :rtype: tuple
        """
        
        label = torch.tensor([0, 1], dtype=torch.float)
        if isinstance(self.data.iloc[idx, 0], str):
            label = torch.tensor([1, 0], dtype=torch.float)

        if isinstance(self.data.iloc[idx, 0], str):
            img_path = self.data.iloc[idx, 0]
        else:
            img_path = self.data.iloc[idx, 1]

        img_path = self._base_img_path + img_path
        image = Image.open(img_path).convert("RGB") 

        if self.transform:
            image = self.transform(image)

        return image, label

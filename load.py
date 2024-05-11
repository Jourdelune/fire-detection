from dataloader import CustomDataset
import torch


class Data:
    """Class to load the data and return the trainloader and the testloader"""
    
    def __init__(self, csv_file: str = 'data/images.csv', base_img_path: str = 'data/', batch_size: int = 4, transform: list = None):
        """Method to initialize the class

        :param csv_file: the path of the csv file, defaults to 'data/images.csv'
        :type csv_file: str, optional
        :param base_img_path: the base path to the images, defaults to 'data/'
        :type base_img_path: str, optional
        :param batch_size: the size of the batch, defaults to 4
        :type batch_size: int, optional
        :param transform: a list of transformation applied to the image, defaults to None
        :type transform: list, optional
        """
        
        data = CustomDataset(
            csv_file=csv_file, transform=transform, base_img_path=base_img_path)

        test_size = min(int(0.2 * len(data)), 2000)
        train_size = len(data) - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            data, [train_size, test_size])

        self._trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self._testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    def get_loader(self) -> tuple:
        """Method to return the trainloader and the testloader

        :return: a tuple containing the trainloader and the testloader
        :rtype: tuple
        """
        
        return self._trainloader, self._testloader

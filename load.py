from dataloader import CustomDataset
import torch


data = CustomDataset(csv_file='data/images.csv')


class Data:
    def __init__(self, csv_file='data/images.csv', batch_size=4, transform=None):
        data = CustomDataset(csv_file=csv_file, transform=transform)
 
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            data, [train_size, test_size])

        self._trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self._testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    def get_loader(self):
        return self._trainloader, self._testloader

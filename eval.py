import torch


class Eval:
    """Class to evaluate the model on the testloader"""
    
    def __init__(self, testloader: torch.utils.data.dataloader.DataLoader, device: torch.device) -> None:
        """Method to initialize the class

        :param testloader: the testloader
        :type testloader: torch.utils.data.dataloader.DataLoader
        :param device: the device to use
        :type device: torch.device
        """

        self._testloader = testloader
        self.device = device

    def eval(self, model) -> int:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self._testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                
                predicted = torch.nn.functional.sigmoid(outputs)
                total += labels.size(0)

                for i in range(len(labels)):
                    if labels[i].bool() == (predicted[i] >= 0.5):
                        correct += 1

        return 100 * correct / total

import torch.nn as nn
import torch


class Net(nn.Module):
    """This class define the neural network architecture applied at the end of the resnet18 model"""

    def __init__(self, input_features: int) -> None:
        """Method to initialize the class

        :param input_features: the number of input features (the number of output features of the resnet18 model)
        :type input_features: int
        """

        super(Net, self).__init__()
        self.fc = nn.Linear(input_features, 1)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """Method to forward propagate the input through the network

        :param x: the input tensor
        :type x: torch.tensor
        :return: the output tensor
        :rtype: torch.Tensor
        """
        
        return self.fc(x)

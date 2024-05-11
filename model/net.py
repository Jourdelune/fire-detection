import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_features: int):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_features, 2)
    
    def forward(self, x):
        return self.fc(x)

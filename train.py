import torch
import torch.nn as nn
import torch.optim as optim

from config import *
from early_stoping import EarlyStopper
from eval import Eval
from load import Data
from model.net import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

data = Data(csv_file=CSV_PATH,
            batch_size=BATCH_SIZE, transform=transform, base_img_path=BASE_IMG_PATH)
trainloader, testloader = data.get_loader()

early_stopping = EarlyStopper(patience=4)

model = torch.hub.load('pytorch/vision:v0.10.0',
                       'resnet18', weights=PRETRAINED)
model.to(device)

model.fc = Net(model.fc.in_features)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

eval = Eval(testloader, device)

for epoch in range(EPOCH):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
  
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % PRINT_OCC == PRINT_OCC - 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_OCC:.3f}')
            running_loss = 0.0

    scheduler.step()
    
    if not EARLY_STOPPING:
        continue

    with torch.no_grad():
        # calculate validation_loss
        validation_loss = 0.0
    
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

        if early_stopping.early_stop(validation_loss):
            print(f'Early stopping on epoch {epoch + 1}')
            break

print('Finished Training')

PATH = './weights/fire_detect.pth'
torch.save(model.state_dict(), PATH)

print(
    f'Accuracy of the network on the test images: {eval.eval(model):.2f}%')
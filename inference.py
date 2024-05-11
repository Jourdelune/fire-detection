import torch
from PIL import Image
from model.net import Net
import config

PATH = 'weights/fire_detect.pth'
FILENAME = "/home/jourdelune/Bureau/dev/fire-detection/data/firelookout/img/test/not fire/1.jpg"

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = Net(model.fc.in_features)

model.load_state_dict(torch.load(PATH))

input_image = Image.open(FILENAME)

input_tensor = config.transform(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.sigmoid(output)
print(f"Prob feux : {(probabilities >= 0.5)[0][0]} | {torch.round(probabilities[0][0], decimals=3)}")

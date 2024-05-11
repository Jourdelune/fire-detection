import torch
from PIL import Image
from torchvision import transforms
from model.net import Net
import config

PATH = 'weights/fire_detect.pth'
FILENAME = "PATH"


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
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
print(f"Prob feux : {torch.round(probabilities[0])}, Prob pas feux : {torch.round(probabilities[1])}")

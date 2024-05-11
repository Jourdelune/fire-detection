import torchvision.transforms as transforms

BATCH_SIZE = 1  # Number of image to process at once
EPOCH = 1000  # Number of epoch to train the model
PRINT_OCC = 200  # Number of batch before printing the loss
PATIENCE = 4  # Number of epoch to wait before stopping the training if early stopping is enabled
CSV_PATH = 'data/images2.csv'
BASE_IMG_PATH = 'data/'
PRETRAINED = True  # Set to False to train from scratch
EARLY_STOPPING = True  # Set to False to disable early stopping

# Define the transformation to apply to the image
transform = transforms.Compose([
    transforms.Resize((1200, 720)),
    transforms.CenterCrop(224*3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

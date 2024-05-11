import torchvision.transforms as transforms

BATCH_SIZE = 1
EPOCH = 1000
PRINT_OCC = 200
PATIENCE = 4
CSV_PATH = 'data/images.csv'
BASE_IMG_PATH = 'data/'
PRETRAINED = True
EARLY_STOPPING = True

transform = transforms.Compose([
    transforms.Resize((1200, 720)),
    transforms.CenterCrop(224*3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

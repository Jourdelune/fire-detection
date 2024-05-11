import torchvision.transforms as transforms

BATCH_SIZE = 4
EPOCH = 1000
PRINT_OCC = 200
PATIENCE = 4
CSV_PATH = 'data/images.csv'
PRETRAINED = True
EARLY_STOPPING = True

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

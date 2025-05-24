from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

def get_dataloaders(data_dir, batch_size=16, input_size=224):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'valid')  # adjust if your folder is named differently

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_path, transform=transform_train)
    val_data = datasets.ImageFolder(val_path, transform=transform_val)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_data.classes

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from utils.dataset import get_dataloaders
from utils.metrics import evaluate_model
from models.vgg import CustomVGG
from models.resnet import CustomResNet
from models.mobilenet import CustomMobileNet
from models.inception import CustomInception
from models.densenet import CustomDenseNet

MODEL_MAP = {
    'vgg': CustomVGG,
    'resnet': CustomResNet,
    'mobilenet': CustomMobileNet,
    'inception': CustomInception,
    'densenet': CustomDenseNet
}

def train(model, train_loader, val_loader, class_names, device, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")
        report, _ = evaluate_model(model, val_loader, device, class_names)
        print("\nValidation Report:\n")
        print(report)

        acc = float(report.split('\n')[-2].split()[-2])  # crude macro avg acc
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pth')
            print("✅ Saved best model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/pets')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_MAP.keys())
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, class_names = get_dataloaders(args.data)

    model_class = MODEL_MAP[args.model]
    model = model_class(num_classes=len(class_names)).to(device)

    train(model, train_loader, val_loader, class_names, device, epochs=args.epochs)

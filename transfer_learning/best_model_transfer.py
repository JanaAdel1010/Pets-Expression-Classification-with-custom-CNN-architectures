import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils.dataset import get_dataloaders
from utils.metrics import evaluate_model

if __name__ == '__main__':
    data_dir = 'data/pets'
    train_loader, val_loader, class_names = get_dataloaders(data_dir)
    num_classes = len(class_names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    for epoch in range(5):
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

        print(f"Epoch {epoch+1}/5, Loss: {running_loss:.4f}")

    torch.save(model.state_dict(), 'transfer_best_model.pth')

    report, matrix = evaluate_model(model, val_loader, device, class_names)
    print("\nTransfer Learning Evaluation Report:\n")
    print(report)
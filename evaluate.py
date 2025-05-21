import torch
from models.vgg import CustomVGG
from utils.dataset import get_dataloaders
from utils.metrics import evaluate_model
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/pets')
    parser.add_argument('--model', default='best_model.pth')
    args = parser.parse_args()

    train_loader, val_loader, class_names = get_dataloaders(args.data)
    model = CustomVGG(num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model))
    model = model.cuda()

    report, matrix = evaluate_model(model, val_loader, device='cuda', class_names=class_names)
    
    print("Classification Report:\n", report)
    
    sns.heatmap(matrix, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("confusion_matrix.png")
    plt.show()

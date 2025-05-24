from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    matrix = confusion_matrix(all_labels, all_preds)

    return report, accuracy, matrix

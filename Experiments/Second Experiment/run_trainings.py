import os
import time
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime


project_root = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

training_dirs = [
    os.path.join(project_root, r"_data/datasets/train"),

    # Uncomment these after running all the preprocessing scripts
    # os.path.join(project_root, "_data/datasets/train_cropped_background"),
    # os.path.join(project_root, "_data/datasets/train_blurred_background"),
    # os.path.join(project_root, r"_data/datasets/train_blacked-out_background"),
    # os.path.join(project_root, r"_data/datasets/5) CAM augmentation"),
    # os.path.join(project_root, r"_data/datasets/6) CAM deletion"),
    # os.path.join(project_root, r"_data/datasets/6) CAM deletion"),
    # os.path.join(project_root, r"_data/datasets/train_gradCAM++_blur"),
    # os.path.join(project_root, r"_data/datasets/train_scoreCAM_blur")
]

test_dir = os.path.join(project_root, r'_data/datasets/test')

class_names = ['cat', 'dog']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def save_metrics(metrics_dict, classification_report_str, conf_matrix_fig, model_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("metrics_no_pretrain", f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report_str)

    conf_matrix_fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(conf_matrix_fig)

    with open(os.path.join(output_dir, "1) 1) no training.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return output_dir


class DogsVsCatsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.lower().endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = 0 if 'cat' in self.images[idx].lower() else 1

        if self.transform:
            image = self.transform(image)

        return image, label


def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    report_str = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report_str)

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist()
    }, report_str, fig


def benchmark_model(model, data_loader, num_runs=10, output_dir=None):
    model.eval()
    times = []

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(5):
        _ = model(dummy_input)

    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            start_time = time.time()
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                _ = model(inputs)
            times.append(time.time() - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = len(data_loader.dataset) / avg_time

    benchmark = {
        'avg_inference_time': avg_time,
        'std_inference_time': std_time,
        'throughput': throughput
    }

    if output_dir:
        with open(os.path.join(output_dir, "1) 1) no training.json"), "w") as f:
            json.dump(benchmark, f, indent=4)

    return benchmark


def train_model(model, dataloader, optimizer, criterion, num_epochs=5):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct_preds / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")


test_dataset = DogsVsCatsDataset(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

for train_dir in training_dirs:
    print(f"\n🏁 Training on: {train_dir}")
    model_name = os.path.basename(train_dir.rstrip("/\\"))
    train_dataset = DogsVsCatsDataset(train_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = torchvision.models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=1e-4)

    train_model(model, train_loader, optimizer, criterion, num_epochs=50)

    print("\n📊 Final evaluation on test set:")
    metrics, report_str, conf_fig = evaluate_model(model, test_loader)
    output_dir = save_metrics(metrics, report_str, conf_fig, model_name)

    print("\n⚡ Benchmarking on test set:")
    benchmark_model(model, test_loader, output_dir=output_dir)

    print(f"\n✅ Results for {model_name} saved to: {output_dir}")

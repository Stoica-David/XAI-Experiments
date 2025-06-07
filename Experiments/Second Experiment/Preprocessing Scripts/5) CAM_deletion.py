import os
import shutil
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import auc
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image

from retrieve_project_root import project_root


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = os.path.join(project_root, r"_data/datasets/train")
filtered_dir = os.path.join(project_root, r"_data/datasets/6) CAM deletion")
os.makedirs(filtered_dir, exist_ok=True)

class_names = ['cat', 'dog']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class DogsVsCatsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = 0 if 'cat' in img_name.lower() else 1

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


dataset = DogsVsCatsDataset(data_dir, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = torchvision.models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, 2)
model.to(device)
model.eval()

target_layer = model.features[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

steps = 20
pixels_total = 224 * 224
pixels_per_step = pixels_total // steps
threshold_auc = 0.50


def deletion_test(image_tensor, original_image_np, heatmap, class_idx):
    gray_map = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    indices = np.dstack(np.unravel_index(np.argsort(-gray_map.ravel()), (224, 224)))[0]

    masked_image = original_image_np.copy()
    scores = []

    for step in range(steps + 1):
        if step > 0:
            for i in range((step - 1) * pixels_per_step, step * pixels_per_step):
                if i >= len(indices): break
                y, x = indices[i]
                masked_image[y, x] = 0

        input_tensor = preprocess_image(masked_image, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0, class_idx].item()
            scores.append(probs)

    x = np.linspace(0, 1, steps + 1)

    return auc(x, scores), x, scores


auc_scores = []
for image_tensor, label, img_path in tqdm(dataloader):
    image_tensor = image_tensor.to(device)
    label = label.item()
    rgb_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    rgb_np = (rgb_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    rgb_np = np.clip(rgb_np, 0, 1)

    targets = [ClassifierOutputTarget(label)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
    heatmap = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()

    auc_score, _, _ = deletion_test(image_tensor, (rgb_np * 255).astype(np.uint8), heatmap, class_idx)

    auc_scores.append(auc_score)

    if auc_score >= threshold_auc:
        shutil.copy(img_path[0], os.path.join(filtered_dir, os.path.basename(img_path[0])))

print(f"\nFiltered images saved to: {filtered_dir}")
print(f"\nNumber of saved images after filtering: {len(auc_scores)}")

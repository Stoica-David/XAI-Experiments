import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


project_root = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(project_root, r"_data/datasets/train")
augmented_dir = os.path.join(project_root, r"_data/datasets/5) CAM augmentation")

os.makedirs(augmented_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.alexnet(pretrained=True)
model.to(device).eval()

target_layer = model.features[-1]

cam = GradCAM(model=model, target_layers=[target_layer])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def augment_image_with_gradcam(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

    image_np = np.array(raw_image.resize((224, 224))) / 255.0

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = outputs.argmax().item()

    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    mask = grayscale_cam > 0.5

    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    margin = 10
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(224, x1 + margin)
    y1 = min(224, y1 + margin)

    cropped = image_np[y0:y1, x0:x1]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return None

    cropped_resized = cv2.resize(cropped, (224, 224))
    cropped_resized_bgr = (cropped_resized * 255).astype(np.uint8)
    return cropped_resized_bgr


for filename in tqdm(os.listdir(input_dir)):
    if not filename.endswith(".jpg"):
        continue

    path = os.path.join(input_dir, filename)
    result = augment_image_with_gradcam(path)

    if result is not None:
        output_path = os.path.join(augmented_dir, f"aug_{filename}")
        cv2.imwrite(output_path, result)

print(f"\nAugmented images saved to: {augmented_dir}")

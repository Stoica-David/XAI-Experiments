import os
import cv2
import torch
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


project_root = os.path.dirname(os.path.abspath(__file__))

input_dir = os.path.join(project_root, r"_data/datasets/train")
output_dir = os.path.join(project_root, "_data/datasets/train_cropped_background")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.alexnet(pretrained=True)
model.eval().to(device)
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

norm_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def crop_to_cam_region(image_np, cam_mask, threshold=0.4, margin=10):
    cam_mask_bin = (cam_mask >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(cam_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_np

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    h_img, w_img = image_np.shape[:2]
    x = max(0, x - margin)
    y = max(0, y - margin)
    x2 = min(w_img, x + w + 2 * margin)
    y2 = min(h_img, y + h + 2 * margin)

    cropped = image_np[y:y2, x:x2]
    return cv2.resize(cropped, (224, 224))


for filename in tqdm(os.listdir(input_dir)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_dir, filename)
    orig_pil = Image.open(img_path).convert('RGB')
    orig_resized = orig_pil.resize((224, 224))
    img_np = np.array(orig_resized)
    img_tensor = norm_transform(orig_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_class = outputs.argmax().item()

    grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(predicted_class)])[0]
    cam_mask = cv2.resize(grayscale_cam, (224, 224))

    cropped_img = crop_to_cam_region(img_np, cam_mask, threshold=0.4)

    out_path = os.path.join(output_dir, filename)
    Image.fromarray(cropped_img).save(out_path)

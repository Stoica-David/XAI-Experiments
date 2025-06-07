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

from retrieve_project_root import project_root


input_dir = os.path.join(project_root, r"_data/datasets/train")
output_dir = os.path.join(project_root, r"_data/datasets/6) CAM deletion")
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

    threshold = 0.4
    keep_mask = (cam_mask >= threshold).astype(np.float32)
    blur_mask = 1.0 - keep_mask

    blurred_img = cv2.GaussianBlur(img_np, (21, 21), 0)

    keep_mask_3d = np.repeat(keep_mask[:, :, np.newaxis], 3, axis=2)
    blur_mask_3d = 1.0 - keep_mask_3d
    blended = (img_np * keep_mask_3d + blurred_img * blur_mask_3d).astype(np.uint8)

    out_path = os.path.join(output_dir, filename)
    Image.fromarray(blended).save(out_path)

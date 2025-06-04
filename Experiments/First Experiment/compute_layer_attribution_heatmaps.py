import os

import torch
import torchvision.models as models
import cv2
import numpy as np
import imageio
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import preprocess_image


project_root = os.path.dirname(os.path.abspath(__file__))

CAM_TYPE = "scoreCAM"  # Choose from: 'gradCAM', 'gradCAM++', 'scoreCAM'
IMG_PATH = os.path.join(project_root, r'_data/datasets/test/dog.0.jpg')
OUTPUT_GIF = f"{CAM_TYPE}.gif"

model = models.alexnet(pretrained=True)
model.eval()

conv_layers = [layer for layer in model.features if isinstance(layer, torch.nn.Conv2d)]


def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))

    return preprocess_image(np.array(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def overlay_heatmap(img_path, heatmap, layer_name):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    overlayed = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    cv2.putText(overlayed, layer_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return overlayed


CAM_CLASS_MAP = {
    "gradCAM": GradCAM,
    "gradCAM++": GradCAMPlusPlus,
    "scoreCAM": ScoreCAM
}

CAM_CLASS = CAM_CLASS_MAP.get(CAM_TYPE.lower())
if CAM_CLASS is None:
    raise ValueError(f"Invalid CAM_TYPE: {CAM_TYPE}. Choose from 'gradCAM', 'gradCAM++', 'scoreCAM'.")

input_tensor = load_image(IMG_PATH)

frames = []
for idx, layer in enumerate(conv_layers):
    layer_name = f"Conv2d_{idx}"

    cam = CAM_CLASS(model=model, target_layers=[layer])

    grayscale_cam = cam(input_tensor)[0]

    overlayed = overlay_heatmap(IMG_PATH, grayscale_cam, layer_name)

    frames.append(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))

imageio.mimsave(OUTPUT_GIF, frames, duration=0.5, loop=0)
print(f"Saved GIF: {OUTPUT_GIF}")

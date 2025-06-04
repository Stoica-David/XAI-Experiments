import torch
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import preprocess_image
import os


project_root = os.path.dirname(os.path.abspath(__file__))

METHOD = "gradCAM"  # Options: "gradCAM", "gradCAM++", "scoreCAM"
IMAGE_PATH = os.path.join(project_root, r"_data/datasets/test/dog.0.jpg")

OUTPUT_PATH = f"{METHOD}_max_pooled.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet(pretrained=True).to(DEVICE)
model.eval()

cam_method = {
    "gradCAM": GradCAM,
    "gradCAM++": GradCAMPlusPlus,
    "scoreCAM": ScoreCAM
}.get(METHOD.lower())

if cam_method is None:
    raise ValueError(f"Unsupported method: {METHOD}")


def load_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))

    img_np = np.array(img)

    input_tensor = preprocess_image(img_np, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return input_tensor.to(DEVICE), img_np


input_tensor, original_img_np = load_image(IMAGE_PATH)

conv_layers = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]

max_pooled_cam = None
for layer in conv_layers:
    cam = cam_method(model=model, target_layers=[layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    if max_pooled_cam is None:
        max_pooled_cam = grayscale_cam
    else:
        max_pooled_cam = np.maximum(max_pooled_cam, grayscale_cam)

max_pooled_cam = (max_pooled_cam - max_pooled_cam.min()) / (max_pooled_cam.max() - max_pooled_cam.min())

heatmap = cv2.applyColorMap(np.uint8(255 * max_pooled_cam), cv2.COLORMAP_JET)
original_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
overlayed_img = cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)

cv2.imwrite(OUTPUT_PATH, overlayed_img)
print(f"{METHOD} Max-pooled CAM saved at: {OUTPUT_PATH}")

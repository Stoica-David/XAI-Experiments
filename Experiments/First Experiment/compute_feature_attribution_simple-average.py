import os
import torch
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import preprocess_image


project_root = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(project_root, r"_data/datasets/test/dog.0.jpg")
cam_type = "gradCAM"  # Choose from: "gradCAM", "gradCAM++", "scoreCAM"
output_path = os.path.join(project_root, f"{cam_type}_simple-average.png")
os.makedirs(output_path, exist_ok=True)


def get_cam_class(cam_type):
    if cam_type == "gradCAM":
        return GradCAM

    elif cam_type == "gradCAM++":
        return GradCAMPlusPlus

    elif cam_type == "scoreCAM":
        return ScoreCAM

    else:
        raise ValueError("Unknown CAM technique. Choose from: 'gradCAM', 'gradCAM++', 'scoreCAM'.")


def load_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))

    return preprocess_image(np.array(img), mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]), np.array(img)


def overlay_combined_heatmap(original_img_np, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

    overlayed = cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)

    return overlayed


model = models.alexnet(pretrained=True)
model.eval()

conv_layers = [module for module in model.features if isinstance(module, torch.nn.Conv2d)]

input_tensor, original_img_np = load_image(img_path)

cam_class = get_cam_class(cam_type)

accumulated_cam = None
num_layers = 0

for layer in conv_layers:
    cam = cam_class(model=model, target_layers=[layer])
    grayscale_cam = cam(input_tensor)[0]

    if accumulated_cam is None:
        accumulated_cam = grayscale_cam
    else:
        accumulated_cam += grayscale_cam

    num_layers += 1

average_cam = accumulated_cam / num_layers
average_cam = (average_cam - average_cam.min()) / (average_cam.max() - average_cam.min())

overlayed_img = overlay_combined_heatmap(original_img_np, average_cam)

cv2.imwrite(output_path, overlayed_img)
print(f"Explanation {cam_type.upper()} saved at: {output_path}")

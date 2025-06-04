import torch
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import preprocess_image
import os
from tqdm import tqdm


project_root = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(project_root, r"_data/datasets/test/dog.0.jpg")
cam_type = "gradCAM"  # Choose from: "gradCAM", "gradCAM++", "scoreCAM"
output_path = f"{cam_type}_weighted_combined.png"


def get_cam_class(cam_type):
    if cam_type == "gradCAM":
        return GradCAM

    elif cam_type == "gradCAM++":
        return GradCAMPlusPlus

    elif cam_type == "scoreCAM":
        return ScoreCAM

    else:
        raise ValueError("Invalid type. Choose from 'gradCAM', 'gradCAM++' or 'scoreCAM'.")


def load_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))

    return preprocess_image(np.array(img), mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]), np.array(img)


def overlay_combined_heatmap(original_img_np, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

    return cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.alexnet(pretrained=True).to(device).eval()

conv_layers = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]

input_tensor, original_img_np = load_image(img_path)
input_tensor = input_tensor.to(device)

weighted_sum = None
weight_total = 0.0
cam_class = get_cam_class(cam_type)

for layer in tqdm(conv_layers, desc=f"Weighted average {cam_type.upper()}"):
    cam = cam_class(model=model, target_layers=[layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    weight = grayscale_cam.max().item()

    if weighted_sum is None:
        weighted_sum = grayscale_cam * weight
    else:
        weighted_sum += grayscale_cam * weight

    weight_total += weight

combined_cam = weighted_sum / weight_total
combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min())

overlayed_img = overlay_combined_heatmap(original_img_np, combined_cam)
cv2.imwrite(output_path, overlayed_img)

print(f"Weighted {cam_type.upper()} saved at: {output_path}")

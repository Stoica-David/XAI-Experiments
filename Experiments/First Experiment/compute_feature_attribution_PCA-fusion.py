import torch
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import preprocess_image
import os


project_root = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(project_root, r"_data/datasets/test/dog.0.jpg")
CAM_TYPE = "gradCAM"  # Options: "gradCAM", "gradCAM++", "scoreCAM"
OUTPUT_IMAGE = f"pca_fused_{CAM_TYPE}.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_COMPONENTS = 1


model = models.alexnet(pretrained=True).to(DEVICE)
model.eval()


def load_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    return preprocess_image(np.array(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(DEVICE), np.array(img)


def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)


def get_cam_class(cam_type):
    if cam_type.lower() == "gradCAM":
        return GradCAM
    elif cam_type.lower() == "gradCAM++":
        return GradCAMPlusPlus
    elif cam_type.lower() == "scoreCAM":
        return ScoreCAM
    else:
        raise ValueError("Unsupported CAM type. Choose from 'gradCAM', 'gradCAM++', or 'scoreCAM'.")


conv_layers = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]

input_tensor, original_img = load_image(IMAGE_PATH)

CAMClass = get_cam_class(CAM_TYPE)
all_cams = []

for layer in conv_layers:
    cam = CAMClass(model=model, target_layers=[layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
    all_cams.append(grayscale_cam)

stacked_cams = np.stack(all_cams, axis=0)
h, w = stacked_cams.shape[1:]
stacked_flat = stacked_cams.reshape(stacked_cams.shape[0], -1)
pca = PCA(n_components=N_COMPONENTS)
principal_components = pca.fit_transform(stacked_flat.T)
pca_cam = principal_components[:, 0].reshape(h, w)
pca_cam = (pca_cam - pca_cam.min()) / (pca_cam.max() - pca_cam.min())

if CAM_TYPE == "scoreCAM":
    pca_cam = cv2.GaussianBlur(pca_cam, (11, 11), 5)
    pca_cam = np.clip(pca_cam, 0, 1)

overlayed_img = overlay_heatmap(original_img, pca_cam)
os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
cv2.imwrite(OUTPUT_IMAGE, overlayed_img)
print(f"Saved PCA-fused {CAM_TYPE.upper()} result to: {OUTPUT_IMAGE}")

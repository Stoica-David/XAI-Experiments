import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm


project_root = os.path.dirname(os.path.abspath(__file__))

TARGET_IDS = [17, 18]
input_dir = os.path.join(project_root, r"_data/datasets/train")
output_dir = os.path.join(project_root, "_data/datasets/train_blurred_background")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

to_tensor = transforms.ToTensor()


def enhance_dog_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = to_tensor(image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])[0]

    masks = predictions['masks']
    labels = predictions['labels']
    scores = predictions['scores']

    dog_mask = np.zeros(image.size[::-1], dtype=np.uint8)  # (H, W)
    for i in range(len(masks)):
        if labels[i].item() in TARGET_IDS and scores[i].item() > 0.5:
            mask = masks[i, 0].cpu().numpy()
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            dog_mask = np.maximum(dog_mask, binary_mask)

    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    if np.sum(dog_mask) == 0:
        return image_cv

    blurred_background = cv2.GaussianBlur(image_cv, (21, 21), 0)
    mask_3ch = cv2.merge([dog_mask] * 3)
    inverse_mask = cv2.bitwise_not(mask_3ch)

    object_only = cv2.bitwise_and(image_cv, mask_3ch)
    background_only = cv2.bitwise_and(blurred_background, inverse_mask)
    combined = cv2.add(object_only, background_only)

    return combined


for file in tqdm(os.listdir(input_dir)):
    if not file.lower().endswith('.jpg'):
        continue

    file_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)

    if 'dog' in file.lower():
        enhanced = enhance_dog_image(file_path)
        cv2.imwrite(output_path, enhanced)
    else:
        enhanced = enhance_dog_image(file_path)
        cv2.imwrite(output_path, enhanced)

print(f"\n✅ Enhanced dataset saved to: {output_dir}")

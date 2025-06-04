import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


project_root = os.path.dirname(os.path.abspath(__file__))

# COCO class index for 'dog'
DOG_CLASS_ID = 18
# NOTE: PyTorch indexing is +1 compared to Detectron2 (which uses 17)

# Load Mask R-CNN pre-trained model with a resnet50 backbone
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def get_dog_segmentation_mask(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_tensor = transform(image).to(device)

    # Run the image through the model
    with torch.no_grad():
        predictions = model([image_tensor])[0]

    masks = predictions['masks']
    labels = predictions['labels']
    scores = predictions['scores']

    # Combine all masks that correspond to the dog class and pass a confidence threshold
    dog_mask_total = np.zeros(image.size[::-1], dtype=np.uint8)
    for i in range(len(masks)):
        if labels[i].item() == DOG_CLASS_ID and scores[i].item() > 0.5:
            mask = masks[i, 0].cpu().numpy()
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            dog_mask_total = np.maximum(dog_mask_total, binary_mask)

    # Save the dog mask
    cv2.imwrite('dog_segmentation_mask.jpg', dog_mask_total)

    return dog_mask_total


# Example usage
image_path = os.path.join(project_root, r"_data/datasets/test/dog.0.jpg")
dog_mask = get_dog_segmentation_mask(image_path)

# Show the mask
plt.imshow(dog_mask, cmap='gray')
plt.title('Dog Segmentation Mask')
plt.axis('off')
plt.show()

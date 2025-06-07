import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import os

from retrieve_project_root import project_root


image_path = os.path.join(project_root, r"_data/datasets/test/dog.0.jpg")

heatmap_paths = [
    # GradCAM
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/gradCAM_PCA-fusion.png'),
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/gradCAM_simple-average.png'),
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/gradCAM_max-pooling.png'),
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/gradCAM_weighted-average.png'),

    # GradCAM++
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/gradCAM++_PCA-fusion.png'),
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/gradCAM++_simple-average.png'),
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/gradCAM++_max-pooling.png'),
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/gradCAM++_weighted-average.png'),

    # ScoreCAM
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/scoreCAM_PCA-fusion.png'),
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/scoreCAM_simple-average.png'),
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/scoreCAM_max-pooling.png'),
    os.path.join(project_root, r'_data/heatmaps/Feature attribution/scoreCAM_weighted-average.png'),
]

output_dir = os.path.join(project_root, r'_results/insetion-tests')
os.makedirs(output_dir, exist_ok=True)
steps = 20

model = models.alexnet(pretrained=True)
model.eval()

image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Could not load image from path: {image_path}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

image_resized = cv2.resize(image_rgb, (224, 224))

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(image_resized).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    original_score = probs[0, predicted_class].item()

print(f"Original prediction: class index {predicted_class}, confidence {original_score:.4f}")

for heatmap_path in heatmap_paths:
    method_name = os.path.splitext(os.path.basename(heatmap_path))[0]

    heatmap = cv2.imread(heatmap_path)
    if heatmap is None:
        print(f"Could not load heatmap: {heatmap_path}")
        continue

    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_rgb, (224, 224))
    heatmap_gray = cv2.cvtColor(heatmap_resized, cv2.COLOR_RGB2GRAY)
    heatmap_norm = heatmap_gray.astype(np.float32) / 255.0

    sorted_indices = np.dstack(np.unravel_index(np.argsort(-heatmap_norm.ravel()), (224, 224)))[0]

    scores = [0.0]
    insertion_image = np.zeros_like(image_resized)
    pixels_per_step = (224 * 224) // steps

    for step in range(1, steps + 1):
        for i in range((step - 1) * pixels_per_step, step * pixels_per_step):
            if i >= len(sorted_indices): break
            y, x = sorted_indices[i]
            insertion_image[y, x] = image_resized[y, x]

        insertion_tensor = preprocess(insertion_image).unsqueeze(0)
        with torch.no_grad():
            out = model(insertion_tensor)
            score = torch.nn.functional.softmax(out, dim=1)[0, predicted_class].item()
            scores.append(score)

    np.save(os.path.join(output_dir, f'scores_insertion_{method_name}.npy'), scores)

    x_vals = np.linspace(0, 100, steps + 1)
    plt.figure(figsize=(4, 2.5), dpi=100)
    plt.plot(x_vals, scores, marker='o', label='Insertion curve')

    plt.title(f'Insertion Test: (AlexNet + RGB Heatmap)', fontsize=12)
    plt.xlabel("Pixels Inserted (%)", fontsize=10)
    plt.ylabel(f"Confidence for class {predicted_class}", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.grid(True)
    plt.legend(fontsize=9, loc='upper right')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'insertion_{method_name}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")

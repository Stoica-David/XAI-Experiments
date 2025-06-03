import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import os

CUR_DIR = os.getcwd()

image_path = os.path.join(CUR_DIR, r"datasets/test/dog.0.jpg")

heatmap_paths = [
    # GradCAM
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/gradCAM_PCA-fusion.png'),
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/gradCAM_simple-average.png'),
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/gradCAM_max-pooling.png'),
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/gradCAM_weighted-average.png'),

    # GradCAM++
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/gradCAM++_PCA-fusion.png'),
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/gradCAM++_simple-average.png'),
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/gradCAM++_max-pooling.png'),
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/gradCAM++_weighted-average.png'),

    # ScoreCAM
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/scoreCAM_PCA-fusion.png'),
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/scoreCAM_simple-average.png'),
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/scoreCAM_max-pooling.png'),
    os.path.join(CUR_DIR, r'_data/heatmaps/Feature attribution/scoreCAM_weighted-average.png'),
]

output_dir = os.path.join(CUR_DIR, r'_results/deletion-tests')
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
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_rgb, (224, 224))
    heatmap_gray = cv2.cvtColor(heatmap_resized, cv2.COLOR_RGB2GRAY)
    heatmap_norm = heatmap_gray.astype(np.float32) / 255.0

    sorted_indices = np.dstack(np.unravel_index(np.argsort(-heatmap_norm.ravel()), (224, 224)))[0]

    scores = [original_score]
    masked_image = image_resized.copy()
    pixels_per_step = (224 * 224) // steps

    for step in range(1, steps + 1):
        for i in range((step - 1) * pixels_per_step, step * pixels_per_step):
            if i >= len(sorted_indices): break
            y, x = sorted_indices[i]
            masked_image[y, x] = 0

        masked_tensor = preprocess(masked_image).unsqueeze(0)

        with torch.no_grad():
            out = model(masked_tensor)
            score = torch.nn.functional.softmax(out, dim=1)[0, predicted_class].item()
            scores.append(score)

    score_path = os.path.join(output_dir, f'scores_deletion_{method_name}.npy')
    np.save(score_path, np.array(scores))

    x_vals = np.linspace(0, 100, steps + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, scores, marker='o', label='Deletion curve')
    plt.title(f'Deletion Test: {method_name}')
    plt.xlabel("Pixels Removed (%)")
    plt.ylabel(f"Confidence for class {predicted_class}")
    plt.grid(True)
    plt.legend()

    image_save_path = os.path.join(output_dir, f'deletion_{method_name}.png')
    plt.savefig(image_save_path)
    plt.close()

    print(f"Saved: {image_save_path}")
    print(f"Saved: {score_path}")

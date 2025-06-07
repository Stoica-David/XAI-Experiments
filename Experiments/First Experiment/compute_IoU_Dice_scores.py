import numpy as np
import cv2
import os

from retrieve_project_root import project_root


image_path = os.path.join(project_root, r"_data/datasets/test/dog.0.jpg")
segmentation_mask_path = os.path.join(project_root, r'_data/dog_segmentation_mask.jpg')
heatmaps_dir = os.path.join(project_root, r'_data/heatmaps')

threshold_value = 0.5  # Threshold for Grad-CAM heatmap

# Load original image and segmentation mask
image = cv2.imread(image_path)
segmentation_mask_rgb = cv2.imread(segmentation_mask_path)

# Resize segmentation mask to match Grad-CAM size (assuming all Grad-CAMs are the same size)
example_heatmap = cv2.imread(os.path.join(heatmaps_dir, os.listdir(heatmaps_dir)[0]))
target_size = (example_heatmap.shape[1], example_heatmap.shape[0])  # width, height

segmentation_mask_resized = cv2.resize(segmentation_mask_rgb, target_size)
segmentation_mask_gray = cv2.cvtColor(segmentation_mask_resized, cv2.COLOR_BGR2GRAY)
segmentation_mask_binary = np.where(segmentation_mask_gray > 127, 255, 0).astype(np.uint8)

results = []

for fname in os.listdir(heatmaps_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        heatmap_path = os.path.join(heatmaps_dir, fname)
        grad_cam_rgb = cv2.imread(heatmap_path)
        grad_cam_gray = cv2.cvtColor(grad_cam_rgb, cv2.COLOR_BGR2GRAY)
        grad_cam_gray = cv2.normalize(grad_cam_gray, None, 0, 255, cv2.NORM_MINMAX)

        _, thresholded_grad_cam = cv2.threshold(grad_cam_gray, threshold_value * 255, 255, cv2.THRESH_BINARY)

        # Compute metrics
        intersection = np.sum(np.logical_and(thresholded_grad_cam == 255, segmentation_mask_binary == 255))
        union = np.sum(np.logical_or(thresholded_grad_cam == 255, segmentation_mask_binary == 255))
        iou = intersection / union if union != 0 else 0
        dice = 2 * intersection / (np.sum(thresholded_grad_cam == 255) + np.sum(segmentation_mask_binary == 255))

        results.append({
            'name': fname,
            'iou': iou,
            'dice': dice,
            'average': (iou + dice) / 2
        })

        print(f"{fname} => IoU: {iou:.4f} ({iou*100:.1f}%), Dice: {dice:.4f} ({dice*100:.1f}%)")

best_iou = max(results, key=lambda x: x['iou'])
best_dice = max(results, key=lambda x: x['dice'])
best_avg = max(results, key=lambda x: x['average'])

print("\n=== Best Grad-CAM Results ===")
print(f"→ By IoU:   {best_iou['name']} (IoU: {best_iou['iou']*100:.1f}%)")
print(f"→ By Dice:  {best_dice['name']} (Dice: {best_dice['dice']*100:.1f}%)")
print(f"→ Combined: {best_avg['name']} (Avg: {(best_avg['average'])*100:.1f}%)")

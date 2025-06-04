import os
import numpy as np
import matplotlib.pyplot as plt


project_root = os.path.dirname(os.path.abspath(__file__))

# Folder containing all saved .npy score files
input_dir = os.path.join(project_root, r'_results/deletion-tests')
output_dir = os.path.join(input_dir, 'averaged')
os.makedirs(output_dir, exist_ok=True)

steps = 20
x_vals = np.linspace(0, 100, steps + 1)

# Initialize buckets for scores
methods = {
    "gradCAM": [],
    "gradCAM++": [],
    "scoreCAM": []
}

# Load all .npy files and group by method
for fname in os.listdir(input_dir):
    if fname.startswith("scores_deletion_") and fname.endswith(".npy"):
        full_path = os.path.join(input_dir, fname)

        if "gradCAM++" in fname:
            methods["gradCAM++"].append(np.load(full_path))

        elif "scoreCAM" in fname:
            methods["scoreCAM"].append(np.load(full_path))

        elif "gradCAM" in fname:
            methods["gradCAM"].append(np.load(full_path))


for method, score_lists in methods.items():
    if not score_lists:
        print(f"No scores found for {method}")
        continue

    scores_array = np.stack(score_lists)
    mean_scores = scores_array.mean(axis=0)
    std_scores = scores_array.std(axis=0)

    np.save(os.path.join(output_dir, f"avg_scores_deletion_{method}.npy"), mean_scores)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, mean_scores, label=f'{method} (mean)', color='blue')
    plt.fill_between(x_vals, mean_scores - std_scores, mean_scores + std_scores,
                     alpha=0.2, color='blue', label='±1 std dev')
    plt.title(f"Averaged Insertion Curve - {method}")
    plt.xlabel("Pixels Inserted (%)")
    plt.ylabel("Confidence Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'avg_deletion_curve_{method}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Averaged insertion curve saved for {method}: {save_path}")

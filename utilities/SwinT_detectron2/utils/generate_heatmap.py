import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_class_distribution_heatmap(csv_path, output_dir, filename="class_distribution_heatmap.png"):
    """
    Reads a class distribution CSV and generates a heatmap saved to the output directory.

    Args:
        csv_path (str): Path to the CSV file containing batch-level class counts.
        output_dir (str): Directory to save the heatmap image.
        filename (str): Filename for the saved image.
    """
    if not os.path.exists(csv_path):
        print(f"[Heatmap] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Drop the 'iteration' column for heatmap plotting
    data_only = df.drop(columns=["iteration"], errors="ignore")

    plt.figure(figsize=(12, 6))
    sns.heatmap(data_only.T, cmap="Blues", annot=True, fmt="d")
    plt.xlabel("Batch Index")
    plt.ylabel("Class Name")
    plt.title("Class Distribution Per Batch")

    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    print(f"[Heatmap] Saved class distribution heatmap to: {heatmap_path}")

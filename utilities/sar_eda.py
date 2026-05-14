import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from tqdm import tqdm
import cv2

# --- CONFIGURATION ---
# TODO: User, please specify the path to your COCO JSON file and image directory.
JSON_PATH = "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/SAR/SHIPSv3/mySAR_Ship_dataset/annotations.json"
IMAGE_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/SAR/SHIPSv3/mySAR_Ship_dataset/images"
OUTPUT_DIR = "/media/gpaps/My Passport/CVRL-GeorgeP/_/Campaing/SAR/SHIPSv3/mySAR_Ship_dataset/sar_eda_outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MAIN SCRIPT ---

def load_data(json_path):
    """Loads COCO data and converts to pandas DataFrames."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images_df = pd.DataFrame(data['images'])
    ann_df = pd.DataFrame(data['annotations'])
    
    # Add image file paths
    if IMAGE_DIR:
        images_df['file_path'] = images_df['file_name'].apply(lambda x: os.path.join(IMAGE_DIR, x))

    # Merge annotations with image info
    df = pd.merge(ann_df, images_df, left_on='image_id', right_on='id', suffixes=('', '_img'))
    
    return df, images_df, data.get('categories', [])

def plot_bbox_distributions(df, output_dir):
    """Plots distributions of bbox width, height, and area."""
    
    df['bbox_width'] = df['bbox'].apply(lambda x: x[2])
    df['bbox_height'] = df['bbox'].apply(lambda x: x[3])
    df['bbox_area'] = df['bbox_width'] * df['bbox_height']
    
    # Width
    plt.figure(figsize=(10, 6))
    sns.histplot(df['bbox_width'], bins=50, kde=True)
    plt.title('Bounding Box Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.savefig(os.path.join(output_dir, 'bbox_width_distribution.png'))
    plt.close()

    # Height
    plt.figure(figsize=(10, 6))
    sns.histplot(df['bbox_height'], bins=50, kde=True)
    plt.title('Bounding Box Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.savefig(os.path.join(output_dir, 'bbox_height_distribution.png'))
    plt.close()

    # Area
    plt.figure(figsize=(10, 6))
    sns.histplot(df['bbox_area'], bins=50, kde=True)
    plt.title('Bounding Box Area Distribution')
    plt.xlabel('Area (pixels^2)')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'bbox_area_distribution.png'))
    plt.close()

def plot_ship_density(df, output_dir):
    """Plots the distribution of ships per image."""
    ship_counts = df['image_id'].value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(ship_counts, discrete=True)
    plt.title('Ship Density per Image')
    plt.xlabel('Number of Ships in Image')
    plt.ylabel('Number of Images')
    plt.savefig(os.path.join(output_dir, 'ship_density_per_image.png'))
    plt.close()
    
    return ship_counts

def analyze_object_size_ratio(df, output_dir):
    """Analyzes and plots the ratio of small, medium, and large objects."""
    
    # Using square root of area as a proxy for side length
    side_lengths = np.sqrt(df['bbox_area'])
    
    # Thresholds for SAR ships (these might need tuning)
    small_thr = 32  # side length < 32px
    medium_thr = 96 # 32px <= side length < 96px
    # large is >= 96px
    
    small_objects = side_lengths < small_thr
    medium_objects = (side_lengths >= small_thr) & (side_lengths < medium_thr)
    large_objects = side_lengths >= medium_thr
    
    size_counts = {
        'small': small_objects.sum(),
        'medium': medium_objects.sum(),
        'large': large_objects.sum()
    }
    
    plt.figure(figsize=(8, 8))
    plt.pie(size_counts.values(), labels=size_counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Small / Medium / Large Object Ratio')
    plt.savefig(os.path.join(output_dir, 'object_size_ratio.png'))
    plt.close()
    
    return size_counts, (small_thr, medium_thr)

def plot_image_resolutions(images_df, output_dir):
    """Plots image resolutions (width x height)."""
    if 'width' in images_df.columns and 'height' in images_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='width', y='height', data=images_df, alpha=0.5)
        plt.title('Image Resolutions Distribution')
        plt.xlabel('Image Width (pixels)')
        plt.ylabel('Image Height (pixels)')
        plt.savefig(os.path.join(output_dir, 'image_resolutions.png'))
        plt.close()

def analyze_image_brightness(images_df, output_dir):
    """Analyzes the brightness of SAR images."""
    brightness_vals = []
    
    print("Analyzing image brightness...")
    for idx, row in tqdm(images_df.iterrows(), total=len(images_df)):
        if 'file_path' in row and os.path.exists(row['file_path']):
            img = cv2.imread(row['file_path'], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                brightness_vals.append(np.mean(img))
    
    if brightness_vals:
        plt.figure(figsize=(10, 6))
        sns.histplot(brightness_vals, bins=50, kde=True)
        plt.title('Image Brightness (Mean Pixel Intensity) Distribution')
        plt.xlabel('Mean Brightness (0-255)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'image_brightness_distribution.png'))
        plt.close()
        return np.mean(brightness_vals), np.std(brightness_vals)
    return None, None

def generate_summary(df, images_df, ship_counts, size_counts, thresholds, brightness_stats, output_dir):
    """Generates a text summary of the EDA."""
    
    num_images = df['image_id'].nunique()
    num_annotations = len(df)
    
    summary_path = os.path.join(output_dir, 'eda_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("--- SAR Ship EDA Summary ---\n\n")
        f.write(f"Total Images Analyzed (with annotations): {num_images}\n")
        f.write(f"Total Images in Dataset: {len(images_df)}\n")
        f.write(f"Total Ship Annotations: {num_annotations}\n\n")
        
        f.write("--- Image Properties ---\n")
        if 'width' in images_df.columns and 'height' in images_df.columns:
             f.write("Image Widths:\n")
             f.write(images_df['width'].describe().to_string())
             f.write("\n\nImage Heights:\n")
             f.write(images_df['height'].describe().to_string())
             f.write("\n\n")
        
        if brightness_stats[0] is not None:
             f.write(f"Mean Image Brightness: {brightness_stats[0]:.2f}\n")
             f.write(f"Std Image Brightness: {brightness_stats[1]:.2f}\n\n")
        
        f.write("--- Bounding Box Statistics ---\n")
        f.write(df[['bbox_width', 'bbox_height', 'bbox_area']].describe().to_string())
        f.write("\n\n")
        
        f.write("--- Ship Density per Image ---\n")
        f.write(ship_counts.describe().to_string())
        f.write("\n\n")
        
        f.write("--- Object Size Ratios ---\n")
        small_thr, medium_thr = thresholds
        total_objects = sum(size_counts.values())
        if total_objects > 0:
            f.write(f"Small (< {small_thr}px side): {size_counts['small']} ({size_counts['small']/total_objects:.2%})\n")
            f.write(f"Medium ({small_thr}-{medium_thr}px side): {size_counts['medium']} ({size_counts['medium']/total_objects:.2%})\n")
            f.write(f"Large (>= {medium_thr}px side): {size_counts['large']} ({size_counts['large']/total_objects:.2%})\n\n")
        else:
            f.write("No objects found to analyze size ratios.\n\n")


        f.write("--- Notes ---\n")
        f.write("- 'offshore vs mixed/harbor split' is a complex task that likely requires more than just bounding box data (e.g., land masks, ship density clustering). A simple proxy is the 'Ship Density per Image' analysis.\n")
        f.write("- Image resolutions vary in this dataset, this can affect object size representation, checking the 'Image Resolutions Distribution' is recommended.\n")
        f.write("- Brightness distribution provides insight into the variance of SAR scattering intensities, which may relate to the sensors (SLED, SLEDF, SLEDP) and their configurations.\n")

def main():
    """Main function to run the EDA."""
    print("Starting SAR Ship EDA...")
    
    if not os.path.exists(JSON_PATH) or JSON_PATH == "path/to/your/coco_annotations.json":
        print(f"ERROR: Please update the JSON_PATH variable in this script to point to your COCO annotations file.")
        return

    df, images_df, categories = load_data(JSON_PATH)
    
    print("1. Plotting bounding box distributions...")
    plot_bbox_distributions(df, OUTPUT_DIR)
    
    print("2. Plotting ship density per image...")
    ship_counts = plot_ship_density(df, OUTPUT_DIR)
    
    print("3. Analyzing object size ratios...")
    size_counts, thresholds = analyze_object_size_ratio(df, OUTPUT_DIR)
    
    print("4. Analyzing Image Resolutions...")
    plot_image_resolutions(images_df, OUTPUT_DIR)
    
    print("5. Analyzing Image Brightness (this may take a while)...")
    brightness_stats = analyze_image_brightness(images_df, OUTPUT_DIR)
    
    print("6. Generating summary report...")
    generate_summary(df, images_df, ship_counts, size_counts, thresholds, brightness_stats, OUTPUT_DIR)
    
    print(f"\nEDA complete. Outputs are in the '{OUTPUT_DIR}' directory.")

if __name__ == '__main__':
    main()
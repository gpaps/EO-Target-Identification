import os
import glob
from PIL import Image


def convert_images_to_png(source_folder):
    # Supported image extensions
    image_extensions = ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp"]

    # Create output folder if needed (optional)
    os.makedirs(source_folder, exist_ok=True)

    # Loop over all extensions
    for ext in image_extensions:
        for filepath in glob.glob(os.path.join(source_folder, ext)):
            try:
                img = Image.open(filepath)
                img = img.convert("RGB")  # Optional: ensure 3-channel format
                png_path = os.path.splitext(filepath)[0] + ".png"
                img.save(png_path, "PNG")
                print(f"Converted: {filepath} -> {png_path}")
            except Exception as e:
                print(f"Error converting {filepath}: {e}")


# Example usage:
convert_images_to_png("/home/gpaps/PycharmProject/Esa_Ships/Bench_sat/Images_crop_256p/_new/")

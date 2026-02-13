import json
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ==== CONFIGURATION ====
# Map Original xView IDs to New Classes
# Target Classes: 0: Commercial, 1: Military, 2: Submarines, 3: Recreational, 4: Fishing

CLASS_MAPPING = {
    # --- Class 0: Commercial ---
    40: 0,  # Maritime Vessel (Generic - mapped per instruction)
    44: 0,  # Tugboat
    45: 0,  # Barge
    49: 0,  # Ferry
    51: 0,  # Container Ship
    52: 0,  # Oil Tanker

    # --- Class 3: Recreational Boats ---
    41: 3,  # Motorboat
    42: 3,  # Sailboat
    50: 3,  # Yacht

    # --- Class 4: Fishing Boats ---
    47: 4  # Fishing Vessel

    # Note: Classes 1 (Military) and 2 (Submarines) are intentionally
    # left out of this mapping as xView does not have specific labels for them.
    # They will be populated when you merge with VHRShips/ShipRSImageNET.
}

# Visualization Colors
TYPE_ID_COLORS = {
    0: 'blue',  # Commercial
    1: 'red',  # Military (Won't appear yet)
    2: 'purple',  # Submarines (Won't appear yet)
    3: 'orange',  # Recreational
    4: 'green',  # Fishing
    'default': 'grey'
}

# Class Names for Labeling
CLASS_NAMES = {
    0: 'Commercial',
    1: 'Military',
    2: 'Submarines',
    3: 'Recreational Boats',
    4: 'Fishing Boats'
}


def load_image(image_path):
    try:
        img = Image.open(image_path)
        return img, img.width, img.height
    except FileNotFoundError:
        # print(f"Warning: Image {image_path} not found.")
        return None, 0, 0
    except Exception as e:
        print(f"Error reading image {image_path}: {str(e)}")
        return None, 0, 0


def pixel_to_yolo(bounds_imcoords, img_width, img_height):
    try:
        x_min, y_min, x_max, y_max = map(int, bounds_imcoords.split(','))

        # Clamp to image dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width, x_max)
        y_max = min(img_height, y_max)

        # Validation: width and height must be > 0
        if x_max <= x_min or y_max <= y_min:
            return None

        # Calculate YOLO coordinates
        dw = 1. / img_width
        dh = 1. / img_height

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min

        x_center *= dw
        width = w * dw
        y_center *= dh
        height = h * dh

        return (x_center, y_center, width, height)
    except Exception as e:
        return None


def draw_yolo_bounding_boxes(image, yolo_coords_list, img_width, img_height):
    draw = ImageDraw.Draw(image)
    try:
        # Try to load a larger font for visibility
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for new_class_id, x_center, y_center, width, height in yolo_coords_list:
        color = TYPE_ID_COLORS.get(new_class_id, TYPE_ID_COLORS['default'])

        # Convert YOLO back to Pixel for drawing
        x_min = int((x_center - width / 2) * img_width)
        x_max = int((x_center + width / 2) * img_width)
        y_min = int((y_center - height / 2) * img_height)
        y_max = int((y_center + height / 2) * img_height)

        # Draw box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        # Draw Label Background
        class_name = CLASS_NAMES.get(new_class_id, str(new_class_id))
        draw.text((x_min, y_min - 15), class_name, fill=color, font=font)

    return image


def process_xview_data(geojson_path, image_dir, output_viz_dir, output_txt_dir):
    os.makedirs(output_viz_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    print("Loading GeoJSON...")
    with open(geojson_path) as f:
        geojson = json.load(f)

    # Group features by image
    image_annotations = {}
    for feature in tqdm(geojson['features'], desc="Parsing Features"):
        properties = feature['properties']
        image_id = properties['image_id']
        original_type_id = properties['type_id']
        bounds = properties['bounds_imcoords']

        # ==== KEY STEP: FILTER AND MAP CLASSES ====
        if original_type_id not in CLASS_MAPPING:
            continue  # Skip buses, random objects, etc.

        new_class_id = CLASS_MAPPING[original_type_id]

        if image_id not in image_annotations:
            image_annotations[image_id] = []

        image_annotations[image_id].append((new_class_id, bounds))

    print(f"Found {len(image_annotations)} images with relevant objects.")

    # Process Images
    for filename, annotations in tqdm(image_annotations.items(), desc="Processing Images"):
        # Handle cases where filename might have .tif or not
        base_name = os.path.splitext(filename)[0]
        # xView images usually come as .tif in the folder
        image_path = os.path.join(image_dir, f"{base_name}.tif")

        # If .tif doesn't exist, try .jpg (just in case)
        if not os.path.exists(image_path):
            image_path = os.path.join(image_dir, f"{base_name}.jpg")

        img, w, h = load_image(image_path)
        if img is None:
            continue

        valid_yolo_anns = []

        for class_id, bounds in annotations:
            yolo_box = pixel_to_yolo(bounds, w, h)
            if yolo_box:
                valid_yolo_anns.append((class_id, *yolo_box))

        if not valid_yolo_anns:
            continue

        # 1. Save YOLO .txt file
        txt_path = os.path.join(output_txt_dir, f"{base_name}.txt")
        with open(txt_path, 'w') as f:
            for cls, xc, yc, ww, hh in valid_yolo_anns:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        # 2. Save Visualization (Optional - slows down processing if too many)
        # Only saving first 50 for check to save time/space, remove limit if needed
        # if len(os.listdir(output_viz_dir)) < 50: 
        vis_img = draw_yolo_bounding_boxes(img.copy(), valid_yolo_anns, w, h)
        if vis_img.mode != 'RGB':
            vis_img = vis_img.convert('RGB')
        vis_path = os.path.join(output_viz_dir, f"{base_name}_vis.jpg")
        vis_img.save(vis_path)


if __name__ == '__main__':
    # UPDATE THESE PATHS
    GEOJSON_PATH = '/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Multiclass_dataset/xView[Annot-Yes][Extract_Geojson]/OG/train_labels/xView_train.geojson'
    IMAGE_DIR = '/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Multiclass_dataset/xView[Annot-Yes][Extract_Geojson]/OG/train_images/'

    OUTPUT_VIZ = './xview_vessels_vis'
    OUTPUT_TXT = './xview_vessels_labels'

    process_xview_data(GEOJSON_PATH, IMAGE_DIR, OUTPUT_VIZ, OUTPUT_TXT)

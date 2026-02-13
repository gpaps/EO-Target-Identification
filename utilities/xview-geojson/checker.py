import os
from PIL import Image
from tqdm import tqdm

# Your exact path
dataset_root = "/media/gpaps/My Passport/CVRL-GeorgeP/ESA_DATASET/Ships/_Optical/SuperDataset[clusterfucked]/Dataset/images/"
TINY_THRESH = 500  # Images smaller than 500x500
SMALL_THRESH = 800  # Images smaller than 800x800
# ----------------

try:
    all_files = os.listdir(dataset_root)
except FileNotFoundError:
    print(f"❌ Error: Path not found: {dataset_root}")
    exit()

# Filter for images
image_files = [os.path.join(dataset_root, f) for f in all_files if f.lower().endswith(('.jpg', '.png', '.bmp'))]
print(f"Scanning {len(image_files)} images...")

tiny_imgs = []  # < 500
small_imgs = []  # < 800
widths = []
heights = []

for img_path in tqdm(image_files):
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

            # Get filename from path
            filename = os.path.basename(img_path)

            # Check for Tiny (Outliers?)
            if w < TINY_THRESH or h < TINY_THRESH:
                tiny_imgs.append((filename, w, h))

            # Check for Small (Need resizing?)
            if w < SMALL_THRESH or h < SMALL_THRESH:
                small_imgs.append((filename, w, h))

    except Exception as e:
        print(f"Error reading {img_path}: {e}")

# --- REPORT ---
total = len(image_files)
print(f"\n==========================================")
print(f"📊 DATASET SIZE AUDIT")
print(f"==========================================")
print(f"Total Images: {total}")

if total > 0:
    print(f"\n🔴 TINY IMAGES (< {TINY_THRESH}px): {len(tiny_imgs)} ({len(tiny_imgs) / total * 100:.2f}%)")
    if len(tiny_imgs) > 0:
        print("   -> Sample Filenames:")
        for name, w, h in tiny_imgs[:5]:
            print(f"      - {name} ({w}x{h})")
        if len(tiny_imgs) > 5: print(f"      ... and {len(tiny_imgs) - 5} more.")

    print(f"\n🟡 SMALL IMAGES (< {SMALL_THRESH}px): {len(small_imgs)} ({len(small_imgs) / total * 100:.2f}%)")

    print(f"\n✅ NATIVE STATS:")
    if widths:
        print(f"   Min: {min(widths)}x{min(heights)}")
        print(f"   Max: {max(widths)}x{max(heights)}")
        print(f"   Avg: {int(sum(widths) / len(widths))}x{int(sum(heights) / len(heights))}")

print(f"==========================================")

# --- DECISION LOGIC ---
if len(tiny_imgs) < 100:
    print("\n💡 VERDICT: OUTLIERS.")
    print("   Action: DELETE these few images and stick to standard resolution (1024, 1200).")
else:
    print("\n💡 VERDICT: SIGNIFICANT CLUSTER.")
    print("   Action: Use 'Resolution Normalization' (900-1200) to upscale them.")
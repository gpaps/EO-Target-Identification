import cv2


def downsample_image(image_path, scale_factor=2, output_path="downsampled_image_006_2.jpg"):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")

    # Calculate new dimensions
    new_width = int(image.shape[1] / scale_factor)
    new_height = int(image.shape[0] / scale_factor)

    # Resize image
    downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save output
    cv2.imwrite(output_path, downsampled)
    print(f"Image downsampled and saved to {output_path}")


# Example usage
# downsample_image("/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/VHRShips_ShipRSImageNEt/000006.bmp", scale_factor=2)


def resample_image(image_path, sigma=1.8, output_path="resampled_image_GAUS_017.jpg"):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")

    # Apply Gaussian blur (resampling with kernel influence)
    # Kernel size is auto-calculated from sigma by OpenCV
    resampled = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

    # Save result
    cv2.imwrite(output_path, resampled)
    print(f"Image resampled with sigma={sigma} and saved to {output_path}")


# Example usage
resample_image("/home/gpaps/Documents/CVRL Projects/ESA/ESA-Datasets/1_Ships/_Optical-Ship/VHRShips_ShipRSImageNEt"
               "/000017.bmp", sigma=1)

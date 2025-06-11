import os
import albumentations as A
from tqdm import tqdm
import cv2
import numpy as np 
from PIL import Image

def augment_image(image):
    """
    Apply augmentations to the image.
    """
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0), contrast_limit=0, p=1.0),  # Darken the image
        A.Blur(blur_limit=(7, 15), p=0.5),
        A.MotionBlur(blur_limit=(7, 15), p=0.9),
    ])
    augmented = transform(image=image)
    return augmented['image']

def check_image_okay(rgb_img, seg_img, min_tag_area=1000, min_tag_pix_mean=25, max_tag_pix_mean=250):
    """
    Check if the image is valid based on segmentation and RGB image properties.
    """
    if rgb_img is None or seg_img is None:
        return False

    seg_img = np.array(seg_img)
    # Compute pixel area of tag segmentation
    tag_pix_area = np.sum(seg_img == 255)

    # Create list of marker pixels using segmentation
    marker_pixels = np.argwhere(seg_img == 255)  # Get the indices of pixels where the tag is present

    # Compute contrast of marker pixels using RGB image
    rgb_img = np.array(rgb_img, dtype=np.float32)  # Ensure RGB image is in float32 format
    if rgb_img.max() <= 1.0:
        rgb_img *= 255.0
    marker_rgb_values = rgb_img[marker_pixels[:, 0], marker_pixels[:, 1]]  # Get the RGB values of the marker pixels
    marker_grey_values = np.mean(marker_rgb_values, axis=1)  # Compute the mean RGB values of the marker pixels

    # Compute contrast as the difference in magnitude of the RGB values of the marker pixels
    tag_pix_contrast = marker_grey_values.max() - marker_grey_values.min()
    tag_pix_mean = marker_grey_values.mean()

    # Check thresholds for tag area, mean pixel value, and contrast
    if tag_pix_area > min_tag_area and min_tag_pix_mean < tag_pix_mean < max_tag_pix_mean:
        return True
    return False

def get_roi_image(rgb, seg, roi_size=128, padding=5): 

    image_border_size = np.max([np.array(seg).shape[0], np.array(seg).shape[1]]) 

    # get pixel info of seg 
    seg = np.array(seg) 
    seg = cv2.copyMakeBorder(seg, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0) 
    tag_pixels = np.argwhere(seg == 255)
    seg_tag_min_x = np.min(tag_pixels[:,1])
    seg_tag_max_x = np.max(tag_pixels[:,1])
    seg_tag_min_y = np.min(tag_pixels[:,0])
    seg_tag_max_y = np.max(tag_pixels[:,0])
    seg_height = seg_tag_max_y - seg_tag_min_y  
    seg_width = seg_tag_max_x - seg_tag_min_x 
    seg_center_x = (seg_tag_min_x + seg_tag_max_x) // 2
    seg_center_y = (seg_tag_min_y + seg_tag_max_y) // 2 

    # get pixel info of rgb 
    rgb = cv2.copyMakeBorder(rgb, image_border_size, image_border_size, image_border_size, image_border_size, cv2.BORDER_CONSTANT, value=0) 
    rgb_side = max(seg_height, seg_width) + 2*padding 
    rgb_tag_min_x = seg_center_x - rgb_side // 2
    rgb_tag_max_x = seg_center_x + rgb_side // 2
    rgb_tag_min_y = seg_center_y - rgb_side // 2
    rgb_tag_max_y = seg_center_y + rgb_side // 2
    roi_img = rgb[rgb_tag_min_y:rgb_tag_max_y, rgb_tag_min_x:rgb_tag_max_x, :]

    # resize rgb bbox to roi size         
    try: 
        roi_img = cv2.resize(roi_img, (roi_size, roi_size))
    except: 
        print("error resizing") 
        import pdb; pdb.set_trace() 

    return roi_img

def main():
    """
    Main function to perform data re-augmentation.
    """
    # Directory containing the original augmented data
    dir_data = "./segmentation_model/data/data_20250330-013534_reaugmented/"
    max_augmentation_attempts = 100

    # Create directories
    train_rgb_dir = os.path.join(dir_data, "train", "rgb")
    train_seg_dir = os.path.join(dir_data, "train", "seg")
    val_rgb_dir = os.path.join(dir_data, "val", "rgb")
    val_seg_dir = os.path.join(dir_data, "val", "seg")

    list_image_directories = [
        (train_rgb_dir, train_seg_dir),
        (val_rgb_dir, val_seg_dir),
    ]
    for rgb_dir, seg_dir in list_image_directories:

        rgb_augmented_dir = rgb_dir + "_augmented"
        if not os.path.exists(rgb_augmented_dir):
            os.makedirs(rgb_augmented_dir)

        roi_augmented_dir = rgb_dir + "_roi_augmented"
        if not os.path.exists(roi_augmented_dir):
            os.makedirs(roi_augmented_dir)

        # List all images in the RGB directory
        image_files = [f for f in os.listdir(rgb_dir) if f.endswith(".png") or f.endswith(".jpg")]

        # Process images with a progress bar
        for filename in tqdm(image_files, desc=f"Augmenting images in {rgb_dir}"):

            rgb_image_path = os.path.join(rgb_dir, filename)
            seg_image_path = os.path.join(seg_dir, filename.replace("img","seg").replace("_0",""))

            # Read RGB and segmentation images
            rgb_img = cv2.imread(rgb_image_path)
            seg_img = cv2.imread(seg_image_path, cv2.IMREAD_GRAYSCALE)  # Read segmentation as grayscale

            # Convert RGB image to RGB format for albumentations
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            # Augment the image
            augmented_image = rgb_img.copy() * 0 
            loop_count = 0
            while not check_image_okay(augmented_image, seg_img):
                # Apply augmentations
                augmented_image = augment_image(rgb_img)
                loop_count += 1
                if loop_count > max_augmentation_attempts:
                    print(f"Skipping augmentation of {filename} after {max_augmentation_attempts} attempts to augment.")
                    break
            if loop_count > max_augmentation_attempts:
                augmented_image = rgb_img

            # Convert back to BGR for saving
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

            # Save the augmented image
            augmented_image_path = os.path.join(rgb_augmented_dir, filename)
            cv2.imwrite(augmented_image_path, augmented_image)

            # Get ROI image
            if seg_img is not None and rgb_img is not None:
                roi_image = get_roi_image(rgb_img, seg_img)
            else: 
                # delete the augmented image if ROI cannot be created
                os.remove(augmented_image_path)
                print(f"Skipping {filename} due to missing segmentation or RGB image.")
                continue 
            roi_image_path = os.path.join(roi_augmented_dir, filename.replace("img","roi").replace("_0",""))
            cv2.imwrite(roi_image_path, roi_image)

if __name__ == "__main__":
    main()
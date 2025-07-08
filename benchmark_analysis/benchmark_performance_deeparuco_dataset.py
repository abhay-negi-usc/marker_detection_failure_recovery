import os
from PIL import Image
import numpy as np
import json 
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
from torchvision import transforms

from benchmark_analysis.utils import list_filetype_alphanumeric_order 
from segmentation_model.utils import load_checkpoint as load_seg_ckpt
from segmentation_model.model import UNETWithDropout
from keypoints_model.utils import overlay_points_on_image 


class datapoint(): 
    def __init__(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
        self.get_idx() 

    def get_idx(self): 
        filename = "0000.png"
        idx = self.image_path.split(".")[0]
        self.idx = idx 
        return idx 

    def get_image(self): 
        """
        Returns the image as an np array 
        """
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        image = Image.open(self.image_path)
        image_np = np.array(image)
        image.close() 
        return image_np

    def get_label(self):
        """
        Returns the label as a dictionary loaded from the JSON file.
        """
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label file not found: {self.label_path}")
        with open(self.label_path, 'r') as f:
            label_data = json.load(f)
        self.label_data = label_data['markers']  # Store label data for later use
        return self.label_data 

    def get_corners_and_ids(self): 
        self.corners = [] 
        self.ids = [] 
        if not hasattr(self, 'label_data'):
            self.get_label()
        for marker in self.label_data:
            corners = marker.get('corners', [])
            if corners:
                corners_np = np.empty((4,2))
                for i, corner in enumerate(corners): 
                    corners_np[i, 0] = corner['x']
                    corners_np[i, 1] = corner['y']
                self.corners.append(corners_np)
                self.ids.append(marker.get('id', None))

    def __str__(self):
        return f"Image: {self.image_path}, Label: {self.label_path}"

    def __repr__(self):
        return self.__str__()   

class benchmark_dataset(): 
    def __init__(self, dir_images, dir_annotations):
        self.dir_images = dir_images
        self.dir_annotations = dir_annotations
        self.datapoints = self.get_datapoints()

    def get_datapoints(self):
        image_files = list_filetype_alphanumeric_order(self.dir_images, ".png")
        label_files = list_filetype_alphanumeric_order(self.dir_annotations, ".json")
        if len(image_files) != len(label_files):
            raise ValueError("Number of images and labels do not match.")
        datapoints = []
        for img_file, label_file in zip(image_files, label_files):
            img_path = os.path.join(self.dir_images, img_file)
            label_path = os.path.join(self.dir_annotations, label_file)
            datapoints.append(datapoint(img_path, label_path))
        return datapoints
     
class estimator(): 
    def __init__(self, segmentation_model_path):
        self.segmentation_model_path = segmentation_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.segmentation_model_path):
            raise FileNotFoundError(f"Segmentation model file not found: {self.segmentation_model_path}")
        self.seg_model = UNETWithDropout(in_channels=3, out_channels=1).to(self.device)
        load_seg_ckpt(torch.load(self.segmentation_model_path, map_location=self.device), self.seg_model)
        self.seg_model.eval()

    def segment_image(self, image_np):
        """
        Run the image through the segmentation model and return the segmentation mask.
        """
        if not hasattr(self, 'seg_model'):
            raise RuntimeError("Segmentation model is not loaded.")
        
        seg_size = (640, 480)

        # Resize RGB image to match segmentation input
        resized_rgb = cv2.resize(image_np, seg_size)  # shape (H, W, 3)

        # Segmentation transform: normalized for model
        seg_transform = A.Compose([
            A.Normalize(max_pixel_value=1.0),
            ToTensorV2()
        ])
        transformed = seg_transform(image=resized_rgb)
        img_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            seg_mask = torch.sigmoid(self.seg_model(img_tensor))
            seg_mask = (seg_mask > 0.5).float().cpu().numpy()

        seg_mask = seg_mask.squeeze(0).squeeze()  # Remove batch dimension

        # seg_mask_img = transforms.ToPILImage()(seg_mask.squeeze(0))  # shape matches resized_rgb
        # return seg_mask_img 

        return seg_mask 
        
        # # Convert image_np to tensor and run through model
        # image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).to(self.device)
        # if image_tensor.shape[3] == 3:  # Check if the image has 3 channels
        #     image_tensor = image_tensor.permute(0, 3, 1, 2)  # Change to (batch_size, channels, height, width)
        # else:
        #     raise ValueError("Image must have 3 channels (RGB).")   
        
        # with torch.no_grad():
        #     self.seg_model.eval()
        #     segmentation_mask = self.seg_model(image_tensor)  # Run the model
        #     segmentation_mask = torch.sigmoid(segmentation_mask)  # Apply sigmoid to get probabilities
        #     segmentation_mask = (segmentation_mask > 0.1).float()  # Threshold
        #     segmentation_mask = segmentation_mask.squeeze(0).cpu().numpy()  # Remove batch dimension
        # segmentation_mask = segmentation_mask.squeeze()  # Remove channel dimension if present
        # return segmentation_mask


# List all image and label files 
# loop through each file in dir 
# read image and corresponding annotation  
# store: pose, corners 
# run image through segmentaion model 
# compute corners from segmentation mask 
# compare corners errors 
# compute true seg from true corners 
# compare IOU error 
# output summary image: 
#    - true image with true segmentation and true corners 
#    - predicted image with predicted segmentation and predicted corners 
#    - idx, pose error, corner error, IOU value in title of summary image 

def get_datapoints(dir_images, dir_annotations):
    """
    Get a list of datapoints with image and label paths.
    """
    image_files = list_filetype_alphanumeric_order(dir_images, ".jpg")
    label_files = list_filetype_alphanumeric_order(dir_annotations, ".json")

    if len(image_files) != len(label_files):
        raise ValueError("Number of images and labels do not match.")

    datapoints = []
    for img_file, label_file in zip(image_files, label_files):
        img_path = os.path.join(dir_images, img_file)
        label_path = os.path.join(dir_annotations, label_file)
        datapoints.append(datapoint(img_path, label_path))

    return datapoints

def main():

    video_idx = 6 
    config = {
        'dir_images': f"/home/anegi/abhay_ws/deeparuco/datasets/10791293/video_{video_idx}/",  
        'dir_annotations': f"/home/anegi/abhay_ws/deeparuco/datasets/10791293/video_{video_idx}/corrected_annotations/",  
        'segmentation_model_path': "/home/nom4d/marker_ws/segmentation_checkpoints/my_checkpoint_multimarker_epoch_2_batch_30000.pth.tar",
    }

    # Initialize the dataset
    dataset = benchmark_dataset(config['dir_images'], config['dir_annotations'])
    print(f"Number of datapoints: {len(dataset.datapoints)}")

    # Initialize the estimator with the segmentation model
    estimator_model = estimator(config['segmentation_model_path'])
    estimator_model.load_model()  # Ensure the model is loaded

    for i, dp in enumerate(dataset.datapoints):
        print(f"Datapoint {i}: {dp}")
        # Get image and label
        image_np = dp.get_image()
        label_data = dp.get_label()
        dp.get_corners_and_ids()  # Call to get corners, currently does nothing

        segmentation_mask = estimator_model.segment_image(image_np) 
        if segmentation_mask.max() > 0:  
            # show segmentation prediction 
            plt.imshow(segmentation_mask, cmap='gray')
            plt.title(f"Datapoint {i} - Segmentation Mask")
            plt.axis('off')
            plt.show()
        else: 
            print(f"No segmentation detected for datapoint {i}. Skipping...")

        # show image_np with corners overlaid

        # overlay_points_on_image(image_np, dp.corners) 
        if i > 50:
            break
    

if __name__ == "__main__":
    main()


import os 
from PIL import Image 
from torch.utils.data import Dataset 
import numpy as np 
from torchvision import transforms
from os.path import splitext, isfile, join
import torch
import json 

convert_tensor = transforms.ToTensor()

def load_image(filename):
    ext = splitext(filename)[1]
    # import pdb; pdb.set_trace()
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

from utils import generate_heatmap

class MarkersDataset(Dataset):
    def __init__(self, image_dir, keypoints_dir, transform=None, heatmap_size=64):
        self.image_dir = image_dir
        self.keypoints_dir = keypoints_dir
        self.transform = transform or transforms.ToTensor()
        self.images = os.listdir(image_dir)
        self.heatmap_size = heatmap_size

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        img_filename = os.path.basename(img_path)
        keypoints_filename = img_filename.replace("_", "_keypoints_").replace(".png", ".json")
        keypoints_path = os.path.join(self.keypoints_dir, keypoints_filename)

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        image_tensor = self.transform(image)

        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        keypoints_list = [np.array(v) for v in keypoints_data.values()]
        keypoints = np.stack(keypoints_list)  # shape (K, 2)

        heatmaps = generate_heatmap(keypoints, img_size=(w, h), heatmap_size=self.heatmap_size)
        heatmaps = torch.from_numpy(heatmaps)

        return image_tensor, heatmaps



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

class MarkersDataset(Dataset):
    def __init__(self, image_dir, keypoints_dir, transform=None):
        self.image_dir = image_dir
        self.keypoints_dir = keypoints_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        img_filename = os.path.basename(img_path)

        # keypoints_filename = img_filename.replace("_","_keypoints_").replace(".png",".json") 
        # keypoints_filename = keypoints_filename.replace('img', 'keypoints')  

        keypoints_filename = img_filename.replace("img","keypoints").replace(".png",".json") 
        
        keypoints_path = os.path.join(
            self.keypoints_dir, 
            keypoints_filename
        )

        # Load image and convert to numpy array
        image = np.array(Image.open(img_path).convert("RGB"))
        # Load keypoints from json as np array
        # TODO: preprocess json's as npy files  
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f) 
        keypoints_list = [] 
        for key in keypoints_data.keys():
            keypoints_list.append(np.array(keypoints_data[key])) 
        keypoints = np.array(keypoints_list).flatten() 

        return image, keypoints

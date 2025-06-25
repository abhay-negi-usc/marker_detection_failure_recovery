import os 
from PIL import Image 
from torch.utils.data import Dataset 
import numpy as np 
from torchvision import transforms
from os.path import splitext, isfile, join
import torch

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
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        img_filename = os.path.basename(img_path) 
        # replace all characters after last underscore with '.png' 
        # mask_filename = img_filename[:img_filename.rfind('_')] + '.png' 
        # mask_filename = mask_filename.replace('img', 'seg')  
        mask_filename = img_filename.replace('img', 'seg')  
        mask_path = os.path.join(
            self.mask_dir, 
            mask_filename
        )

        # Load image and convert to numpy array
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load mask and convert to binary
        mask = np.array(
            Image.open(mask_path).convert("L"), 
            dtype=np.float32
        )
        
        # Convert mask to binary (0 and 1)
        mask = np.where(mask >= 100.0, 1.0, 0.0)

        # Apply transforms if specified
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask



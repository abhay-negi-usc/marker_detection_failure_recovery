# imports 
import numpy as np 
import torch
from torchvision import transforms
from PIL import Image
import os 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 
import time 
import sys 

# module imports 
sys.path.append('C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery')
from marker_detection_failure_recovery.utils.utils import *
from utils.utils import * 
from segmentation_model.utils import * 
from pose_estimation_model.utils import * 

## DATA PROCESSING 
# read data 
sim_data_processor = DataProcessor(
    data_folders = ["C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/segmentation_model/sim_data/markers_20250314-181037"], 
    out_dir = "C:/Users/NegiA/Desktop/abhay_ws/marker_detection_failure_recovery/segmentation_model/sim_data/markers_20250314-181037/outputs"
)

# data processing 

import pdb; pdb.set_trace() 

## INFERENCE 
# run segmentation model 
# compute detection accuracy (IOU) 
# run optimization pose estimation 
# run classical pose estimation 
# save data 

## ANALYSIS 
# compute pose errors 
# bar chart comparing accuracy 
# violin plots comparing pose error distributions 
# scatter plots comparing error vs variables (distance, lighting, etc.) 
# save data 

# ~/.local/share/ov/pkg/isaac-sim-4.5.0/python.sh synthetic_data_generation/debug_script.py 

import argparse
import json
import os

import yaml
from isaacsim import SimulationApp
import time 
import asyncio
from PIL import Image
import numpy as np 

import sys 
from scipy.spatial.transform import Rotation as R 


# import sdg_utils 
timestr = time.strftime("%Y%m%d-%H%M%S") 
print(os.getcwd())
if os.getcwd() == '/home/anegi/abhay_ws/marker_detection_failure_recovery': # isaac machine 
    OUT_DIR = os.path.join("/media/anegi/easystore/abhay_ws/marker_detection_failure_recovery/output","markers_"+timestr)
    dir_textures = "/home/anegi/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags" 
    sys.path.append("/home/anegi/.local/share/ov/pkg/isaac-sim-4.5.0/standalone_examples/replicator/object_based_sdg")
    dir_backgrounds = "/media/anegi/easystore/abhay_ws/marker_detection_failure_recovery/background_images" 
else: # CAM machine 
    OUT_DIR = os.path.join("/media/rp/Elements/abhay_ws/marker_detection_failure_recovery/data/marker_obj_sdg/","markers_"+timestr) 
    dir_textures = "/home/rp/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags"
    sys.path.append("/home/rp/.local/share/ov/pkg/isaac-sim-4.5.0/standalone_examples/replicator/object_based_sdg")
    dir_backgrounds = "/media/rp/Elements/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/background_images" 

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"rgb"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"seg"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"pose"), exist_ok=True) 
tag_textures = [os.path.join(dir_textures, f) for f in os.listdir(dir_textures) if os.path.isfile(os.path.join(dir_textures, f))] 

# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {
        "renderer": "RayTracedLighting",
        "headless": False,
    },
    "env_url": "",
    "working_area_size": (4, 4, 3),
    "rt_subdistractors_num": 0,
}

import carb

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
else:
    carb.log_warn(f"File {args.config} does not exist, will use default config")

# Update the default config dict with the external one
config.update(args_config)

print(f"[SDG] Using config:\n{config}")

launch_config = config.get("launch_config", {})
simulation_app = SimulationApp(launch_config=launch_config)

import random
import time
from itertools import chain

import carb.settings

# Custom util functions for the example
import sys 
# sys.path.append("/home/rp/.local/share/ov/pkg/isaac-sim-4.2.0/standalone_examples/replicator/object_based_sdg")
import object_based_sdg_utils  
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import usdrt
# from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
from isaacsim.core.utils.semantics import add_update_semantics, remove_all_semantics
from omni.isaac.nucleus import get_assets_root_path
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics
from pxr import Usd, UsdShade, Gf


def get_world_transform_xform(prim: Usd.Prim):
    """
    Get the local transformation of a prim using Xformable.
    See https://openusd.org/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    return translation, rotation, scale

cam = rep.create.camera(
    position=(10,10,10)
)  
rp_cam = rep.create.render_product(cam, (640, 480)) 
cam_prim = cam.get_output_prims()["prims"][0] 

print(get_world_transform_xform(cam_prim)) 

with cam: 
    rep.modify.pose(
        position = (1,2,3) 
    )

print(get_world_transform_xform(cam_prim)) 

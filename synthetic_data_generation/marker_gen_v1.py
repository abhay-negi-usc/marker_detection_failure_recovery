# run using the following command: 
# ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh synthetic_data_generation/marker_gen_v1.py 

# fixed marker, random camera position, camera orientation determined to point at marker, random lighting, random aluminum bars in background 
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid 
import numpy as np

from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
import omni
import quaternion 

from pxr import Gf, Usd, UsdGeom, Sdf, UsdLux
import omni.replicator.core as rep
import os 
from PIL import Image
from omni.replicator.core import AnnotatorRegistry, Writer
from omni.isaac.nucleus import get_assets_root_path
from scipy.spatial.transform import Rotation as R 

import time
import random 
import carb
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage

MAIN_PATH = "/home/rp/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/" # FIXME: make this a relative path (?) or read from a file (?)  
SAVE_PATH = "/media/rp/Elements/abhay_ws/marker_detection_failure_recovery/data"
timestr = time.strftime("%Y%m%d-%H%M%S") 
OUT_PATH  = os.path.join(SAVE_PATH, "output", timestr)      
os.makedirs(OUT_PATH, exist_ok=True) 

ENV_URL = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"


# Save rgb image to file
def save_rgb(rgb_data, file_name):
    rgb_img = Image.fromarray(rgb_data, "RGBA")
    rgb_img.save(file_name + ".png")

# Access data through a custom replicator writer
class MyWriter(Writer):
    def __init__(self, rgb: bool = True):
        self._frame_id = 0
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))

        # Create writer output directory
        self.file_path = os.path.join(OUT_PATH, "_out_mc_writer", "") 
        print(f"Writing writer data to {self.file_path}")
        dir = os.path.dirname(self.file_path)
        os.makedirs(dir, exist_ok=True)

    def write(self, data):
        for annotator in data.keys():
            annotator_split = annotator.split("-")
            if 'render_product_name' not in locals(): 
                render_product_name = 'default_name'
            if len(annotator_split) > 1:
                render_product_name = annotator_split[-1]
            if annotator.startswith("rgb"):
                save_rgb(data[annotator], f"{self.file_path}/{render_product_name}_frame_{self._frame_id}")
        self._frame_id += 1

def quatd_to_euler(quat): 
    q0 = quat.GetReal() 
    q1 = quat.GetImaginary()[0] 
    q2 = quat.GetImaginary()[1] 
    q3 = quat.GetImaginary()[2] 
    eul = R.from_quat([q1,q2,q3,q0]).as_euler('xyz', degrees=True)  
    return eul 

def sphere_lights(num):
    lights = []
    for i in range(num):
        # "CylinderLight", "DiskLight", "DistantLight", "DomeLight", "RectLight", "SphereLight"
        prim_type = "DistantLight"
        next_free_path = omni.usd.get_stage_next_free_path(stage, f"/World/{prim_type}", False)
        light_prim = stage.DefinePrim(next_free_path, prim_type)
        # UsdGeom.Xformable(light_prim).AddTranslateOp().Set((0.0, 0.0, 0.0))
        UsdGeom.Xformable(light_prim).AddTranslateOp().Set((0.0, 0.0, 10.0))
        UsdGeom.Xformable(light_prim).AddRotateXYZOp().Set((0.0, 0.0, 0.0))
        UsdGeom.Xformable(light_prim).AddScaleOp().Set((1.0, 1.0, 1.0))
        light_prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)
        # light_prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(6500.0)
        light_prim.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(0.5)
        light_prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(30000.0)
        light_prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
        light_prim.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(0.0)
        light_prim.CreateAttribute("inputs:diffuse", Sdf.ValueTypeNames.Float).Set(1.0)
        light_prim.CreateAttribute("inputs:specular", Sdf.ValueTypeNames.Float).Set(1.0)
        lights.append(light_prim)
    return lights

def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception("Nucleus server not found, could not access Isaac Sim assets folder")
    return assets_root_path + relative_path

print(f"Loading Stage {ENV_URL}")
open_stage(prefix_with_isaac_asset_server(ENV_URL))

# omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

rep.WriterRegistry.register(MyWriter)

rand_color = np.random.rand(3) 
rand_angles = ((np.random.rand(3) - 0.5) * 2) * np.pi # range: -pi to +pi  
rand_q = quaternion.from_euler_angles(rand_angles)
fixed_q = np.array([1,0,0,0])
assets_root_path = get_assets_root_path()

MARKER_PATH = MAIN_PATH + "/assets/usd/tag0_v2.usd" 
marker_prim = stage.DefinePrim("/World/marker", "Xform")
marker_prim.GetReferences().AddReference(MARKER_PATH)
if not marker_prim.GetAttribute("xformOp:translate"):
    UsdGeom.Xformable(marker_prim).AddTranslateOp()
if not marker_prim.GetAttribute("xformOp:rotateXYZ"):
    UsdGeom.Xformable(marker_prim).AddRotateXYZOp()
marker_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0,0,-10)) 
marker_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(90,0,0))

# Create cameras
camera_prim1 = stage.DefinePrim("/World/Camera1", "Camera")
UsdGeom.Xformable(camera_prim1).AddTranslateOp().Set((0.0, 10.0, 20.0))
UsdGeom.Xformable(camera_prim1).AddRotateXYZOp().Set((0.0, 0.0, 0.0))

# Create render products
rp1 = rep.create.render_product(str(camera_prim1.GetPrimPath()), resolution=(1024, 1024))

# Acess the data through a custom writer
writer = rep.WriterRegistry.get("MyWriter")
writer.initialize(rgb=True)
writer.attach([rp1])

# Acess the data through annotators
rgb_annotators = []
for rp in [rp1]:
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    rgb_annotators.append(rgb)

# Create annotator output directory
dir = os.path.join(OUT_PATH, "_out_mc_annotator") 
os.makedirs(dir, exist_ok=True)  
print(f"Writing annotator data to {dir}")

# Data will be captured manually using step
rep.orchestrator.set_capture_on_play(False)

# add lights 
lights = sphere_lights(1)

for i in range(10):

    for light in lights:
        light.GetAttribute("inputs:intensity").Set(np.random.uniform(0, 2000)) 
        rand_light_angle = ((np.random.rand(3) - 0.5) * 2) * 30  
        light.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(rand_light_angle[0], rand_light_angle[1], rand_light_angle[2])) 

    rand_cam_position = (2*(np.random.rand()-0.5)*1, 50*(np.random.rand()-0.5)*1, 10) 

    marker_pos = omni.usd.get_world_transform_matrix(marker_prim).ExtractTranslation()
    cam_pos = rand_cam_position 
    camera_prim1.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*cam_pos))

    eye = Gf.Vec3d(*cam_pos)
    target = Gf.Vec3d(*marker_pos)
    up_axis = Gf.Vec3d(0, 0, 1)
    look_at_quatd = Gf.Matrix4d().SetLookAt(eye, target, up_axis).GetInverse().ExtractRotation().GetQuat()
    euler_angles_clean = quatd_to_euler(look_at_quatd) 
    euler_angles_noise = 2*(np.random.rand(3)-0.5) * 10 
    cam_euler_angles = euler_angles_clean + euler_angles_noise
    camera_prim1.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(cam_euler_angles[0], cam_euler_angles[1], cam_euler_angles[2]))

    rep.orchestrator.step(rt_subframes=4)
    for j, rgb_annot in enumerate(rgb_annotators): 
        save_rgb(rgb_annot.get_data(), f"{dir}/rp{j}_step_{i}") 

simulation_app.close() # close Isaac Sim
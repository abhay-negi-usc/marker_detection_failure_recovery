# ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh synthetic_data_generation/marker_obj_sdg.py 

# TODO: 
# output pose data corresponding to each image AND segmentation image 
# add variable light sources: dome light, directional light, point light, spot light 
# add distractors: mesh, shape, texture 
# add distractor randomization: color, texture, position, rotation, scale 
# add distractor physics: floating, falling, bouncing 

import argparse
import json
import os

import yaml
from isaacsim import SimulationApp
import time 
import asyncio
from PIL import Image
import numpy as np 

# import sdg_utils 
timestr = time.strftime("%Y%m%d-%H%M%S") 
OUT_DIR = os.path.join("/media/rp/Elements/abhay_ws/marker_detection_failure_recovery/data/marker_obj_sdg/","markers_"+timestr) 
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"rgb"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"seg"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR,"pose"), exist_ok=True) 
dir_textures = "/home/rp/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags" 
tag_textures = [os.path.join(dir_textures, f) for f in os.listdir(dir_textures) if os.path.isfile(os.path.join(dir_textures, f))] 

# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {
        "renderer": "RayTracedLighting",
        "headless": True,
    },
    "env_url": "",
    "working_area_size": (4, 4, 3),
    "rt_subframes": 4,
    "num_frames": 100,
    "num_cameras": 1,
    "camera_collider_radius": 0.5,
    "disable_render_products_between_captures": False,
    "simulation_duration_between_captures": 0.05,
    "resolution": (640, 480),
    "camera_properties_kwargs": {
        "focalLength": 24.0,
        "focusDistance": 400,
        "fStop": 0.0,
        "clippingRange": (0.01, 10000),
    },
    "camera_look_at_target_offset": 0.5,
    "camera_distance_to_target_min_max": (0.100, 1.000),
    "writer_type": "PoseWriter",
    "writer_kwargs": {
        "output_dir": OUT_DIR,
        "format": None,
        "use_subfolders": False,
        "write_debug_images": True,
        "skip_empty_frames": False,
        # "semantic_segmentation": True,  
        # "colorize_semantic_segmentation": True,
    },
    "labeled_assets_and_properties": [
        # {
        #     "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
        #     "label": "pudding_box",
        #     "count": 5,
        #     "floating": True,
        #     "scale_min_max": (0.85, 1.25),
        # },
        # {
        #     "url": "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        #     "label": "mustard_bottle",
        #     "count": 7,
        #     "floating": True,
        #     "scale_min_max": (0.85, 1.25),
        # },
        {
            # "url": "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Props/Shapes/plane.usd", 
            "label": "tag0", 
            "count": 1, 
            "floating": True, 
            "scale_min_max": (0.1, 0.1), 
        }
    ],
    "shape_distractors_types": ["capsule", "cone", "cylinder", "sphere", "cube"],
    "shape_distractors_scale_min_max": (0.015, 0.15),
    "shape_distractors_num": 350,
    "mesh_distractors_urls": [
        # "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd",
        # "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    ],
    "mesh_distractors_scale_min_max": (0.35, 1.35),
    "mesh_distractors_num": 75,
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
# import object_based_sdg_utils
import sys 
sys.path.append("/home/rp/.local/share/ov/pkg/isaac-sim-4.2.0/standalone_examples/replicator/object_based_sdg")
import object_based_sdg_utils  
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import usdrt
from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
from omni.isaac.nucleus import get_assets_root_path
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics
from pxr import Usd, UsdShade, Gf

# HELPER FUNCTIONS
# Add transformation properties to the prim (if not already present)
def set_transform_attributes(prim, location=None, orientation=None, rotation=None, scale=None):
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if orientation is not None:
        if not prim.HasAttribute("xformOp:orient"):
            UsdGeom.Xformable(prim).AddOrientOp()
        prim.GetAttribute("xformOp:orient").Set(orientation)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            UsdGeom.Xformable(prim).AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)


# Enables collisions with the asset (without rigid body dynamics the asset will be static)
def add_colliders(prim):
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            # PhysX
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            physx_collision_api.CreateRestOffsetAttr(0.0)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")


# Enables rigid body dynamics (physics simulation) on the prim (having valid colliders is recommended)
def add_rigid_body_dynamics(prim, disable_gravity=False, angular_damping=None):
    # Physics
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    else:
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    # PhysX
    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    else:
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
    physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
    if angular_damping is not None:
        physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)


# Create a new prim with the provided asset URL and transform properties
def create_asset(stage, asset_url, path, location=None, rotation=None, orientation=None, scale=None):
    prim_path = omni.usd.get_stage_next_free_path(stage, path, False)
    reference_url = asset_url if asset_url.startswith("omniverse://") else get_assets_root_path() + asset_url
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(reference_url)
    set_transform_attributes(prim, location=location, rotation=rotation, orientation=orientation, scale=scale)
    return prim


# Create a new prim with the provided asset URL and transform properties including colliders
def create_asset_with_colliders(stage, asset_url, path, location=None, rotation=None, orientation=None, scale=None):
    prim = create_asset(stage, asset_url, path, location, rotation, orientation, scale)
    add_colliders(prim)
    return prim

# Isaac nucleus assets root path
assets_root_path = get_assets_root_path()
stage = None

# ENVIRONMENT
# Create an empty or load a custom stage (clearing any previous semantics)
env_url = config.get("env_url", "")
if env_url:
    env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
    omni.usd.get_context().open_stage(env_path)
    stage = omni.usd.get_context().get_stage()
    # Remove any previous semantics in the loaded stage
    for prim in stage.Traverse():
        remove_all_semantics(prim)
else:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    # Add a distant light to the empty stage
    distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
    distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(400.0)
    if not distant_light.HasAttribute("xformOp:rotateXYZ"):
        UsdGeom.Xformable(distant_light).AddRotateXYZOp()
    distant_light.GetAttribute("xformOp:rotateXYZ").Set((0, 60, 0))

# Get the working area size and bounds (width=x, depth=y, height=z)
working_area_size = config.get("working_area_size", (3, 3, 3))
working_area_min = (working_area_size[0] / -2, working_area_size[1] / -2, working_area_size[2] / -2)
working_area_max = (working_area_size[0] / 2, working_area_size[1] / 2, working_area_size[2] / 2)

# Create a collision box area around the assets to prevent them from drifting away
object_based_sdg_utils.create_collision_box_walls(
    stage, "/World/CollisionWalls", working_area_size[0], working_area_size[1], working_area_size[2]
)

# Create a physics scene to add or modify custom physics settings
usdrt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
physics_scenes = usdrt_stage.GetPrimsWithAppliedAPIName("PhysxSceneAPI")
if physics_scenes:
    physics_scene = physics_scenes[0]
else:
    physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
physx_scene.GetTimeStepsPerSecondAttr().Set(60)


# TRAINING ASSETS
# Add the objects to be trained in the environment with their labels and properties
labeled_assets_and_properties = config.get("labeled_assets_and_properties", [])
floating_labeled_prims = []
falling_labeled_prims = []
labeled_prims = []
for obj in labeled_assets_and_properties:
    obj_url = obj.get("url", "")
    label = obj.get("label", "unknown")
    count = obj.get("count", 1)
    floating = obj.get("floating", False)
    scale_min_max = obj.get("scale_min_max", (1, 1))
    for i in range(count):
        # Create a prim and add the asset reference
        rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
            loc_min=working_area_min, loc_max=working_area_max, scale_min_max=scale_min_max
        )

        tag = rep.create.plane(
            position = rand_loc,
            scale = rand_scale, 
            rotation = rand_rot,   
            name = "tag0", 
            semantics=[("class", label)],
        )
        tag_prim = tag.get_output_prims()["prims"][0] 
        set_transform_attributes(tag_prim, location=rand_loc, rotation=rand_rot, scale=rand_scale) 
        add_colliders(tag_prim)
        add_rigid_body_dynamics(tag_prim, disable_gravity=floating)

        with tag:       
            mat = rep.create.material_omnipbr(
                # diffuse_texture="/home/rp/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/tags/tag36h11_0.png",
                diffuse_texture=tag_textures[0],
                roughness_texture=rep.distribution.choice(rep.example.TEXTURES),
                metallic_texture=rep.distribution.choice(rep.example.TEXTURES),
                emissive_texture=rep.distribution.choice(rep.example.TEXTURES),
                emissive_intensity=rep.distribution.uniform(0, 1000),
            )    
            rep.modify.material(mat) 
        if floating:
            floating_labeled_prims.append(tag_prim)
        else:
            falling_labeled_prims.append(tag_prim)

labeled_prims = floating_labeled_prims + falling_labeled_prims

# DISTRACTORS
# Add shape distractors to the environment as floating or falling objects
shape_distractors_types = config.get("shape_distractors_types", ["capsule", "cone", "cylinder", "sphere", "cube"])
shape_distractors_scale_min_max = config.get("shape_distractors_scale_min_max", (0.02, 0.2))
shape_distractors_num = config.get("shape_distractors_num", 350)
shape_distractors = []
floating_shape_distractors = []
falling_shape_distractors = []
# for i in range(shape_distractors_num):
#     rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
#         loc_min=working_area_min, loc_max=working_area_max, scale_min_max=shape_distractors_scale_min_max
#     )
#     rand_shape = random.choice(shape_distractors_types)
#     prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/{rand_shape}", False)
#     prim = stage.DefinePrim(prim_path, rand_shape.capitalize())
#     object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
#     object_based_sdg_utils.add_colliders(prim)
#     disable_gravity = random.choice([True, False])
#     object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity)
#     if disable_gravity:
#         floating_shape_distractors.append(prim)
#     else:
#         falling_shape_distractors.append(prim)
#     shape_distractors.append(prim)

# Add mesh distractors to the environment as floating of falling objects
# mesh_distactors_urls = config.get("mesh_distractors_urls", [])
# mesh_distactors_scale_min_max = config.get("mesh_distractors_scale_min_max", (0.1, 2.0))
# mesh_distactors_num = config.get("mesh_distractors_num", 10)
# mesh_distractors = []
# floating_mesh_distractors = []
# falling_mesh_distractors = []
# for i in range(mesh_distactors_num):
#     rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
#         loc_min=working_area_min, loc_max=working_area_max, scale_min_max=mesh_distactors_scale_min_max
#     )
#     mesh_url = random.choice(mesh_distactors_urls)
#     prim_name = os.path.basename(mesh_url).split(".")[0]
#     prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/{prim_name}", False)
#     prim = stage.DefinePrim(prim_path, "Xform")
#     asset_path = mesh_url if mesh_url.startswith("omniverse://") else assets_root_path + mesh_url
#     prim.GetReferences().AddReference(asset_path)
#     object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
#     object_based_sdg_utils.add_colliders(prim)
#     disable_gravity = random.choice([True, False])
#     object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=disable_gravity)
#     if disable_gravity:
#         floating_mesh_distractors.append(prim)
#     else:
#         falling_mesh_distractors.append(prim)
#     mesh_distractors.append(prim)
#     # Remove any previous semantics on the mesh distractor
#     remove_all_semantics(prim, recursive=True)

# REPLICATOR
# Disable capturing every frame (capture will be triggered manually using the step function)
rep.orchestrator.set_capture_on_play(False)

# Create the camera prims and their properties
cameras = []
num_cameras = config.get("num_cameras", 1)
camera_properties_kwargs = config.get("camera_properties_kwargs", {})
for i in range(num_cameras):
    # Create camera and add its properties (focal length, focus distance, f-stop, clipping range, etc.)
    cam_prim = stage.DefinePrim(f"/World/Cameras/cam_{i}", "Camera")
    for key, value in camera_properties_kwargs.items():
        if cam_prim.HasAttribute(key):
            cam_prim.GetAttribute(key).Set(value)
        else:
            print(f"Unknown camera attribute with {key}:{value}")
    cameras.append(cam_prim)

# Add collision spheres (disabled by default) to cameras to avoid objects overlaping with the camera view
camera_colliders = []
camera_collider_radius = config.get("camera_collider_radius", 0)
if camera_collider_radius > 0:
    for cam in cameras:
        cam_path = cam.GetPath()
        cam_collider = stage.DefinePrim(f"{cam_path}/CollisionSphere", "Sphere")
        cam_collider.GetAttribute("radius").Set(camera_collider_radius)
        object_based_sdg_utils.add_colliders(cam_collider)
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(False)
        UsdGeom.Imageable(cam_collider).MakeInvisible()
        camera_colliders.append(cam_collider)

# Wait an app update to ensure the prim changes are applied
simulation_app.update()

# Create render products using the cameras
render_products = []
resolution = config.get("resolution", (640, 480))
for cam in cameras:
    rp = rep.create.render_product(cam.GetPath(), resolution)
    render_products.append(rp)

# Enable rendering only at capture time
disable_render_products_between_captures = config.get("disable_render_products_between_captures", True)
if disable_render_products_between_captures:
    object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

# # WRITER 
# # Create the writer and attach the render products
# writer_type = config.get("writer_type", "PoseWriter")
# writer_kwargs = config.get("writer_kwargs", {})
# # If not an absolute path, set it relative to the current working directory
# if out_dir := writer_kwargs.get("output_dir"):
#     if not os.path.isabs(out_dir):
#         out_dir = os.path.join(out_dir)
#         import pdb; pdb.set_trace()
#         writer_kwargs["output_dir"] = out_dir
#     print(f"[SDG] Writing data to: {out_dir}")
# if writer_type is not None and len(render_products) > 0:
#     writer = rep.writers.get(writer_type)
#     writer.initialize(**writer_kwargs)
#     writer.attach(render_products)

# Example of accessing the data directly from annotators
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach(rp)
sem_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": True})
sem_annot.attach(rp)

# Util function to save rgb annotator data
def write_rgb_data(rgb_data, file_path):
    rgb_img = Image.fromarray(rgb_data, "RGBA")
    rgb_img.save(file_path + ".png")

# Util function to save semantic segmentation annotator data
def write_sem_data(sem_data, file_path):
    id_to_labels = sem_data["info"]["idToLabels"]
    with open(file_path + ".json", "w") as f:
        json.dump(id_to_labels, f)
    sem_image_data = np.frombuffer(sem_data["data"], dtype=np.uint8).reshape(*sem_data["data"].shape, -1)
    sem_img = Image.fromarray(sem_image_data, "RGBA")
    sem_img.save(file_path + ".png")

def write_pose_data(pose_data, file_path):
    with open(file_path + ".json", "w") as f:
        json.dump(pose_data, f) 

# RANDOMIZERS
# Apply a random (mostly) uppwards velocity to the objects overlapping the 'bounce' area
def on_overlap_hit(hit):
    prim = stage.GetPrimAtPath(hit.rigid_body)
    # Skip the camera collision spheres
    if prim not in camera_colliders:
        rand_vel = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(4, 8))
        prim.GetAttribute("physics:velocity").Set(rand_vel)
    return True  # return True to continue the query


# Area to check for overlapping objects (above the bottom collision box)
overlap_area_thickness = 0.1
overlap_area_origin = (0, 0, (-working_area_size[2] / 2) + (overlap_area_thickness / 2))
overlap_area_extent = (
    working_area_size[0] / 2 * 0.99,
    working_area_size[1] / 2 * 0.99,
    overlap_area_thickness / 2 * 0.99,
)


# Triggered every physics update step to check for overlapping objects
def on_physics_step(dt: float):
    hit_info = get_physx_scene_query_interface().overlap_box(
        carb.Float3(overlap_area_extent),
        carb.Float3(overlap_area_origin),
        carb.Float4(0, 0, 0, 1),
        on_overlap_hit,
        False,  # pass 'False' to indicate an 'overlap multiple' query.
    )


# Subscribe to the physics step events to check for objects overlapping the 'bounce' area
physx_sub = get_physx_interface().subscribe_physics_step_events(on_physics_step)


# Pull assets towards the working area center by applying a random velocity towards the given target
def apply_velocities_towards_target(assets, target=(0, 0, 0)):
    for prim in assets:
        loc = prim.GetAttribute("xformOp:translate").Get()
        strength = random.uniform(0.1, 1.0)
        pull_vel = ((target[0] - loc[0]) * strength, (target[1] - loc[1]) * strength, (target[2] - loc[2]) * strength)
        prim.GetAttribute("physics:velocity").Set(pull_vel)


# Randomize camera poses to look at a random target asset (random distance and center offset)
camera_distance_to_target_min_max = config.get("camera_distance_to_target_min_max", (0.1, 0.5))
camera_look_at_target_offset = config.get("camera_look_at_target_offset", 0.2)


def randomize_camera_poses():
    for cam in cameras:
        # Get a random target asset to look at
        target_asset = random.choice(labeled_prims)
        # Add a look_at offset so the target is not always in the center of the camera view
        loc_offset = (
            random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
            random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
            random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
        )
        target_loc = target_asset.GetAttribute("xformOp:translate").Get() + loc_offset
        # Get a random distance to the target asset
        distance = random.uniform(camera_distance_to_target_min_max[0], camera_distance_to_target_min_max[1])
        # Get a random pose of the camera looking at the target asset from the given distance
        cam_loc, quat = object_based_sdg_utils.get_random_pose_on_sphere(origin=target_loc, radius=distance)
        object_based_sdg_utils.set_transform_attributes(cam, location=cam_loc, orientation=quat)


# Temporarily enable camera colliders and simulate for the given number of frames to push out any overlapping objects
def simulate_camera_collision(num_frames=1):
    for cam_collider in camera_colliders:
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(True)
    if not timeline.is_playing():
        timeline.play()
    for _ in range(num_frames):
        simulation_app.update()
    for cam_collider in camera_colliders:
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(False)


# Create a randomizer for the shape distractors colors, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
    shape_distractors_paths = [prim.GetPath() for prim in chain(floating_shape_distractors, falling_shape_distractors)]
    shape_distractors_group = rep.create.group(shape_distractors_paths)
    with shape_distractors_group:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

# Create a randomizer to apply random velocities to the floating shape distractors
# with rep.trigger.on_custom_event(event_name="randomize_floating_distractor_velocities"):
#     shape_distractors_paths = [prim.GetPath() for prim in chain(floating_shape_distractors, floating_mesh_distractors)]
#     shape_distractors_group = rep.create.group(shape_distractors_paths)
#     with shape_distractors_group:
#         rep.physics.rigid_body(
#             velocity=rep.distribution.uniform((-2.5, -2.5, -2.5), (2.5, 2.5, 2.5)),
#             angular_velocity=rep.distribution.uniform((-45, -45, -45), (45, 45, 45)),
#         )


# Create a randomizer for lights in the working area, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_lights"):
    lights = rep.create.light(
        light_type="Sphere",
        color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
        temperature=rep.distribution.normal(6500, 500),
        intensity=rep.distribution.normal(35000, 5000),
        position=rep.distribution.uniform(working_area_min, working_area_max),
        scale=rep.distribution.uniform(0.1, 1),
        count=3,
    )

with rep.trigger.on_custom_event(event_name="randomize_tag_texture"): 
    with tag:       
        mat = rep.create.material_omnipbr(
            diffuse_texture=rep.distribution.choice(tag_textures),
            roughness_texture=rep.distribution.choice(rep.example.TEXTURES),
            metallic_texture=rep.distribution.choice(rep.example.TEXTURES),
            emissive_texture=rep.distribution.choice(rep.example.TEXTURES),
            emissive_intensity=rep.distribution.uniform(0, 1000),
        )    
        rep.modify.material(mat) 

# Create a randomizer for the dome background, manually triggered at custom events
dir_backgrounds = "/media/rp/Elements/abhay_ws/marker_detection_failure_recovery/synthetic_data_generation/assets/background_images" 
dome_textures = [os.path.join(dir_backgrounds, f) for f in os.listdir(dir_backgrounds) if os.path.isfile(os.path.join(dir_backgrounds, f))] 
with rep.trigger.on_custom_event(event_name="randomize_dome_background"):
    dome_light = rep.create.light(light_type="Dome")
    with dome_light:
        rep.modify.attribute("inputs:texture:file", rep.distribution.choice(dome_textures))
        rep.randomizer.rotation()

# Capture motion blur by combining the number of pathtraced subframes samples simulated for the given duration
def capture_with_motion_blur_and_pathtracing(duration=0.05, num_samples=8, spp=64):
    # For small step sizes the physics FPS needs to be temporarily increased to provide movements every syb sample
    orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
    target_physics_fps = 1 / duration * num_samples
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Changing physics FPS from {orig_physics_fps} to {target_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)

    # Enable motion blur (if not enabled)
    is_motion_blur_enabled = carb.settings.get_settings().get("/omni/replicator/captureMotionBlur")
    if not is_motion_blur_enabled:
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
    # Number of sub samples to render for motion blur in PathTracing mode
    carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not timeline.is_playing():
        timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore the original physics FPS
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Restoring physics FPS from {target_physics_fps} to {orig_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(orig_physics_fps)

    # Restore the previous render and motion blur  settings
    carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", is_motion_blur_enabled)
    print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)


# Update the app until a given simulation duration has passed (simulate the world between captures)
def run_simulation_loop(duration):
    timeline = omni.timeline.get_timeline_interface()
    elapsed_time = 0.0
    previous_time = timeline.get_current_time()
    if not timeline.is_playing():
        timeline.play()
    app_updates_counter = 0
    while elapsed_time <= duration:
        simulation_app.update()
        elapsed_time += timeline.get_current_time() - previous_time
        previous_time = timeline.get_current_time()
        app_updates_counter += 1
        print(
            f"\t Simulation loop at {timeline.get_current_time():.2f}, current elapsed time: {elapsed_time:.2f}, counter: {app_updates_counter}"
        )
    print(
        f"[SDG] Simulation loop finished in {elapsed_time:.2f} seconds at {timeline.get_current_time():.2f} with {app_updates_counter} app updates."
    )


# SDG
# Number of frames to capture
num_frames = config.get("num_frames", 1000)

# Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
rt_subframes = config.get("rt_subframes", -1)

# Amount of simulation time to wait between captures
sim_duration_between_captures = config.get("simulation_duration_between_captures", 0.025)

# Initial trigger for randomizers before the SDG loop with several app updates (ensures materials/textures are loaded)
rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
rep.utils.send_og_event(event_name="randomize_dome_background")
rep.utils.send_og_event(event_name="randomize_tag_texture") 
print("[SDG] Initial randomizers triggered") 
for _ in range(5):
    simulation_app.update()

# Set the timeline parameters (start, end, no looping) and start the timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
# If no custom physx scene is created, a default one will be created by the physics engine once the timeline starts
timeline.play()
timeline.commit()
simulation_app.update()

# Store the wall start time for stats
wall_time_start = time.perf_counter()

# Run the simulation and capture data triggering randomizations and actions at custom frame intervals
print(f"[SDG] Starting SDG loop for {num_frames} frames")
for i in range(num_frames):

    if i % 1 == 0: 
        print(f"\t Randomizing marker texture") 
        rep.utils.send_og_event(event_name="randomize_tag_texture") 

    # Cameras will be moved to a random position and look at a randomly selected labeled asset
    if i % 3 == 0:
        print(f"\t Randomizing camera poses")
        randomize_camera_poses()
        # Temporarily enable camera colliders and simulate for a few frames to push out any overlapping objects
        if camera_colliders:
            simulate_camera_collision(num_frames=4)

    # # Apply a random velocity towards the origin to the working area to pull the assets closer to the center
    # if i % 10 == 0:
    #     print(f"\t Applying velocity towards the origin")
    #     apply_velocities_towards_target(chain(labeled_prims, shape_distractors, mesh_distractors))

    # Randomize lights locations and colors
    if i % 5 == 0:
        print(f"\t Randomizing lights")
        rep.utils.send_og_event(event_name="randomize_lights")

    # # Randomize the colors of the primitive shape distractors
    # if i % 15 == 0:
    #     print(f"\t Randomizing shape distractors colors")
    #     rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

    # Randomize the texture of the dome background
    if i % 25 == 0:
        print(f"\t Randomizing dome background")
        rep.utils.send_og_event(event_name="randomize_dome_background")

    # # Apply a random velocity on the floating distractors (shapes and meshes)
    # if i % 17 == 0:
    #     print(f"\t Randomizing shape distractors velocities")
    #     rep.utils.send_og_event(event_name="randomize_floating_distractor_velocities")

    # Enable render products only at capture time
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

    # Capture the current frame
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    if i % 5 == 0:
        capture_with_motion_blur_and_pathtracing(duration=0.025, num_samples=8, spp=128)
    else:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

    # gather pose data 
    xform_cam = UsdGeom.Xformable(cam) 
    tf_cam_pxr = xform_cam.ComputeLocalToWorldTransform(0) 
    tf_cam = np.asarray(tf_cam_pxr) 
    xform_tag = UsdGeom.Xformable(tag_prim) 
    tf_cam_pxr = xform_tag.ComputeLocalToWorldTransform(0) 
    tf_tag = np.asarray(tf_cam_pxr) 
    pose_data = {"cam": tf_cam.tolist(), "tag": tf_tag.tolist()} 
    write_rgb_data(rgb_annot.get_data(), f"{OUT_DIR}/rgb/rgb_{i}")
    write_sem_data(sem_annot.get_data(), f"{OUT_DIR}/seg/seg_{i}")
    write_pose_data(pose_data, f"{OUT_DIR}/pose/pose_{i}") 

    # Disable render products between captures
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

    # Run the simulation for a given duration between frame captures
    if sim_duration_between_captures > 0:
        run_simulation_loop(duration=sim_duration_between_captures)
    else:
        simulation_app.update()

# Wait for the data to be written (default writer backends are asynchronous)
rep.orchestrator.wait_until_complete()

# Get the stats
wall_duration = time.perf_counter() - wall_time_start
sim_duration = timeline.get_current_time()
avg_frame_fps = num_frames / wall_duration
num_captures = num_frames * num_cameras
avg_capture_fps = num_captures / wall_duration
print(
    f"[SDG] Captured {num_frames} frames, {num_captures} entries (frames * cameras) in {wall_duration:.2f} seconds.\n"
    f"\t Simulation duration: {sim_duration:.2f}\n"
    f"\t Simulation duration between captures: {sim_duration_between_captures:.2f}\n"
    f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
    f"\t Average capture entries (frames * cameras) FPS: {avg_capture_fps:.2f}\n"
)

# Unsubscribe the physics overlap checks and stop the timeline
physx_sub.unsubscribe()
physx_sub = None
simulation_app.update()
timeline.stop()

simulation_app.close()

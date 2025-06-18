import os 
import sys     
if os.getcwd() == '/home/anegi/abhay_ws/marker_detection_failure_recovery': # isaac machine 
    sys.path.append("/home/anegi/.local/share/ov/pkg/isaac-sim-4.5.0/standalone_examples/replicator/object_based_sdg")
else: # CAM machine 
    sys.path.append("/home/rp/.local/share/ov/pkg/isaac-sim-4.5.0/standalone_examples/replicator/object_based_sdg")
    
import numpy as np 

# SYS DEPENDENT IMPORTS 
import object_based_sdg_utils  
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import usdrt

# from isaacsim.core.utils.semantics import add_update_semantics, remove_all_semantics
from omni.isaac.nucleus import get_assets_root_path
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics
from pxr import Usd, UsdShade, Gf

# HELPER FUNCTIONS 
# TODO: export to a separate file 

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


def get_world_transform_xform_as_np_tf(prim: Usd.Prim):
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

    return np.array(world_transform).transpose()

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

def write_metadata(metadata, file_path): 
    with open(file_path + ".json", "w") as f:
        json.dump(metadata, f) 

def serialize_vec3f(vec3f):
    # Convert Gf.Vec3f to a list or dictionary
    return [vec3f[0], vec3f[1], vec3f[2]]

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

def quatf_to_eul(quatf): 
    qw = quatf.real 
    qx, qy, qz = np.array(quatf.imaginary) 
    a,b,c = R.from_quat([qx,qy,qz,qw]).as_euler('xyz',degrees=True) 
    return a,b,c 

def get_random_pose_on_hemisphere(origin, radius, camera_forward_axis=(0, 0, -1)):
    origin = Gf.Vec3f(origin)
    camera_forward_axis = Gf.Vec3f(camera_forward_axis)

    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arcsin(np.random.uniform(-1, 1))

    # Spherical to Cartesian conversion
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(phi)
    z = abs(radius * np.sin(theta) * np.cos(phi))

    location = origin + Gf.Vec3f(x, y, z)

    # Calculate direction vector from camera to look_at point
    direction = origin - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), Gf.Vec3d(direction_normalized))
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation

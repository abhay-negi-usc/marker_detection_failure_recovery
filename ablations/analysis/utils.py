from scipy.spatial.transform import Rotation as R
import numpy as np 

def compute_tf_error(tf_ref, tf_est):
    tf_err = np.linalg.inv(tf_ref) @ tf_est 
    return tf_err 

def tf_to_pose(tf): 
    """
    Convert a transformation matrix to a pose (position and orientation).
    """
    position = tf[:3, 3]
    euler = R.from_matrix(tf[:3, :3]).as_euler('xyz', degrees=True) 
    pose = np.concatenate((position, euler))
    return pose 


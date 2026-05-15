"""
Helper utilities for geometry, vision, and MuJoCo operations.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R

def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Convert a MuJoCo quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) in radians to a MuJoCo
    quaternion [w, x, y, z].
    """
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    q = r.as_quat()  # (x, y, z, w) in scipy
    return np.array([q[3], q[0], q[1], q[2]])  # Swap to (w, x, y, z)

def pixel_to_world(
    u: float,
    v: float,
    z_plane: float,
    camera_pos: np.ndarray,
    camera_quat: np.ndarray,
    camera_fx: float,
    camera_fy: float,
    camera_cx: float,
    camera_cy: float,
) -> np.ndarray:
    """
    Convert a pixel (u, v) to a 3D world point on a plane z=z_plane.

    This uses a ray-plane intersection where the plane is horizontal at z=z_plane.
    It's a simplification of a full raycast that assumes the camera is roughly
    overhead and the table is horizontal.
    """
    # Build camera rotation matrix from quaternion
    R_cam = quat_to_rotmat(camera_quat)

    # Pixel in normalized device coordinates (NCD)
    x_ndc = (u - camera_cx) / camera_fx
    y_ndc = (v - camera_cy) / camera_fy

    # Ray direction in camera frame
    ray_cam = np.array([x_ndc, y_ndc, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    # Ray direction in world frame
    ray_world = R_cam @ ray_cam

    # Ray-plane intersection with z=z_plane
    # camera_pos + t * ray_world = (x, y, z_plane)
    # z = camera_pos[2] + t * ray_world[2] = z_plane
    if abs(ray_world[2]) < 1e-9:
        return camera_pos.copy()  # Parallel to plane, fallback
    t = (z_plane - camera_pos[2]) / ray_world[2]
    if t < 0:
        # Ray points away from plane
        return camera_pos.copy()

    point = camera_pos + t * ray_world
    return point

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))

def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to [-pi, pi].
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def normalize_joint_positions(joint_pos: np.ndarray) -> np.ndarray:
    """
    Normalize joint positions to [-1, 1] based on joint limits.
    """
    from utils.constants import JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH
    normalized = 2.0 * (joint_pos - JOINT_LIMITS_LOW) / (JOINT_LIMITS_HIGH - JOINT_LIMITS_LOW) - 1.0
    return normalized

def denormalize_joint_positions(normalized: np.ndarray) -> np.ndarray:
    """
    Denormalize joint positions from [-1, 1] to actual angles.
    """
    from utils.constants import JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH
    return 0.5 * (normalized + 1.0) * (JOINT_LIMITS_HIGH - JOINT_LIMITS_LOW) + JOINT_LIMITS_LOW


def compute_camera_extrinsics(
    camera_pos: np.ndarray,
    camera_quat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute camera extrinsics (rotation matrix and translation vector) for
    converting between world and camera coordinates.
    """
    R_cam = quat_to_rotmat(camera_quat)  # World-to-cam rotation
    t_vec = -R_cam.T @ camera_pos  # Camera position in camera frame
    return R_cam, t_vec

"""
planning/geometry_utils.py
Small, dependency-light geometry helpers shared by the path planner:
turning a surface normal into a full 6-DOF pose, and transforming points
/ vectors between camera and robot-base frames.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def normal_to_rotation_matrix(normal, reference_x=np.array([1.0, 0.0, 0.0])):
    """
    Builds a rotation matrix whose Z axis is ANTI-parallel to `normal`,
    i.e. the tool's Z axis points INTO the surface when approaching it.
    `reference_x` just fixes the tool's roll about that axis; swap it if
    you need a particular approach orientation for a given tool.
    """
    normal = np.asarray(normal, dtype=float)
    z_axis = -normal / np.linalg.norm(normal)

    x_axis = reference_x - np.dot(reference_x, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        reference_x = np.array([0.0, 1.0, 0.0])
        x_axis = reference_x - np.dot(reference_x, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def rotation_matrix_to_rotvec(R):
    return Rotation.from_matrix(R).as_rotvec()


def pose_from_position_normal(position, normal):
    """Returns a UR-style 6-vector [x, y, z, rx, ry, rz]."""
    R = normal_to_rotation_matrix(normal)
    rotvec = rotation_matrix_to_rotvec(R)
    return list(position) + list(rotvec)


def transform_point(T, point_xyz):
    p = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0])
    return (T @ p)[:3]


def transform_vector(T, vec_xyz):
    """Transforms a direction (not a position) -- rotation only, no translation."""
    R = T[:3, :3]
    return R @ np.asarray(vec_xyz, dtype=float)

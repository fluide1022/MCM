import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle
from pytorch3d.transforms.rotation_conversions import _angle_from_tan, _index_from_letter, axis_angle_to_matrix

from mcm.utils.geometry.quaternion import qmul


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_matrix_np(quaternions):
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d


def quaternion_to_cont6d(quaternions):
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = torch.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d


def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def cont6d_to_matrix_np(cont6d):
    q = torch.from_numpy(cont6d).contiguous().float()
    return cont6d_to_matrix(q).numpy()


def matrix_to_cont6d(matrix: torch.Tensor) -> torch.Tensor:
    quat = matrix_to_quaternion(matrix)
    cont6d = quaternion_to_cont6d(quat)
    return cont6d


def matrix_to_cont6d_np(matrix):
    matrix = torch.from_numpy(matrix)
    return matrix_to_cont6d(matrix).numpy()


def euler_to_quaternion(e: torch.Tensor, order, deg=True):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.view(-1, 3)

    ## if euler angles in degrees
    if deg:
        e = e * torch.pi / 180.

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack((torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    ry = torch.stack((torch.cos(y / 2), torch.zeros_like(y), torch.sin(y / 2), torch.zeros_like(y)), dim=1)
    rz = torch.stack((torch.cos(z / 2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z / 2)), dim=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.view(original_shape)


def euler_to_quaternion_np(e: np.ndarray, order, deg=True):
    e = torch.from_numpy(e)
    return euler_to_quaternion(e, order, deg).numpy()


def matrix_to_euler_angles(matrix: torch.Tensor, order: str, deg=True) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians or degree.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        order: order string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(order) != 3:
        raise ValueError("order must have 3 letters.")
    if order[1] in (order[0], order[2]):
        raise ValueError(f"Invalid order {order}.")
    for letter in order:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in order string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    ## if euler angles in degrees

    i0 = _index_from_letter(order[0])
    i2 = _index_from_letter(order[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            order[0], order[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            order[2], order[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    o = torch.stack(o, -1)
    if deg:
        o = o / torch.pi * 180.
    return o


def quaternion_to_euler(quat: torch.Tensor, order, deg=True):
    matrix = quaternion_to_matrix(quat)
    euler = matrix_to_euler_angles(matrix, order, deg)
    return euler


def quaternion_to_euler_np(quat: np.ndarray, order, deg=True):
    quat = torch.from_numpy(quat)
    return quaternion_to_euler(quat, order, deg).numpy()


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def quaternion_to_axis_angle(quaternion: torch.Tensor):
    rot_matrix = quaternion_to_matrix(quaternion)
    axis_angle = matrix_to_axis_angle(rot_matrix)
    return axis_angle


def quaternion_to_axis_angle_np(quaternion: np.ndarray):
    quaternion = torch.from_numpy(quaternion)
    return quaternion_to_axis_angle(quaternion).numpy()


def axis_angle_to_quaternion(axis_angle: torch.Tensor):
    rot_matrix = axis_angle_to_matrix(axis_angle)
    return matrix_to_quaternion(rot_matrix)


def axis_angle_to_quaternion_np(axis_angle: np.ndarray):
    axis_angle = torch.from_numpy(axis_angle)
    return axis_angle_to_quaternion(axis_angle).numpy()

def matrix_to_axis_angle_np(matrix:np.ndarray):
    matrix = torch.from_numpy(matrix)
    return matrix_to_axis_angle(matrix).numpy()
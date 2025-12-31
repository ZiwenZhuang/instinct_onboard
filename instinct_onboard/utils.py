from typing import Sequence

import numpy as np
import quaternion


def quat_rotate_inverse(q: np.quaternion, v: np.array):
    """q must be numpy-quaternion object in w, x, y, z order
    NOTE: non-batchwise version
    """
    q_inv = q.conjugate()
    return quaternion.rotate_vectors(q_inv, v)


def quat_to_tan_norm(quat: np.quaternion) -> np.array:
    """Convert axis-angle representation to tangent-normal representation.
    Args:
        quat must be numpy-quaternion object in w, x, y, z order
    Returns:
        np.array: Tangent-normal vector with shape (6,).
    """
    # represents a rotation using the tangent and normal vectors
    ref_tan = np.zeros(3)
    ref_tan[0] = 1
    tan = quaternion.rotate_vectors(quat, ref_tan)

    ref_norm = np.zeros(3)
    ref_norm[-1] = 1
    norm = quaternion.rotate_vectors(quat, ref_norm)

    tan_norm = np.concatenate([tan, norm], axis=-1)  # shape (6,)
    return tan_norm


TANNORM_PROTOTYPE = np.array(
    [
        [1.0, 0.0, 0.0],  # tangent vector
        [0.0, 0.0, 1.0],  # normal vector
    ]
)  # shape (2, 3)


def quat_to_tan_norm_batch(quats: np.ndarray) -> np.ndarray:
    """Convert a batch of quaternions to tangent-normal representation.

    Args:
        quats: A batch of quaternions in (N, 4) shape, where N is the batch size.

    Returns:
        A batch of tangent-normal vectors in (N, 6) shape.
    """

    if not quats.dtype == quaternion.quaternion:
        quats = quaternion.from_float_array(quats)

    tannorm = quaternion.rotate_vectors(quats, TANNORM_PROTOTYPE)  # (N, 2, 3)
    tannorm = tannorm.reshape(len(quats), 6)
    return tannorm


def normalize_quat(quat: np.quaternion) -> np.quaternion:
    """Normalize the quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Not Batched.

    Returns:
        A normalized quaternion, with w > 0.
    """
    quat = quat / np.linalg.norm(quaternion.as_float_array(quat), axis=-1).clip(min=1e-6)
    quat = quat * np.sign(quaternion.as_float_array(quat)[..., 0])
    return quat


def inv_quat(quat: np.quaternion) -> np.quaternion:
    quat_norm = np.linalg.norm(quaternion.as_float_array(quat)).clip(min=1e-6)
    return quat.conjugate() / (quat_norm**2)


def yaw_quat(quat: np.quaternion) -> np.quaternion:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Not Batched.

    Returns:
        A quaternion with only yaw component.
    """
    qw = quat.w
    qx = quat.x
    qy = quat.y
    qz = quat.z
    # NOTE: arctan2 is used instead of atan2 because numpy 1.24.4 does not have atan2.
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw = np.zeros(4)
    quat_yaw[3] = np.sin(yaw / 2)
    quat_yaw[0] = np.cos(yaw / 2)
    return normalize_quat(quaternion.as_quat_array(quat_yaw))


def quat_slerp_batch(
    q1: np.ndarray,
    q2: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Slerp between two quaternions.
    Args:
        q1: The first quaternions in (N, 4) shape.
        q2: The second quaternions in (N, 4) shape.
        tau: The interpolation factors in (N,) shape.
    Returns:
        The slerped quaternions in (N, 4) shape.
    """
    # ensure the input is in the right shape
    assert q1.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert q2.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert tau.shape == q1.shape[:-1], "The batch size must be the same for all inputs."
    assert q1.shape[0] == q2.shape[0] == tau.shape[0], "The batch size must be the same for all inputs."
    assert (tau >= 0).all() and (tau <= 1).all(), "The interpolation factor must be in (0, 1) range."

    # if the dot product is negative, flip the quaternion
    dot_product = np.sum(q1 * q2, axis=-1, keepdims=True).clip(-1, 1)  # shape (N, 1)
    q2 = np.where(dot_product < 0, -q2, q2)
    dot_product = np.where(dot_product < 0, -dot_product, dot_product)

    # calculate the angle between the two quaternions
    theta = np.arccos(dot_product)
    sin_theta = np.sin(theta)
    q_too_similar = (dot_product > (1 - 1e-9)) | (np.abs(theta) < (1e-9))

    # avoid division by zero
    sin_theta = np.where(np.abs(sin_theta) < 1e-9, np.ones_like(sin_theta), sin_theta)

    # calculate the interpolation factor
    s1 = np.sin((1 - tau)[:, None] * theta) / sin_theta
    s2 = np.sin(tau[:, None] * theta) / sin_theta

    # calculate the interpolated quaternion
    interpolated_quat = (s1 * q1 + s2 * q2) * (~q_too_similar) + q_too_similar * q1
    interpolated_quat = interpolated_quat / np.linalg.norm(interpolated_quat, axis=-1, keepdims=True).clip(
        min=1e-6
    )  # shape (N, 4)

    return interpolated_quat


class CircularBuffer:
    """A circular buffer with fixed length and filled with a default value."""

    def __init__(self, length: int):
        self._buffer: np.ndarray | None = None
        self._length = length
        self._num_pushes = 0  # in case of reset or the buffer is not full, this will be less than length

    def append(self, value: float):
        """Append a value to the buffer, if the buffer is full, the oldest value will be removed."""
        if self._buffer is None:
            self._buffer = np.zeros((self._length,) + tuple(value.shape), dtype=np.float32)
        if self._num_pushes == 0:
            self._buffer[:] = value
        else:
            self._buffer = np.roll(self._buffer, -1, axis=0)
            self._buffer[-1] = value
        self._num_pushes += 1

    @property
    def buffer(self):
        return self._buffer

    def reset(self):
        if self._buffer is None:
            return
        self._buffer[:] = 0.0
        self._num_pushes = 0

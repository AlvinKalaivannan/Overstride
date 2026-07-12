"""Pinhole camera projection: 3D world points -> 2D image pixel coordinates.

Used to validate 2D pose extraction against datasets (like ASPset-510) that
ship 3D ground truth + camera calibration rather than 2D keypoints directly.
"""

from __future__ import annotations

import numpy as np


def project_points(
    points_3d: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
) -> np.ndarray:
    """Project (..., 3) world-space points to (..., 2) pixel coordinates.

    `intrinsic_matrix` is 3x4, `extrinsic_matrix` is 4x4 (world -> camera space).
    """
    projection_matrix = intrinsic_matrix @ extrinsic_matrix  # 3x4
    ones_shape = points_3d.shape[:-1] + (1,)
    homogeneous = np.concatenate([points_3d, np.ones(ones_shape)], axis=-1)
    image_homogeneous = homogeneous @ projection_matrix.T
    return image_homogeneous[..., :2] / image_homogeneous[..., 2:3]

from typing import Optional
import torch
from torch.nn import functional as F

def rotmat_to_axis_angle(matrix: torch.Tensor):
    """Principal axis-angle through the best-conditioned quaternion branch.

    Supports identity and rotations near pi without dividing by sin(angle).
    Like any principal axis-angle chart, it has a branch cut at pi.
    """
    r = matrix
    r00, r11, r22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    diagonal = torch.stack((1+r00+r11+r22, 1+r00-r11-r22,
                            1-r00+r11-r22, 1-r00-r11+r22), -1)
    w = torch.stack((diagonal[..., 0], r[..., 2, 1]-r[..., 1, 2],
                     r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]), -1)
    x = torch.stack((w[..., 1], diagonal[..., 1], r[..., 0, 1]+r[..., 1, 0],
                     r[..., 0, 2]+r[..., 2, 0]), -1)
    y = torch.stack((w[..., 2], x[..., 2], diagonal[..., 2], r[..., 1, 2]+r[..., 2, 1]), -1)
    z = torch.stack((w[..., 3], x[..., 3], y[..., 3], diagonal[..., 3]), -1)
    candidates = torch.stack((w, x, y, z), -2)
    index = diagonal.argmax(-1)[..., None, None].expand(*diagonal.shape[:-1], 1, 4)
    q = F.normalize(candidates.gather(-2, index).squeeze(-2), dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    half_angle = torch.atan2(torch.linalg.vector_norm(q[..., 1:], dim=-1), q[..., 0])
    return q[..., 1:] / (0.5 * torch.sinc(half_angle / torch.pi))[..., None]

def weak_perspective_projection(points: torch.Tensor, camera: torch.Tensor):
    """Scaled orthographic projection into full-crop pixel / size - 0.5 units.

    camera=(s, tx, ty), with translation before scaling, as in the existing
    camera-to-translation conversion. xy = s/2 * (XY + t); Z has no effect.
    The factor 1/2 matches this repository's [-0.5, 0.5] label convention.
    The paper names weak perspective but does not specify this normalization.
    """
    return camera[:, None, :1] * (points[..., :2] + camera[:, None, 1:]) / 2


def weak_perspective_feature_grid(points, camera, image_size=256, crop_margin=32):
    """Map full-crop projection to the horizontally cropped ViT feature grid.

    align_corners=False: pixel centers and all pyramid resolutions share one
    normalized grid. Account for the 32-pixel crop on each side of the input.
    """
    xy = weak_perspective_projection(points, camera)
    pixels = (xy + 0.5) * image_size
    x = 2 * (pixels[..., 0] - crop_margin + 0.5) / (image_size - 2 * crop_margin) - 1
    y = 2 * (pixels[..., 1] + 0.5) / image_size - 1
    return torch.stack((x, y), dim=-1)

def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


WSAD_OFFSET = 12.35
DIAGONAL_OFFSET = 8.73
MOUSE_PITCH_SENSITIVITY = 15.0
MOUSE_YAW_SENSITIVITY = 15.0
MOUSE_THRESHOLD = 0.02


def compute_all_poses_from_actions(
    keyboard_conditions,
    mouse_conditions,
    first_pose=None,
    return_last_pose: bool = False,
):
    total_frames = len(keyboard_conditions)
    all_poses = np.zeros((total_frames, 5), dtype=np.float32)
    if first_pose is not None:
        all_poses[0] = first_pose
    for idx in range(total_frames - 1):
        all_poses[idx + 1] = compute_next_pose_from_action(
            all_poses[idx],
            keyboard_conditions[idx],
            mouse_conditions[idx],
        )
    if return_last_pose:
        last_pose = compute_next_pose_from_action(
            all_poses[-1],
            keyboard_conditions[-1],
            mouse_conditions[-1],
        )
        return all_poses, last_pose
    return all_poses


def compute_next_pose_from_action(current_pose, keyboard_action, mouse_action):
    x, y, z, pitch, yaw = current_pose
    w, s, a, d = keyboard_action[:4]
    mouse_x, mouse_y = mouse_action[:2]

    delta_pitch = (
        MOUSE_PITCH_SENSITIVITY * mouse_x
        if abs(mouse_x) >= MOUSE_THRESHOLD
        else 0.0
    )
    delta_yaw = (
        MOUSE_YAW_SENSITIVITY * mouse_y
        if abs(mouse_y) >= MOUSE_THRESHOLD
        else 0.0
    )

    new_pitch = pitch + delta_pitch
    new_yaw = yaw + delta_yaw
    while new_yaw > 180:
        new_yaw -= 360
    while new_yaw < -180:
        new_yaw += 360

    local_forward = 0.0
    if w > 0.5 and s < 0.5:
        local_forward = WSAD_OFFSET
    elif s > 0.5 and w < 0.5:
        local_forward = -WSAD_OFFSET

    local_right = 0.0
    if d > 0.5 and a < 0.5:
        local_right = WSAD_OFFSET
    elif a > 0.5 and d < 0.5:
        local_right = -WSAD_OFFSET

    if abs(local_forward) > 0.1 and abs(local_right) > 0.1:
        local_forward = np.sign(local_forward) * DIAGONAL_OFFSET
        local_right = np.sign(local_right) * DIAGONAL_OFFSET

    avg_yaw = float((yaw + new_yaw) / 2.0)
    yaw_rad = float(np.deg2rad(avg_yaw))
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    delta_x = cos_yaw * local_forward - sin_yaw * local_right
    delta_y = sin_yaw * local_forward + cos_yaw * local_right
    new_x = x + delta_x
    new_y = y + delta_y

    return np.array([new_x, new_y, z, new_pitch, new_yaw], dtype=np.float32)


def get_extrinsics(video_rotation, video_position):
    num_frames = len(video_rotation)
    extrinsics_vid = []
    for idx in range(num_frames):
        roll, pitch, yaw = np.radians(video_rotation[idx])
        frame_position = video_position[idx]

        rz = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )
        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )
        rotation = rz @ ry @ rx
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = frame_position
        extrinsics_vid.append(extrinsics)

    r_init = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, -1, 0],
        ]
    )
    extrinsics = torch.from_numpy(np.array(extrinsics_vid))
    extrinsics[:, :3, :3] = extrinsics[:, :3, :3] @ r_init
    extrinsics[:, :3, 3] = extrinsics[:, :3, 3] * 0.01
    return extrinsics


def get_intrinsics(height: int, width: int) -> torch.Tensor:
    fov_deg = 90
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = height / (2 * np.tan(fov_rad / 2))
    cx = width / 2
    cy = height / 2
    return torch.tensor([fx, fy, cx, cy], dtype=torch.float32)


@torch.no_grad()
def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias
    return grid_xy[None, ...].repeat(n_frames, 1, 1)


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
):
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(
        n_frames,
        height,
        width,
        device=c2ws_mat.device,
        dtype=c2ws_mat.dtype,
    )
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)

    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)
    rays_o = c2ws_mat[:, :3, 3]
    rays_o = rays_o[:, None, :].expand_as(rays_d)
    plucker = torch.cat([rays_o, rays_d], dim=-1)
    return plucker.view([n_frames, height, width, 6])


def build_frame_extrinsics_from_conditions(
    keyboard_cond: torch.Tensor,
    mouse_cond: Optional[torch.Tensor],
    num_latents: int,
) -> torch.Tensor:
    total_frames = 1 + 4 * max(0, num_latents - 1)
    keyboard_np = (
        keyboard_cond[0, :total_frames].detach().float().cpu().numpy()
    )
    if mouse_cond is None:
        mouse_np = np.zeros((total_frames, 2), dtype=np.float32)
    else:
        mouse_np = mouse_cond[0, :total_frames].detach().float().cpu().numpy()

    first_pose = np.concatenate([np.zeros(3), np.zeros(2)], axis=0)
    all_poses = compute_all_poses_from_actions(
        keyboard_np,
        mouse_np,
        first_pose=first_pose,
    )
    positions = all_poses[:, :3].tolist()
    rotations = np.concatenate(
        [
            np.zeros((all_poses.shape[0], 1)),
            all_poses[:, 3:5],
        ],
        axis=1,
    ).tolist()
    return get_extrinsics(rotations, positions).float()


def latent_idx_to_frame_idx(latent_idx: int) -> int:
    return latent_idx * 4


def build_plucker_signature_from_extrinsic(
    extrinsic: torch.Tensor,
    height: int = 352,
    width: int = 640,
) -> torch.Tensor:
    if extrinsic.dim() == 2:
        extrinsic = extrinsic.unsqueeze(0)
    extrinsic = extrinsic.float()
    ks = get_intrinsics(height, width).to(
        device=extrinsic.device,
        dtype=extrinsic.dtype,
    )
    ks = ks.unsqueeze(0).repeat(extrinsic.shape[0], 1)
    plucker = get_plucker_embeddings(extrinsic, ks, height, width)
    signature = plucker.mean(dim=(0, 1, 2))
    return F.normalize(signature, dim=0, eps=1e-6)


def rank_candidate_extrinsics_by_fov(
    current_extrinsic: torch.Tensor,
    candidate_extrinsics: torch.Tensor,
    topk: int,
    width: int = 640,
    height: int = 352,
    near: float = 0.1,
    far: float = 30.0,
    grid_size: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if candidate_extrinsics.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=current_extrinsic.device)
        empty_scores = torch.empty(0, dtype=torch.float32, device=current_extrinsic.device)
        return empty, empty_scores

    device = current_extrinsic.device
    current_extrinsic = current_extrinsic.float().to(device)
    candidate_extrinsics = candidate_extrinsics.float().to(device)

    fov_rad = np.deg2rad(90)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = height / (2 * np.tan(fov_rad / 2))

    z_samples = torch.linspace(near, far, grid_size, device=device)
    x_samples = torch.linspace(-1, 1, grid_size, device=device)
    y_samples = torch.linspace(-1, 1, grid_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(
        x_samples,
        y_samples,
        z_samples,
        indexing="ij",
    )
    points_cam = torch.stack(
        [
            grid_x.reshape(-1) * grid_z.reshape(-1) * (width / (2 * fx)),
            grid_y.reshape(-1) * grid_z.reshape(-1) * (height / (2 * fy)),
            grid_z.reshape(-1),
        ],
        dim=0,
    )

    points_world = (
        current_extrinsic[:3, :3] @ points_cam + current_extrinsic[:3, 3:4]
    )
    rot_inv = candidate_extrinsics[:, :3, :3].transpose(1, 2)
    trans_inv = -torch.bmm(rot_inv, candidate_extrinsics[:, :3, 3:4])
    points_in_candidates = (
        torch.bmm(
            rot_inv,
            points_world.unsqueeze(0).expand(candidate_extrinsics.shape[0], -1, -1),
        )
        + trans_inv
    )

    x = points_in_candidates[:, 0, :]
    y = points_in_candidates[:, 1, :]
    z = points_in_candidates[:, 2, :]
    u = (x * fx / torch.clamp(z, min=1e-6)) + width / 2
    v = (y * fy / torch.clamp(z, min=1e-6)) + height / 2
    in_view = (
        (z > near)
        & (z < far)
        & (u >= 0)
        & (u <= width)
        & (v >= 0)
        & (v <= height)
    )
    scores = in_view.float().mean(dim=1)
    topk = min(topk, scores.shape[0])
    top_scores, top_indices = torch.topk(scores, k=topk)
    return top_indices, top_scores

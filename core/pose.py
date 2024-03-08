import torch
import torch.nn.functional as F
import logging


# Configure logging with time format
logging.basicConfig(
    filename='pose.log',
    filemode='w',
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
# set the logging level
logging.getLogger().setLevel(logging.DEBUG)


def compute_relative_pose(pose_b, pose_b_plus_1):
    pose_b_inv = torch.inverse(pose_b)
    relative_pose = torch.bmm(pose_b_inv, pose_b_plus_1)
    return relative_pose

def pose_vec_to_mat(pose_vec):
    B = pose_vec.size(0)
    translation = pose_vec[:, :3].unsqueeze(-1)  # Shape: [B, 3, 1]
    rot = pose_vec[:, 3:]
    cos_r, sin_r = torch.cos(rot), torch.sin(rot)
    rx, ry, rz = cos_r[:, 0], cos_r[:, 1], cos_r[:, 2]
    sx, sy, sz = sin_r[:, 0], sin_r[:, 1], sin_r[:, 2]
    Rx = torch.stack([torch.ones(B, device=pose_vec.device), torch.zeros(B, device=pose_vec.device), torch.zeros(B, device=pose_vec.device),
                      torch.zeros(B, device=pose_vec.device), rx, -sx,
                      torch.zeros(B, device=pose_vec.device), sx, rx], dim=1).view(B, 3, 3)
    Ry = torch.stack([ry, torch.zeros(B, device=pose_vec.device), sy,
                      torch.zeros(B, device=pose_vec.device), torch.ones(B, device=pose_vec.device), torch.zeros(B, device=pose_vec.device),
                      -sy, torch.zeros(B, device=pose_vec.device), ry], dim=1).view(B, 3, 3)
    Rz = torch.stack([rz, -sz, torch.zeros(B, device=pose_vec.device),
                      sz, rz, torch.zeros(B, device=pose_vec.device),
                      torch.zeros(B, device=pose_vec.device), torch.zeros(B, device=pose_vec.device), torch.ones(B, device=pose_vec.device)], dim=1).view(B, 3, 3)
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)
    transform_mat = torch.cat([R, translation], dim=2)  # Shape: [B, 3, 4]
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=pose_vec.device).unsqueeze(0).repeat(B, 1, 1)  # Shape: [B, 1, 4]
    transform_mat = torch.cat([transform_mat, bottom_row], dim=1)  # Shape: [B, 4, 4]
    return transform_mat

def warp_image(image, depth, relative_pose):
    B, C, H, W = image.shape
    grid_y, grid_x = torch.meshgrid(torch.linspace(0, H-1, H, device=image.device), torch.linspace(0, W-1, W, device=image.device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=2).to(image.device)  # Shape: [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # Shape: [B, H, W, 2]
    grid = grid.reshape(B, -1, 2)  # Shape: [B, H*W, 2]

    depth = depth.squeeze(1).view(B, -1, 1)  # Reshape depth to [B, H*W, 1] to match grid's last dimension
    grid_depth = torch.cat((grid, depth), dim=2)  # Concatenate along the last dimension to get [B, H*W, 3]

    ones = torch.ones(B, H*W, 1, device=image.device)
    grid_depth_hom = torch.cat((grid_depth, ones), dim=2)  # Shape: [B, H*W, 4]
    grid_warped_hom = torch.bmm(relative_pose, grid_depth_hom.transpose(1, 2))  # Corrected transpose for matrix multiplication, resulting in [B, 4, H*W]
    grid_warped = grid_warped_hom[:, :2, :] / grid_warped_hom[:, 2:3, :]  # Normalize to get [B, 2, H*W]

    # Normalize pixel coordinates to [-1, 1]
    grid_warped[:, 0, :] = (grid_warped[:, 0, :] / (W-1) * 2) - 1
    grid_warped[:, 1, :] = (grid_warped[:, 1, :] / (H-1) * 2) - 1
    grid_warped = grid_warped.transpose(1, 2).view(B, H, W, 2)  # Reshape back to [B, H, W, 2]

    warped_image = F.grid_sample(image, grid_warped, mode='bilinear', padding_mode='zeros', align_corners=True)

    return warped_image


def compute_transformation_loss_for_batch(images, depths, poses, transformation_loss_module, clip_n):
    total_loss = 0.0
    total_clips = images.size(0) // clip_n
    for clip_idx in range(total_clips):
        start = clip_idx * clip_n
        end = start + clip_n
        clip_images = images[start:end]
        clip_depths = depths[start:end]
        clip_poses = poses[start:end]
        clip_poses = pose_vec_to_mat(clip_poses).to(images.device)  # Ensure the pose matrix is on the GPU
        
        relative_pose = compute_relative_pose(clip_poses[:-1], clip_poses[1:])
        warped_image = warp_image(clip_images[:-1], clip_depths[:-1], relative_pose)
        loss = transformation_loss_module(warped_image, clip_images[1:])
        total_loss += loss

    return total_loss / (total_clips)

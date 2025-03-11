#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    """
        intrinsic：内参矩阵，C2pix = C2NDC @ NDC2pix
    """
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]   # NDC坐标下3D点的深度值
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z   # 拉回到图像平面
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())    # 图像坐标系 转换到 相机坐标系
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    """
        sampled_depth: (1,1,1,H,W)
        ref_intrinsic: (1,3,3)
    """
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    # 图像坐标系中x、y方向归一化到(0,1)的像素坐标
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)   # (W,)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)   # (H,)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x) # 两个维度都为(H,W)
    # 拓展到与深度相同的维度，(1,1,1,H,W)
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    # NDC坐标系下3D点的三维坐标
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # (1,1,1,H,W,3)
    # 相机坐标系下的3D点，(1,1,1,H,W,3)
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H)
    return ndc_xyz, cam_xyz

def depth2point_camera(depth_image, intrinsic_matrix, extrinsic_matrix):
    """
        depth_image：(H, W),
        intrinsic_matrix：(3, 3)
        extrinsic_matrix： (4, 4)
    """
    # 获取相机坐标下下的3D点坐标，(H*W, 3)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)

    return xyz_cam

def depth_pcd2normal(xyz, offset=None, gt_image=None):
    hd, wd, _ = xyz.shape
    if offset is not None:
        ix, iy = torch.meshgrid(
            torch.arange(wd), torch.arange(hd), indexing='xy')
        xy = (torch.stack((ix, iy), dim=-1)[1:-1,1:-1]).to(xyz.device)
        p_offset = torch.tensor([[0,1],[0,-1],[1,0],[-1,0]]).float().to(xyz.device)
        new_offset = p_offset[None,None] + offset.reshape(hd, wd, 4, 2)[1:-1,1:-1]
        xys = xy[:,:,None] + new_offset
        xys[..., 0] = 2 * xys[..., 0] / (wd - 1) - 1.0
        xys[..., 1] = 2 * xys[..., 1] / (hd - 1) - 1.0
        sampled_xyzs = torch.nn.functional.grid_sample(xyz.permute(2,0,1)[None], xys.reshape(1, -1, 1, 2))
        sampled_xyzs = sampled_xyzs.permute(0,2,3,1).reshape(hd-2,wd-2,4,3)
        bottom_point = sampled_xyzs[:,:,0]
        top_point = sampled_xyzs[:,:,1]
        right_point = sampled_xyzs[:,:,2]
        left_point = sampled_xyzs[:,:,3]
    else:
        bottom_point = xyz[..., 2:hd,   1:wd-1, :]
        top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
        right_point  = xyz[..., 1:hd-1, 2:wd,   :]
        left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix, offset=None, gt_image=None):
    """
    从深度图计算法向量（相机坐标系下）
        depth： 渲染深度图，(H, W)
        intrinsic_matrix：(3, 3)
        extrinsic_matrix：(4, 4)
        offset：
        gt_image：
        return：计算的法向量图，(H, W, 3)
    """
    # 相机坐标系下的3D点
    xyz_camera = depth2point_camera(depth, intrinsic_matrix, extrinsic_matrix) # (HxW, 3)
    xyz_camera = xyz_camera.reshape(*depth.shape, 3)  # (H, W, 3)
    # 相机坐标系下的法向量
    xyz_normal = depth_pcd2normal(xyz_camera, offset, gt_image)  # (H, W, 3)

    return xyz_normal

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    """
    从渲染深度图 计算 法向量（相机坐标系下）
        viewpoint_cam：当前相机
        depth: 渲染的深度图，(H, W)
    """
    # bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)   # 获取当前相机的 内参矩阵 和 外参矩阵（C2W）
    st = max(int(scale / 2) - 1, 0) # 如果scale>2，则st为(scale/2)-1的向下取整；否则为0
    if offset is not None:
        offset = offset[st::scale,st::scale]    # 如果输入了偏移量，也对其进行采样（减少计算量，并且采样时丢弃初始的行和列避免边缘的影响），采样后的大小为(H-st)//scale
    # 相机坐标系下的法向量图
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render_normal_2(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    """
    从渲染深度图 计算 法向量（相机坐标系下）
        viewpoint_cam：当前相机
        depth: 渲染的深度图，(H, W)
    """
    intrinsic_matrix, _ = viewpoint_cam.get_calib_matrix_nerf(scale=scale)  # 获取当前相机的 内参矩阵 和 外参矩阵（C2W）
    c2w = (viewpoint_cam.world_view_transform.T).inverse()  # C2W，相机到世界坐标系的变换矩阵
    grid_x, grid_y = torch.meshgrid(torch.arange(viewpoint_cam.image_width, device='cuda').float(),
                                    torch.arange(viewpoint_cam.image_height, device='cuda').float(),
                                    indexing='xy')  # 生成1个二维网格，分别包含 x 轴和 y 轴的坐标

    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)  # torch.stack 将 grid_x、grid_y 和一个全为 1 的张量按最后一个维度拼接，得到形状为 (H, W, 3) 的张量，每个位置的值是 (x, y, 1)
    # reshape(-1, 3)将张量展平为形状为(H*W, 3)，即H*W个像素点在图像坐标系中的的齐次坐标，cuda
    # 每个像素点在世界坐标系下的坐标：像素坐标系 => 相机坐标系 => 世界坐标系。cuda
    rays_d = points @ intrinsic_matrix.inverse().T.cuda() @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]  # 相机中心在世界坐标系的位置（射线起点）。cuda

    # 计算每个3D点的世界坐标：深度值与射线方向相乘，并加上射线起点。cuda
    points = depth.cuda().reshape(-1, 1) * rays_d + rays_o
    points = points.reshape(*depth.shape[:], 3)  # 重新调整为 H W 3

    output = torch.zeros_like(points)
    # 计算世界坐标系下3D点云在x、y方向上的梯度
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    # 叉乘 得到法向量，然后归一化
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map  # 中心填充

    normal = output.permute(2, 0, 1).cuda()  # 3 H W
    return normal

def render(viewpoint_camera, xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, 
        pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, cov3D_precomp = None, colors_precomp = None ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    opacity = opacity

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, depth, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth, 
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

    # depth_normal = render_normal(viewpoint_camera, depth.squeeze())  # 从渲染的 深度图计算法向量图 (3,H,W)

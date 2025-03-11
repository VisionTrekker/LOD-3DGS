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
from scene import Scene, GaussianModel
import os
import cv2
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

def render_set(model_path, name, iteration, views, scene, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]

        xyz, features, opacity, scales, rotations, cov3D_precomp, active_sh_degree, max_sh_degree, masks \
            = scene.get_gaussian_parameters(view.world_view_transform, pipeline.compute_cov3D_python)

        render_pkg = render(view, xyz, features, opacity, scales, rotations, active_sh_degree, max_sh_degree, pipeline,
                            background, cov3D_precomp=cov3D_precomp)

        rendering = render_pkg["render"].clamp(0.0, 1.0)

        depth = render_pkg["depth"].squeeze().detach().cpu().numpy()
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        normal = render_pkg["depth_normal"].permute(1, 2, 0)  # H,W,3
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1.0e-8)  # 保存时需归一化到 -1, 1
        normal = normal.detach().cpu().numpy()
        normal_vis = ((normal + 1) * 127.5).astype(np.uint8).clip(0, 255)  # 0, 2 ==> 0, 255

        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".png"), depth_vis)
        np.save(os.path.join(render_depth_path, view.image_name + ".npy"), depth)

        # cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".png"), normal_vis)
        # np.save(os.path.join(render_normal_path, view.image_name + ".npy"), normal)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scene = Scene(dataset, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("[ INFO ] Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
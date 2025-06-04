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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim 
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F
from models.networks import CNN_decoder

#/////////////////////////
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#////////////////////////

from encoders.XFeat.modules.xfeat import XFeat
from utils.pose_utils import getGTXYZ
from scene.feat_pointcloud import FeatPointCloud

"""
python train.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

If we train with disk, we need to point out in the dataset_reader.py where to find the pre-extract disk feature

"""

def gen_coords(W, H):
    x = torch.arange(W)
    y = torch.arange(H)
    xx, yy = torch.meshgrid(x, y, indexing='xy')  
    xx = xx.T
    yy = yy.T
    coords = torch.stack([xx, yy], dim=2).view(-1, 2)
    return coords



def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=15000)
    featpc = FeatPointCloud()
    featpc.init_feat_pc(dataset.source_path, 64)
    xfeat = XFeat(top_k=4096)
    
    # 2D semantic feature map CNN decoder
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))





    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1]*64 if dataset.white_background else [0]*64
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    saving_itr = [5000, 10000, 15000]

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        
        """
        Move the matching gaussian to its Gt position
        
        """ 
        
        # Get the query image
        render_coord =  gen_coords(80, 60).to("cuda")
        query_img = viewpoint_cam.original_image[0:3, :, :]
        
        
        gt_feature = xfeat.get_descriptors(query_img[None])[0]
        gt_feat = gt_feature.view(64, -1).transpose(0, 1)
        depth_map = render_pkg["depth"] 
        depth_map = F.interpolate(depth_map.unsqueeze(0), size=(60, 80), mode='bilinear', align_corners=True).squeeze(0) #60x80
        #For each keypoint detected in query image, find its coordinate in 3DGS
        query_keypoints_3d = getGTXYZ(viewpoint_cam.projection_matrix, viewpoint_cam.world_view_transform, render_coord, depth_map)

        featpc.update_ply(query_keypoints_3d, gt_feat)
        
        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

        
            if (iteration in saving_itr):
                print("\n[ITER {}] Saving Point Cloud".format(iteration))
                point_cloud_path = os.path.join(scene.model_path, "kpfeat_point_cloud_head/iteration_{}".format(iteration))
                featpc.save_ply(os.path.join(point_cloud_path, "feature_point_cloud.ply"))

                print("\n[ITER {}] Saving feature decoder ckpt".format(iteration))
  



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    ###### Fan WU #######
    args.eval = True
    #####################
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

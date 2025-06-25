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
import sys
import uuid
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from random import randint
 
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from encoders.XFeat.modules.xfeat import XFeat
from utils.pose_utils import getGTXYZ
from utils.general_utils import safe_state, image_process
from utils.loss_utils import l1_loss, ssim
from utils.sampling_utils import sample_random_points, sample_features
from gaussian_renderer import render
from scene.feat_pointcloud import FeatPointCloud

"""
python train.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

If we train with disk, we need to point out in the dataset_reader.py where to find the pre-extract disk feature

"""



def training(dataset, opt, pipe,  saving_iterations, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    featpc = FeatPointCloud()
    featpc.init_feat_pc(dataset.source_path, 64)
    xfeat = XFeat(top_k=4096)
    
    gaussians.training_setup(opt)


    bg_color = [1]*64 if dataset.white_background else [0]*64
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    

    for iteration in range(first_iter, opt.iterations + 1):

        """
        3DGS training 
        """
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Get training image 
        try:
            image = Image.open(viewpoint_cam.image_path) 
        except:
            print(f"Error opening image: {viewpoint_cam.image_path}")
            continue

        original_image = image_process(image)
        gt_im = original_image.cuda()
        
        img_width = gt_im.shape[2]
        img_height = gt_im.shape[1]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, img_width, img_height)
        
        
        image, viewspace_point_tensor, visibility_filter, radii =  render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # Loss
        
        Ll1 = l1_loss(image, gt_im)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_im))
        loss.backward()
        iter_end.record()
        
        """
        SFM Feature learning
        
        """ 
        #----- feature_map size [C,H,W] = [64,480,640]----#
        gt_feature_map = xfeat.get_descriptors(gt_im[None])[0]
        
        # Generate the sampling coordinates in [480, 640]
        render_coord =  sample_random_points(img_height,img_width, cell_size=8, device="cuda")
        
        # Sample the feature according to the coordinates
        gt_map = F.interpolate(gt_feature_map.unsqueeze(0), size=(img_height, img_width), mode='bilinear', align_corners=True).squeeze(0) #640x480
        gt_feat = sample_features(gt_map, render_coord) # get the feature from the ground truth 

        # Get the depth map and For each pixel in query image, find its coordinate in 3DGS      
        depth_map = render_pkg["depth"] 
        render_keypoints_3d = getGTXYZ(viewpoint_cam.projection_matrix, viewpoint_cam.world_view_transform, render_coord, depth_map)

        featpc.update_ply(render_keypoints_3d, gt_feat)
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, iter_start.elapsed_time(iter_end)) 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                #save feature point cloud 
                point_cloud_path = os.path.join(scene.model_path, "feature_point_cloud_chess/iteration_{}".format(iteration))
                featpc.save_ply(os.path.join(point_cloud_path, "feature_point_cloud.ply"))

                

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, elapsed):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
   
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Objectif of training :Test 
    args.eval = True
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")

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
from utils.loss_utils import l1_loss, ssim, tv_loss 
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
from models.semantic_dataloader import VariableSizeDataset
from torch.utils.data import DataLoader

# Regulation package
from reguler.helper.utils import *
from reguler.helper.transformer import *
from reguler.network import *
from utils.graphics_utils import fov2focal
from encoders.XFeat.modules.xfeat import XFeat

#/////////////////////////
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#////////////////////////

"""
python train.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

If we train with disk, we need to point out in the dataset_reader.py where to find the pre-extract disk feature

"""
def calculate_3d_coordinates(intrinsic_matrix, extrinsic_matrix, depth_map, pixel_coord):
    """
    Calculate the 3D coordinates from the pixel coordinate, depth map, intrinsic and extrinsic matrices.
    
    Args:
    - intrinsic_matrix (torch.Tensor): 3x3 camera intrinsic matrix.
    - extrinsic_matrix (torch.Tensor): 4x4 camera extrinsic matrix.
    - depth_map (torch.Tensor): A 2D depth map with depth values for each pixel.
    - pixel_coord (tuple): (u, v) pixel coordinate for which we want the 3D coordinate.
    
    Returns:
    - world_coord (torch.Tensor): 3D world coordinates of the point (x, y, z).
    """
    
    # Step 1: Invert the intrinsic matrix
    intrinsic_inv = torch.inverse(intrinsic_matrix)
    
    # Step 2: Get the pixel coordinates (u, v)
    u, v = pixel_coord
    
    # Step 3: Get the depth from the depth map at the pixel coordinate
    depth = depth_map[int(v)][int(u)]
    
    # Step 4: Convert 2D pixel to normalized camera coordinates (x_cam, y_cam, z_cam)
    # Apply inverse intrinsic matrix to the pixel coordinate (u, v)
    pixel_homogeneous = torch.tensor([u, v, 1], dtype=torch.double).to("cuda")
    normalized_cam_coords = torch.matmul(intrinsic_inv, pixel_homogeneous)
    
    # Step 5: Scale the normalized camera coordinates by depth to get the 3D camera coordinates
    x_cam = normalized_cam_coords[0] * depth
    y_cam = normalized_cam_coords[1] * depth
    z_cam = depth
    
    # Step 6: Convert to world coordinates using the extrinsic matrix
    # Extrinsic matrix defines the transformation from camera coordinates to world coordinates
    camera_coords = torch.tensor([x_cam, y_cam, z_cam, 1], dtype=torch.float32).to("cuda")  # Homogeneous coordinates
    
    # Apply the extrinsic matrix to get world coordinates
    world_coords = torch.matmul(extrinsic_matrix, camera_coords)
    
    return world_coords[:3]  # Return (x, y, z) world coordi


def find_2d3d_correspondences(keypoints, image_features, gaussian_pcd, gaussian_feat, chunk_size=10000):
    device = image_features.device
    f_N, feat_dim = image_features.shape
    P_N = gaussian_feat.shape[0]
    
    # Normalize features for faster cosine similarity computation
    image_features = F.normalize(image_features, p=2, dim=1)
    gaussian_feat = F.normalize(gaussian_feat, p=2, dim=1)
    
    max_similarity = torch.full((f_N,), -float('inf'), device=device)
    max_indices = torch.zeros(f_N, dtype=torch.long, device=device)
    
    for part in range(0, P_N, chunk_size):
        chunk = gaussian_feat[part:part + chunk_size]
        # Use matrix multiplication for faster similarity computation
        similarity = torch.mm(image_features, chunk.t())
        
        chunk_max, chunk_indices = similarity.max(dim=1)
        update_mask = chunk_max > max_similarity
        max_similarity[update_mask] = chunk_max[update_mask]
        max_indices[update_mask] = chunk_indices[update_mask] + part

    point_vis = gaussian_pcd[max_indices].cpu().numpy().astype(np.float64)
    point_vis_feature = gaussian_feat[max_indices].cpu().numpy()
    keypoints_matched = keypoints[..., :2].cpu().numpy().astype(np.float64)
    
    return keypoints_matched, point_vis, point_vis_feature

def get_match_gt(all_2d, matched_2d, all_feature, all_3d, device="cuda"):
    
    diff = all_2d.unsqueeze(0).to(device) - matched_2d.unsqueeze(1).to(device)  # Reshape for broadcasting
    # Check where the difference is zero (i.e., exact match)
    match_mask = torch.all(diff == 0, dim=2)  # Check equality along the last dimension (x and y)
    # Get the indices of the matches
    matching_indices_list = match_mask.nonzero(as_tuple=True)[1].to("cpu").numpy() if device=="cuda" else match_mask.nonzero(as_tuple=True)[1].numpy()
    return all_3d[matching_indices_list], all_feature[matching_indices_list]

def get_match_mass_center_density(matched_2d, xy_mass_center, device="cuda"):

    xys = xy_mass_center.t()[:,:2]
    massy_center = xy_mass_center.t()[:,[2,3,4]]
    density = xy_mass_center.t()[:,5]
    
    diff = xys.unsqueeze(0).to(device) - matched_2d.unsqueeze(1).to(device)  # Reshape for broadcasting
    # Check where the difference is zero (i.e., exact match)
    match_mask = torch.all(diff == 0, dim=2)  # Check equality along the last dimension (x and y)
    # Get the indices of the matches
    matching_indices_list = match_mask.nonzero(as_tuple=True)[1].to("cpu").numpy() if device=="cuda" else match_mask.nonzero(as_tuple=True)[1].numpy()
    return massy_center[matching_indices_list], density[matching_indices_list]


def getIntrinsic(view):
    K = np.eye(3)
    focal_length = fov2focal(view.FoVx, view.image_width)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2] = view.image_width / 2
    K[1, 2] = view.image_height / 2
    return K





def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    lftr = LocalFeatureTransformer()
    config = Config()
    refiner = Refiner(config)
    regulation_optimizer = torch.optim.SGD(refiner.parameters(), lr=1.e-6)
    xfeat = XFeat(top_k=4096)
    
    
    # 2D semantic feature map CNN decoder
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    gt_feature_map = viewpoint_cam.semantic_feature.cuda()
    feature_out_dim = gt_feature_map.shape[0]

    
    # speed up
    if dataset.speedup:
        feature_in_dim = int(feature_out_dim/4)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)


    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1]*128 if dataset.white_background else [0]*128
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

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
        
        query_img = viewpoint_cam.original_image[0:3, :, :]
        #Get the training camera view intrinsic
        query_K = getIntrinsic(viewpoint_cam)
        
        # Extract sparse features    
        # # [1,C,H,W] = [1,3,480,640]
        query_keypoints, _, query_feature = xfeat.detectAndCompute(query_img[None], 
                                                                 top_k=4096)[0].values()   #ref_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate
        

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        
        #//////////// Add the regulation module /////////////////
        depth_map = render_pkg["depth"] 
        xy_mass_center = render_pkg["xy_to_3D_ranges"].detach().to("cpu")

                        
        query_keypoints_3d = [calculate_3d_coordinates(torch.tensor(query_K).to("cuda"), viewpoint_cam.world_view_transform, depth_map.squeeze().detach(), kp) for kp in query_keypoints]
        query_keypoints_3d = torch.stack(query_keypoints_3d, dim=0)
        with torch.no_grad():
            gaussian_pcd = gaussians.get_xyz
            gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
            
            query_feature = torch.squeeze(lftr(query_feature[None].cpu())).to("cuda")
            query_feature = torch.nn.functional.normalize(query_feature,dim=0)

            matched_2d, matched_3d, match_3d_feature = find_2d3d_correspondences(
                    query_keypoints,
                    query_feature,
                    gaussian_pcd,
                    gaussian_feat
                )

        matched_gt_3d, matched_gt_feature = get_match_gt(query_keypoints, torch.tensor(matched_2d), query_feature,  query_keypoints_3d)
        gt_diff_3d = matched_gt_3d - torch.tensor(matched_3d).to("cuda")
        
        # Get the mass center with given 2D point
        #mass_centers,density = get_match_mass_center_density(torch.tensor(matched_2d), xy_mass_center)

        #mass_center_density = torch.cat([mass_centers, torch.transpose(density[None], 0,1)], dim=1)
        
        regulation_optimizer.zero_grad()

        pred_shift, gen_feature = refiner(torch.tensor(match_3d_feature), torch.tensor(matched_3d).to(torch.float32))
        regulation_loss = 0.6*l1_loss(pred_shift, gt_diff_3d.cpu()) + 0.4*l1_loss(gen_feature, matched_gt_feature.cpu()) 
        tb_writer.add_scalar("Loss/train_regulation", regulation_loss, iteration)
        print("regulation loss = ", regulation_loss)
        regulation_optimizer.step()
        #/////////////////////////        

        feature_map, image, viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        gt_feature_map = viewpoint_cam.semantic_feature.cuda() #64x48

        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) #640x480
        if dataset.speedup:
            feature_map = cnn_decoder(feature_map)
        Ll1_feature = l1_loss(feature_map, gt_feature_map) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 1.0 * Ll1_feature + 1.0* regulation_loss
        
        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background)) 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                print("\n[ITER {}] Saving feature decoder ckpt".format(iteration))
                 # Log and save 
                print("\n[ITER {}] Saving Regulation".format(iteration))
                torch.save(refiner.state_dict(), scene.model_path + "/regulation_" + str(iteration) + ".pth")
                if dataset.speedup:
                    torch.save(cnn_decoder.state_dict(), scene.model_path + "/decoder_chkpnt" + str(iteration) + ".pth")
  

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
                if dataset.speedup:
                    cnn_decoder_optimizer.step()
                    cnn_decoder_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

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

def training_report(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_feature', Ll1_feature.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
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
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

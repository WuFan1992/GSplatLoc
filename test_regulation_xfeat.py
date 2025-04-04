
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

import cv2
import numpy as np
import time
import torch
import torch.optim as optim

from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2, fov2focal

from encoders.XFeat.modules.xfeat import XFeat

from warping.warping_loss import *
from warping.warp_utils import *
from utils.loc_utils import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from reguler.network import *

"""
This file is the complet version of 2d_3d_xfeat.py that launch direct 2D 3D macthing within all the test image 
 command: 
python test_regulation_xfeat.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

"""

def getIntrinsic(view):
    K = np.eye(3)
    focal_length = fov2focal(view.FoVx, view.image_width)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2] = view.image_width / 2
    K[1, 2] = view.image_height / 2
    return K

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


def localize_set(model_path, name, views, gaussians, pipeline, background, args):


        prior_rErr = []
        prior_tErr = []
        inliers = []

        xfeat = XFeat(top_k=15)
        #Load model 
        refiner_pth = model_path + "/regulation_600.pth"
        config = Config()
        refiner = Refiner(config)
        refiner.load_state_dict(torch.load(refiner_pth, weights_only=True))
        refiner.eval()
        
        gaussian_pcd = gaussians.get_xyz
        gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
        
    
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            gt_im = view.original_image[0:3, :, :]
            # Extract sparse features
            gt_keypoints, _, gt_feature = xfeat.detectAndCompute(gt_im[None], 
                                                                 top_k=15)[0].values()

            # Define intrinsic matrix
            K = getIntrinsic(view)

            start = time.time()

            # Find initial pose prior via 2D-3D matching
            with torch.no_grad():
                matched_2d, matched_3d, matched_3d_feature = find_2d3d_correspondences(
                    gt_keypoints,
                    gt_feature,
                    gaussian_pcd,
                    gaussian_feat
                )

            gt_R = view.R
            gt_t = view.T

            print(f"Match speed: {time.time() - start}")
            #feature_matching_time.append(time.time()-start)
            _, R, t, inl = cv2.solvePnPRansac(matched_3d, matched_2d, 
                                                  K, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=args.ransac_iters
                                                  )
            
            R, _ = cv2.Rodrigues(R)    
            
             # Calculate the rotation and translation errors using existing function
            rotError, transError = calculate_pose_errors(gt_R, gt_t, R.T, t)

            # Print the errors
            print(f"Coarse Rotation Error: {rotError} deg")
            print(f"Coarse Translation Error: {transError} cm")
            
            
            cam_int = torch.Tensor(K).view(1,-1)
            cam_ext_R = torch.reshape(torch.Tensor(R), (1,9))
            cam_ext_T = torch.reshape(torch.Tensor(t), (1,3))
            cam_ext = torch.cat([cam_ext_R, cam_ext_T], dim=1)  #1x12
            cam_ext = torch.cat([cam_ext, torch.Tensor([[0,0,0,1]])], dim=1) #1x16

            pred_pos, _ = refiner(torch.tensor(matched_3d_feature), torch.tensor(matched_3d).to(torch.float32), cam_int, cam_ext)
            
            _, R, t, inl = cv2.solvePnPRansac(pred_pos.detach().numpy(), matched_2d, 
                                                  K, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=args.ransac_iters
                                                  )
            R, _ = cv2.Rodrigues(R)  
            # Calculate the rotation and translation errors using existing function
            rotError, transError = calculate_pose_errors(gt_R, gt_t, R.T, t)

            # Print the errors
            print(f"Fine Rotation Error: {rotError} deg")
            print(f"Fine Translation Error: {transError} cm")

            if inl is not None:
                inliers.append(len(inl))
                prior_rErr.append(rotError)
                prior_tErr.append(transError)
            
        err_mean_rot =  np.mean(prior_rErr)
        err_mean_trans = np.mean(prior_tErr)
        mean_inliers = np.mean(inliers) 
        #mean_match_time = np.mean(feature_matching_time)
        print(f"Rotation Average Error: {err_mean_rot} deg ")
        print(f"Translation Average Error: {err_mean_trans} cm ") 
        print(f"Mean inliers : {mean_inliers}  ")
        #print(f"Running time = ", runing_time)
        #print(f"Mean match time = ", mean_match_time)
       

def launch_inference(dataset : ModelParams, pipeline : PipelineParams, args): 
         
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    localize_set(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, args)


if __name__ == "__main__":
# Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--top_k", default=4096, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    parser.add_argument("--warp_lr", default=0.0005, type=float)
    parser.add_argument("--warp_iters", default=251, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    launch_inference(model.extract(args), pipeline.extract(args), args)
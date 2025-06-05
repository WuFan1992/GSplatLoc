
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

from diffestimator.model import *
from scene.feat_pointcloud import *

from utils.refiner import refiner

"""
This file is the complet version of 2d_3d_xfeat.py that launch direct 2D 3D macthing within all the test image 
 command: 
python test_3dgs_with_moving_feat.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

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

def get_match_gt(all_2d, matched_2d, all_feature, device="cuda"):
    
    diff = all_2d.unsqueeze(0).to(device) - matched_2d.unsqueeze(1).to(device)  # Reshape for broadcasting
    # Check where the difference is zero (i.e., exact match)
    match_mask = torch.all(diff == 0, dim=2)  # Check equality along the last dimension (x and y)
    # Get the indices of the matches
    matching_indices_list = match_mask.nonzero(as_tuple=True)[1].to("cpu").numpy() if device=="cuda" else match_mask.nonzero(as_tuple=True)[1].numpy()
    return all_feature[matching_indices_list]

def calculate_fine_pose_errors(R_gt, t_gt, R_est, t_est):
    
    #Point numbers 
    N = R_est.shape[0]
    
    repeated_R_gt = np.tile(R_gt, (N, 1, 1))
    repeated_T_gt = np.tile(t_gt, (N, 1,1))
    #
    rotError = np.array([np.matmul(R_est[i].T, repeated_R_gt[i]) for i in range(N)])
    rotError = np.array([cv2.Rodrigues(rotError[i])[0] for i in range(N)])
    rotError = np.array([np.linalg.norm(rotError[i]) * 180 / np.pi for i in range(N)])
    rotError = np.mean(rotError)
    
    
    transError = np.array([np.linalg.norm(repeated_T_gt[i] - t_est[i]) * 100 for i in range(N)])
    
    # Calculate translation error
    transError = np.mean(transError)
    
    return rotError, transError



"""
Use to get the gt 3d coord from query keypoint
"""
def new_calculate_ndc2camera(proj_matrix, xndc, yndc, depth):
    a1 = proj_matrix[0,0]
    a2 = proj_matrix[0,1]
    a3 = proj_matrix[0,2]
    a4 = proj_matrix[0,3]
    
    a5 = proj_matrix[1,0]
    a6 = proj_matrix[1,1]
    a7 = proj_matrix[1,2]
    a8 = proj_matrix[1,3]
    
    
    a13 = proj_matrix[3,0]
    a14 = proj_matrix[3,1]
    a15 = proj_matrix[3,2]
    a16 = proj_matrix[3,3]
    
    A1 = a1-xndc*a13
    B1 = a2-xndc*a14
    C1 = (a3-xndc*a15)*depth+a4-xndc*a16
    
    A2 = a5-yndc*a13
    B2 = a6-yndc*a14
    C2 = (a7-yndc*a15)*depth+a8-yndc*a16
    
    X = (-C1*B2+C2*B1)/(A1*B2-A2*B1)
    Y = (-A1*C2+A2*C1)/(A1*B2-A2*B1)
    
    return X, Y

def pixel2ndc(pixel, S):
    return (((pixel/0.5)+1.0)/S)-1.0


def getGTXYZ(camera2ndc, view2camera, point_2d, depth_map):
    
    #Get the depth value
    depth_map = depth_map.detach().squeeze(0)
    depth = depth_map[point_2d[:,1].int().to("cpu"), point_2d[:,0].int().to("cpu")] 
    X, Y = new_calculate_ndc2camera(camera2ndc.transpose(0,1), pixel2ndc(point_2d[:,0], 640), pixel2ndc(point_2d[:,1], 480), depth)
    ones = torch.tensor([1.0]).repeat(point_2d.size(0)).to("cuda")
        
    cam_coord_inv = torch.stack([X, Y, depth, ones], dim=1)
    output = torch.matmul(cam_coord_inv.double(), torch.inverse(view2camera).double())
    return output[:, :3]


def localize_set(model_path, name, views, gaussians, pipeline, background, args, feat_pc):


        prior_rErr = []
        prior_tErr = []
        inliers = []
        
        refine_rErr = []
        refine_tErr = []
        
        time_coarse = []
        time_fine = []

        xfeat = XFeat(top_k=4096)
        
        feat_pcd = torch.tensor(feat_pc.get_xyz).to("cuda")
        feat_feat = torch.tensor(feat_pc.get_semantic_feature.squeeze(-1)).to("cuda")
        

    
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):

            
            gt_im = view.original_image[0:3, :, :]
            # Extract sparse features
            gt_keypoints, _, gt_feature = xfeat.detectAndCompute(gt_im[None], 
                                                                 top_k=4096)[0].values()

            # Define intrinsic matrix
            K = getIntrinsic(view)

            start = time.time()

            # Find initial pose prior via 2D-3D matching
            with torch.no_grad():
                matched_2d, matched_3d, matched_3d_feature = find_2d3d_correspondences(
                    gt_keypoints,
                    gt_feature,
                    feat_pcd,
                    feat_feat
                )
                
            
                # get the coarse pose 
                _, R, t, inl = cv2.solvePnPRansac(matched_3d, matched_2d, 
                                                  K, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=args.ransac_iters
                                                  )

                time_coarse.append(time.time() - start)
                R, _ = cv2.Rodrigues(R) 
                
                gt_R = view.R
                gt_t = view.T   
                
                 # Calculate the rotation and translation errors using existing function
                rotError, transError = calculate_pose_errors(gt_R, gt_t, R.T, t)

                # Print the errors
                print(f"Coarse Rotation Error: {rotError} deg")
                print(f"Coarse Translation Error: {transError} cm")
                
                # Fine Pose
                rotError_fine =0
                transError_fine = 0
                start_refine = time.time()
                for i in range(20):
                    view.update_RT(R.T, t[:,0])
                    updated_matched_2d, updated_matched_3d = refiner(matched_2d, matched_3d,  view.full_proj_transform, feat_pcd)

                    _, fine_R, fine_t, inl = cv2.solvePnPRansac(updated_matched_3d.cpu().numpy(), updated_matched_2d, 
                                                  K, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=args.ransac_iters
                                                  )
                
                    fine_R, _ = cv2.Rodrigues(fine_R) 
                    rotError_fine, transError_fine = calculate_pose_errors(gt_R, gt_t, fine_R.T, fine_t)
                
                    
                    matched_2d = updated_matched_2d
                    matched_3d = updated_matched_3d
                    R, t = fine_R, fine_t
                time_fine.append(time.time()-start_refine)
                
                print(f"Fine Rotation Error: {rotError_fine} deg")
                print(f"Fine Translation Error: {transError_fine} cm")
                if inl is not None:
                    inliers.append(len(inl))
                    prior_rErr.append(rotError)
                    prior_tErr.append(transError)
                    refine_rErr.append(rotError_fine)
                    refine_tErr.append(transError_fine)
            
        err_mean_rot =  np.mean(prior_rErr)
        err_mean_trans = np.mean(prior_tErr)
        mean_inliers = np.mean(inliers)
        
        err_mean_refine_rot = np.mean(refine_rErr)
        err_mean_refine_trans = np.mean(refine_tErr) 
        
        mean_coarse_time = np.mean(time_coarse)
        mean_fine_time = np.mean(time_fine)

        print(f"Rotation Coarse Average Error: {err_mean_rot} deg ")
        print(f"Translation CoarseAverage Error: {err_mean_trans} cm ") 
        print(f"Rotation Fine Average Error: {err_mean_refine_rot} deg ")
        print(f"Translation Fine Average Error: {err_mean_refine_trans} cm ") 
        print(f"Time Coarse : {mean_coarse_time} s ")
        print(f"Time Fine : {mean_fine_time} s ")
        print(f"Mean inliers : {mean_inliers}  ")

       

def launch_inference(dataset : ModelParams, pipeline : PipelineParams, args): 
    
     #Load the feature point cloud
    feat_pc = FeatPointCloud()
    feat_pc.load_ply(os.path.join(dataset.model_path,"feature_point_cloud",
                                                      "iteration_15000" ,
                                                      "feature_point_cloud.ply"))  
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, load_gaussian=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

   
    
    localize_set(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, args, feat_pc)


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
    
    

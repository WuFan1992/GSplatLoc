
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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from tqdm import tqdm
from PIL import Image

from scene import Scene
from utils.general_utils import image_process
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from encoders.XFeat.modules.xfeat import XFeat
from utils.loc_utils import *
from utils.refiner import *


from scene.feat_pointcloud import *
from utils.pose_utils import find_2d3d_correspondences, getIntrinsic

from scipy.optimize import least_squares

import torch.nn as nn
from scipy.sparse import lil_matrix


"""
This file is the complet version of 2d_3d_xfeat.py that launch direct 2D 3D macthing within all the test image 
 command: 
python sparse_coarse2fine.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

"""

# Dummy projection function
def ndc2pixel(x, size):
    return (x + 1) * 0.5 * size

    
def bundle_adjustment_sparsity_one_cam(n_points):
    n = 12 + n_points * 3
    m = 2 * n_points  # 残差数：每个点2个残差
    A = lil_matrix((m, n), dtype=float)

    # 12个相机参数对所有残差都有影响
    for i in range(m):
        for s in range(12):
            A[i, s] = 1

    # 每个点的3个参数只影响对应的两个残差（行）
    for l in range(n_points):
        for s in range(3):
            A[2*l, 12 + l*3 + s] = 1       # x残差对应参数
            A[2*l + 1, 12 + l*3 + s] = 1   # y残差对应参数

def projection(X_3D, R, t, proj_matrix, W, H):
        # X_3D shape: (N, 3)
        # R shape: (3,3)
        # t shape: (3,)

        # x_cam = (X_3D @ R.T).T
        x_cam = (X_3D @ R.T).T    # shape (3, N)

        # factor_cam = (X_3D @ t.reshape(3,1)).squeeze(-1) + 1
        factor_cam = (X_3D @ t.reshape(3,1)).squeeze() + 1  # shape (N,)
        factor_cam = factor_cam * 100

        proj_mat_3x3 = proj_matrix[:3, :3]   # (3,3)

        # ndc_coord = (x_cam.T @ proj_mat_3x3.T) + factor_cam.unsqueeze(1) * proj_matrix[3, :3]
        # x_cam.T shape: (N,3), proj_mat_3x3.T (3,3) -> (N,3)
        ndc_coord = (x_cam.T @ proj_mat_3x3.T) + factor_cam[:, np.newaxis] * proj_matrix[3, :3]

        # factor = x_cam.T @ proj_matrix[:3, 3].unsqueeze(1) + factor_cam.unsqueeze(1) * proj_matrix[3, 3]
        factor = (x_cam.T @ proj_matrix[:3, 3].reshape(3,1)).squeeze() + factor_cam * proj_matrix[3, 3]

        weight = 1.0 / (factor + 1e-6)   # shape (N,)

        # x = ndc2pixel(ndc_coord[:, 0].unsqueeze(1) * weight, W)
        # ndc_coord[:,0] shape (N,), weight (N,)
        x = ndc2pixel(ndc_coord[:, 0] * weight, W)   # shape (N,)

        y = ndc2pixel(ndc_coord[:, 1] * weight, H)   # shape (N,)

        # u_proj = torch.cat([x, y], dim=1).squeeze()
        # x,y shape (N,), stack成 (N,2)
        u_proj = np.stack([x, y], axis=1)   # (N,2)

        return u_proj

def fun(param, n_points, proj_mat, x_2d):
    R = param[:9].reshape(3,3)
    t = param[9:12].reshape(3,)
    p3d = param[12:].reshape(n_points, 3)
    
    u_proj = projection(p3d, R, t, proj_mat, 640, 480)
    
    residuals = (u_proj - x_2d).ravel()  
    return residuals
    
        
                        

def localize_set(model_path, views, args, feat_pc):

        prior_rErr = []
        prior_tErr = []
        refine_rErr = []
        refine_tErr = []
        inliers = []
        list_ratio = []

        xfeat = XFeat(top_k=4096)
        

        feat_pcd = torch.tensor(feat_pc.get_xyz).to("cuda")
        feat_feat = torch.tensor(feat_pc.get_semantic_feature.squeeze(-1)).to("cuda")
        
        start = time.time()
    
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
                              
            # Get test image 
            try:
                image = Image.open(view.image_path) 
            except:
                print(f"Error opening image: {view.image_path}")
                continue

            original_image = image_process(image)
            gt_im = original_image.cuda()
            
            img_width = gt_im.shape[2]
            img_height = gt_im.shape[1]
            # Extract sparse features
            gt_keypoints, _, gt_feature = xfeat.detectAndCompute(gt_im[None], 
                                                                 top_k=4096)[0].values()

            # Define intrinsic matrix
            K = getIntrinsic(view, img_width, img_height)


            # Find initial pose prior via 2D-3D matching
            
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
            
            
            R, _ = cv2.Rodrigues(R) 
                
            gt_R = view.R
            gt_t = view.T   
                            
            # Calculate the rotation and translation errors using existing function
            rotError, transError = calculate_pose_errors(gt_R, gt_t, R.T, t)
            coarse_ratio = len(inl)/len(matched_3d)
            # Print the errors
            print(f"Coarse Rotation Error: {rotError} deg")
            print(f"Coarse Translation Error: {transError} cm")
                
            rotError_fine =0
            transError_fine = 0
            
            proj_matrix = view.projection_matrix.cpu().numpy()
            #proj_matrix[3, 3] = 0.01
            n_points = len(matched_2d)
            # R, t and matched_3d to one dimension
            R_init = R.ravel()
            t_init = t.ravel()
            matched_3d_init = matched_3d.ravel()
            
            init_Rt_p3d = np.concatenate([R_init, t_init, matched_3d_init], axis=0)
            A = bundle_adjustment_sparsity_one_cam(n_points)

            res = least_squares(fun, init_Rt_p3d, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_points, proj_matrix, matched_2d), max_nfev=25)
            optim_param = res.x
            fine_R = optim_param[:9].reshape(3,3)
            fine_t = optim_param[9:12].reshape(3,1)
            
            
            rotError_fine, transError_fine = calculate_pose_errors(gt_R, gt_t, fine_R.T, fine_t)
                
                

                
            print(f"Fine Rotation Error: {rotError_fine} deg")
            print(f"Fine Translation Error: {transError_fine} cm")
               
                
            if inl is not None:
                inliers.append(len(inl))
                prior_rErr.append(rotError)
                prior_tErr.append(transError)
                list_ratio.append(coarse_ratio)
                refine_rErr.append(rotError_fine)
                refine_tErr.append(transError_fine)
                print("mean coar trans - rot, mean fine trans - rot =  ", np.mean(prior_tErr), np.mean(prior_rErr), np.mean(refine_tErr) , np.mean(refine_rErr))
            
        err_mean_rot =  np.mean(prior_rErr)
        err_mean_trans = np.mean(prior_tErr)
        mean_inliers = np.mean(inliers)
        mean_ratio = np.mean(list_ratio) 
        err_mean_refine_rot = np.mean(refine_rErr)
        err_mean_refine_trans = np.mean(refine_tErr) 

        print(f"Rotation Average Error: {err_mean_rot} deg ")
        print(f"Translation Average Error: {err_mean_trans} cm ")
        print(f"Rotation Fine Average Error: {err_mean_refine_rot} deg ")
        print(f"Translation Fine Average Error: {err_mean_refine_trans} cm ")  
        print(f"Mean inliers : {mean_inliers}  ")
        print(f"Mean ratio : {mean_ratio}  ")
        log_errors(model_path, "test", prior_rErr, prior_tErr, "coarse")
        log_errors(model_path, "test", refine_rErr, refine_tErr, f"refine")

       
       

def launch_inference(dataset : ModelParams,  args): 
    
     #Load the feature point cloud
    feat_pc = FeatPointCloud()
    feat_pc.load_ply(os.path.join(dataset.model_path,"feature_point_cloud",
                                                      "iteration_"+ str(args.iteration),
                                                      "feature_point_cloud.ply"))  
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, load_gaussian=False)
    
    localize_set(dataset.model_path, scene.getTestCameras(), args, feat_pc)


if __name__ == "__main__":
# Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--top_k", default=4096, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    launch_inference(model.extract(args), args)
    
    

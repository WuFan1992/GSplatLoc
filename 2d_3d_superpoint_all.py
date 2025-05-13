
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


"""
#### Import Superpoint or R2D2 Feature Extractor #######
"""
from encoders.feature_extractor import FeatureExtractor

from warping.warping_loss import *
from warping.warp_utils import *
from utils.loc_utils import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

"""
This file is the complet version of 2d_3d_xfeat.py that launch direct 2D 3D macthing within all the test image 
 command: 
python 2d_3d_superpoint_all.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

"""

from torch import Tensor
def getTopKindices(score_map: Tensor, top_k=20):
    """
    Get the indice (x, y) of the top_k score in the score_map
    score_map : [480,640]
    """
    # Flatten the score map
    flatten_sm = score_map.view(-1)

    _, topk_indices = torch.topk(flatten_sm, top_k)
    
    # Convert flat indices to 2D coordinates
    coords = torch.stack((topk_indices // score_map.size(1), topk_indices % score_map.size(1)), dim=1)
    
    return coords

def getTopKFeat(coords: list, feature_map:Tensor):
    """
    Get the feature value given the coords list 
    """
    Dim = feature_map.shape[0]
    
    # Convert coordinates to tensors
    coords = torch.tensor(coords, dtype=torch.long) # (N,2)
    x = coords[:,0]
    y = coords[:,1]
    
    # Expand x and y for each channel
    N = coords.shape[0]
    x_expand = x.unsqueeze(0).expand(Dim, N)  # (D, N)
    y_expand = y.unsqueeze(0).expand(Dim, N)  # (D, N)
    
    # Create a channel index
    channel_indices = torch.arange(Dim).unsqueeze(1).expand(Dim, N)  # (D, N)

    # Step 2: Use advanced indexing to get values at each (d, x, y)
    # feature_map is (D, W, H), so access as: feature_map[d, x, y]
    values = feature_map[channel_indices, x_expand, y_expand]  # shape: (D, N)

    # Step 3: Transpose to (N, D)
    values = values.permute(1, 0)  # shape: (N, D)
    
    return values


def localize_set(model_path, name, views, gaussians, pipeline, background, args):

        # Keep track of rotation and translation errors for calculation of the median error.
        rErrs = []
        tErrs = []

        prior_rErr = []
        prior_tErr = []
        pnp_p = []
        inliers = []

        feature_extractor = FeatureExtractor("sp").cuda().eval()
        

        gaussian_pcd = gaussians.get_xyz
        gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
        

    
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            gt_im = view.original_image[0:3, :, :]
            feat  = feature_extractor(gt_im[None].cuda())
            orig_feat_map = feat["original_feature_map"][0]
            scores = feat["scores"][0]
            # Extract sparse features
            gt_keypoints = getTopKindices(scores.squeeze(0), 4800)
            gt_feature = getTopKFeat(gt_keypoints, orig_feat_map)

            # Define intrinsic matrix
            K = np.eye(3)
            focal_length = fov2focal(view.FoVx, view.image_width)
            K[0, 0] = K[1, 1] = focal_length
            K[0, 2] = view.image_width / 2
            K[1, 2] = view.image_height / 2

            start = time.time()

            # Find initial pose prior via 2D-3D matching
            with torch.no_grad():
                matched_3d, matched_2d = find_2d3d_correspondences(
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
            print(f"Rotation Error: {rotError} deg")
            print(f"Translation Error: {transError} cm")

            if inl is not None:
                inliers.append(len(inl))
                prior_rErr.append(rotError)
                prior_tErr.append(transError)
        
        #runing_time = time.time() - start_time 
            
        err_mean_rot =  np.mean(prior_rErr)
        err_mean_trans = np.mean(prior_tErr)
        mean_inliers = np.mean(inliers) 
        #mean_match_time = np.mean(feature_matching_time)
        print(f"Rotation Average Error: {err_mean_rot} deg ")
        print(f"Translation Average Error: {err_mean_trans} cm ") 
        print(f"Mean inliers : {mean_inliers}  ")
        #print(f"Running time = ", runing_time)
        #print(f"Mean match time = ", mean_match_time)
        """

            w2c = torch.eye(4, 4, device='cuda')
            w2c[:3, :3] = torch.from_numpy(R).float()
            w2c[:3, 3] = torch.from_numpy(t[:, 0]).float()
            
            # Update the view's pose
            view.update_RT(R.T, t[:,0])
            
            # Render from the current estimated pose
            with torch.no_grad():
                render_pkg = render(view, gaussians, pipeline, background)
            
            render_im = render_pkg["render"]
            depth = render_pkg["depth"]

            quat_opt = rotmat2qvec_tensor(w2c[:3, :3].clone()).view([4]).to(w2c.device)
            t_opt = w2c[:3, 3].clone()

            optimizer = optim.Adam([quat_opt.requires_grad_(True), 
                                    t_opt.requires_grad_(True)], lr=args.warp_lr)


            for i in range(args.warp_iters):                    
                    
                # Compute warp loss for optimizing w2c_opt
                optimizer.zero_grad()
       
                loss = compute_warping_loss(vr=render_im,
                                            qr=gt_im,
                                            quat_opt=quat_opt,
                                            t_opt=t_opt,
                                            pose=w2c,
                                            K=torch.from_numpy(K).float().to('cuda'),
                                            depth=depth) 
        
                loss.backward()
                optimizer.step()
                
                if i % (args.warp_iters // 5) == 0:
                    print(f"Iteration {i}, Loss: {loss.item():.4f}")
                    
                    # After optimization, update the view's pose
                    R_est = qvec2rotmat_tensor(quat_opt).detach().cpu().numpy()
                    t_est = t_opt.detach().cpu().numpy()

                    # Compute final errors
                    rotError, transError = calculate_pose_errors(gt_R, gt_t, R_est.T, t_est.reshape(3,1))

                    print(f"Iteration {i} Rotation Error: {rotError:.2f} deg, Translation Error: {transError:.2f} cm")
                
            # After optimization, update the view's pose
            R_est = qvec2rotmat_tensor(quat_opt).detach().cpu().numpy()
            t_est = t_opt.detach().cpu().numpy()
            
            # Compute final errors
            rotError, transError = calculate_pose_errors(gt_R, gt_t, R_est.T, t_est.reshape(3,1))

            print(f"Final Rotation Error: {rotError:.2f} deg, Translation Error: {transError:.2f} cm")
            
            rErrs.append(rotError)
            tErrs.append(transError)
                        
            print(f"Processed: {view.uid}")            
        
        #////////////////////////////////////////////////////////
        #log_errors(model_path, name, prior_rErr, prior_tErr, f"prior")
        #log_errors(model_path, name, rErrs, tErrs, "warp")
        """

def launch_inference(dataset : ModelParams, pipeline : PipelineParams, args): 
         
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #///////////////////////////////
    #localize_set(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, args)
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
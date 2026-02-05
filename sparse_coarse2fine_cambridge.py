
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
import pickle

from scene import Scene
from utils.general_utils import safe_state, image_process
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from encoders.XFeat.modules.xfeat import XFeat
from utils.loc_utils import *
import utils.global_var as global_var 


from scene.feat_pointcloud import *
from utils.refiner import refiner, extract_patch_features_with_coords
from utils.pose_utils import find_2d3d_correspondences, getIntrinsic

"""
This file is the complet version of 2d_3d_xfeat.py that launch direct 2D 3D macthing within all the test image 
 command: 
python sparse_coarse2fine.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

"""

time_feature_extraction = []
time_sparse_matching = []
time_coarse_matching = []
time_coarse_pose_estim = []



def localize_set(model_path, views, args, feat_pc):

        prior_rErr = []
        prior_tErr = []
        refine_rErr = []
        refine_tErr = []
        inliers = []
        list_ratio = []
        coarse_time = []
        fine_time = []

        xfeat = XFeat(top_k=4096)
        
            # load masks
        masks = None
        if os.path.exists(os.path.join(args.source_path, "masks.pkl")):
            print(
            "Loading masks from",
            os.path.join(args.source_path, "masks.pkl"),
            )
            masks = pickle.load(
            open(os.path.join(args.source_path, "masks.pkl"), "rb")
        )
        

        feat_pcd = torch.tensor(feat_pc.get_xyz).to("cuda")
        feat_feat = torch.tensor(feat_pc.get_semantic_feature.squeeze(-1)).to("cuda")
        
    
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            # Get test image 
            try:
                image = Image.open(view.image_path) 
            except:
                print(f"Error opening image: {view.image_path}")
                continue

            original_image = image_process(image)
            
            
            
            if masks is not None:
                # use mask
                obj_mask = masks[view.image_name][0].cuda()[None]
                sky_mask = masks[view.image_name][1].cuda()[None]
                distort_mask = masks[view.image_name][2].cuda()[None]

                # mask obj and distort border
                mask_sparse = obj_mask & distort_mask
                
                
                mask_sparse = F.interpolate(
                    mask_sparse.unsqueeze(0).float(),
                    size=(original_image.shape[1], original_image.shape[2]),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0) > 0.5
                
                sky_mask = F.interpolate(
                    sky_mask.unsqueeze(0).float(),
                    size=(original_image.shape[1], original_image.shape[2]),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0) > 0.5
                
                gt_im = original_image.cuda()
                gt_im = gt_im * mask_sparse
                # mask sky
                gt_im[sky_mask.repeat(3, 1, 1) == False] = 1  # 全白
            
            
            
            

            img_width = gt_im.shape[2]
            img_height = gt_im.shape[1]
            # Extract sparse features
            start_time = time.time()
            gt_keypoints, _, gt_feature = xfeat.detectAndCompute(gt_im[None], 
                                                                 top_k=4096)[0].values()
            time_feature_extraction.append(time.time()-start_time)
            
            # Define intrinsic matrix
            K = getIntrinsic(view, img_width, img_height)

            # Find initial pose prior via 2D-3D matching
            start_time_c = time.time()
            matched_2d, matched_3d, matched_3d_feature = find_2d3d_correspondences(
                    gt_keypoints,
                    gt_feature,
                    feat_pcd,
                    feat_feat
            )
            time_coarse_matching.append(time.time()-start_time_c)
               
            # get the coarse pose 
            start_time = time.time()
            _, R, t, inl = cv2.solvePnPRansac(matched_3d, matched_2d, 
                                                  K, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=args.ransac_iters
                                            )

            time_coarse_pose_estim.append(time.time()-start_time)
            
            end_time_c = time.time()
            coarse_time.append(end_time_c-start_time_c)
            
            R, _ = cv2.Rodrigues(R) 
                
            gt_R = view.R
            gt_t = view.T   
                
            # Calculate the rotation and translation errors using existing function
            rotError, transError = calculate_pose_errors(gt_R, gt_t, R.T, t)
            if inl is not None:
                coarse_ratio = len(inl)/len(matched_3d)
                # Print the errors
                print(f"Coarse Rotation Error: {rotError} deg")
                print(f"Coarse Translation Error: {transError} cm")
                
                rotError_fine =0
                transError_fine = 0
                
                # Extract 8x8 patch feature
                gt_feature_map = xfeat.get_descriptors(gt_im[None])[0]
                if masks is not None:
                    feature_map_mask = (
                    F.interpolate(
                        mask_sparse[None].float(),
                        size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    > 0.5
                )
                gt_feature_map = gt_feature_map*feature_map_mask
                
                feature_map = F.interpolate(gt_feature_map.unsqueeze(0), size=(img_height, img_width), mode='bilinear', align_corners=True).squeeze(0) #640x480 
                
                patch_feat, patch_coord = extract_patch_features_with_coords(torch.tensor(matched_2d).cuda(), feature_map)
                patch_feat = F.normalize(patch_feat, dim=2)
            
   
                view.update_RT(R.T, t[:,0])
            
                start_time_f = time.time()
                
                for i in range(5):
                    view, updated_3d, mask,  fine_R, fine_t, inl = refiner(matched_2d, matched_3d,  matched_3d_feature ,view,  feat_pcd, feat_feat, patch_coord, patch_feat, K)
          

                    rotError_fine, transError_fine = calculate_pose_errors(gt_R, gt_t, fine_R.T, fine_t)
                    
                    if (transError_fine > transError):
                        transError_fine = transError
                    if (rotError_fine > rotError):
                        rotError_fine = rotError
                    
                
                    matched_3d = updated_3d
                    matched_2d = matched_2d[mask]
                    matched_3d_feature = matched_3d_feature[mask]
                    patch_coord = patch_coord[mask]
                    patch_feat = patch_feat[mask]

                end_time_f = time.time()
                
                print(f"Fine Rotation Error: {rotError_fine} deg")
                print(f"Fine Translation Error: {transError_fine} cm")
               
                
                if inl is not None:
                    inliers.append(len(inl))
                    prior_rErr.append(rotError)
                    prior_tErr.append(transError)
                    list_ratio.append(coarse_ratio)
                    refine_rErr.append(rotError_fine)
                    refine_tErr.append(transError_fine)
                    fine_time.append((end_time_f-start_time_f)/4)
                    #print("mean coar trans - rot, mean fine trans - rot =  ", np.mean(prior_tErr), np.mean(prior_rErr), np.mean(refine_tErr) , np.mean(refine_rErr))
            
        err_mean_rot =  np.mean(prior_rErr)
        err_mean_trans = np.mean(prior_tErr)
        mean_inliers = np.mean(inliers)
        mean_ratio = np.mean(list_ratio) 
        err_mean_refine_rot = np.mean(refine_rErr)
        err_mean_refine_trans = np.mean(refine_tErr)
        mean_c_time = np.mean(coarse_time) 
        mean_f_time = np.mean(fine_time)

        print(f"Rotation Average Error: {err_mean_rot} deg ")
        print(f"Translation Average Error: {err_mean_trans} cm ")
        print(f"Rotation Fine Average Error: {err_mean_refine_rot} deg ")
        print(f"Translation Fine Average Error: {err_mean_refine_trans} cm ")  
        print(f"Mean inliers : {mean_inliers}  ")
        print(f"Mean ratio : {mean_ratio}  ")
        print(f"Mean coarse time : {mean_c_time} ")
        print(f"Mean fine time : {mean_f_time}  ")
        print("Sparse feature extraction: ", np.mean(time_feature_extraction))
        print("Sparse coarse matching: ", np.mean(time_coarse_matching))
        print("Sparse pose estimation : ", np.mean(time_coarse_pose_estim))
        print("Optim pose   = ", np.mean(global_var.time_optim_pose))
        print("Optim 3D  = ", np.mean(global_var.time_optim_3D))
        
        log_errors(model_path, "test", prior_rErr, prior_tErr, "coarse")
        log_errors(model_path, "test", refine_rErr, refine_tErr, f"refine")

       
       

def launch_inference(dataset : ModelParams,  args): 
    
     #Load the feature point cloud
    feat_pc = FeatPointCloud()
    feat_pc.load_ply(os.path.join(dataset.model_path,"kpalign_point_cloud",
                                                      "iteration_"+ str(args.iteration),
                                                      "kpalign_point_cloud.ply"))  
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
    parser.add_argument("--ransac_iters", default=1000, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    launch_inference(model.extract(args), args)
    
    

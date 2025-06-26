
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
import random

from scene import Scene
from utils.general_utils import safe_state, image_process
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from encoders.XFeat.modules.xfeat import XFeat
from utils.loc_utils import *
from utils.debug_utils import Open3DModel


from scene.feat_pointcloud import *
from utils.refiner import refiner, extract_patch_features_with_coords
from utils.pose_utils import find_2d3d_correspondences, getIntrinsic
from utils.debug_utils import drawKps_query
from gaussian_renderer import render
from utils.pose_utils import getGTXYZ

"""
This file is the complet version of 2d_3d_xfeat.py that launch direct 2D 3D macthing within all the test image 
 command: 
python test_3dgs_with_moving_feat.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

"""


def localize_set(dataset, views, gaussians, pipeline, args, feat_pc):


        xfeat = XFeat(top_k=4096)
        

        feat_pcd = torch.tensor(feat_pc.get_xyz).to("cuda")
        feat_feat = torch.tensor(feat_pc.get_semantic_feature.squeeze(-1)).to("cuda")

        # Load the open3D model         
        pc_path = dataset.source_path + "/sparse/0"
        open3d_model = Open3DModel()
        open3d_model.read_model(pc_path, ".bin")
        
        #Set  Coarse matching point Color, Fine matching point Color and GT position
        coarse_color = [(153, 153, 255),    
          (153, 255, 153),    
          (255, 153, 153)]
        
        fine_color = [(102, 102, 255),    
          (102, 255, 102),    
          (255, 102, 102)]
        
        gt_color =  [(0, 0, 255),    
          (0, 255, 0),    
          (255, 0, 0)]    
        
        
        bg_color = [1]*64 if dataset.white_background else [0]*64
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                

    
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
            
            # Randomly retrieve matched kp for demonstration
            kp_num = len(matched_2d)
            rand_kp_indices = random.sample(range(kp_num+1), 3)
            drawKps_query(matched_2d[rand_kp_indices].astype(int), view.image_path)
            
            # Render depth map in order to get the gt 3D position
            render_pkg = render(view, gaussians, pipeline, background, img_width, img_height)
            depth_map = render_pkg["depth"] 
            gt_kps_3d = getGTXYZ(view.projection_matrix, view.world_view_transform, torch.tensor(matched_2d[rand_kp_indices]).cuda(), depth_map)
            coarse_kp_3d = matched_3d[rand_kp_indices]

            # get the coarse pose 
            _, R, t, _ = cv2.solvePnPRansac(matched_3d, matched_2d, 
                                                  K, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=args.ransac_iters
                                            )
            
            
            R, _ = cv2.Rodrigues(R) 
                

                
            # Extract 8x8 patch feature
            gt_feature_map = xfeat.get_descriptors(gt_im[None])[0]

            feature_map = F.interpolate(gt_feature_map.unsqueeze(0), size=(img_height, img_width), mode='bilinear', align_corners=True).squeeze(0) #640x480 
                
            patch_feat, patch_coord = extract_patch_features_with_coords(torch.tensor(matched_2d).cuda(), feature_map)
            patch_feat = F.normalize(patch_feat, dim=2)

            view.update_RT(R.T, t[:,0])
                
            for i in range(25):
                view, updated_3d, fine_R, fine_t, inl = refiner(matched_2d, matched_3d,  matched_3d_feature ,view,  feat_pcd, feat_feat, patch_coord, patch_feat, K)
                
                matched_3d = updated_3d
            
            fine_kp_3d = matched_3d[rand_kp_indices]
            
            open3d_model.create_window()

    
            open3d_model.add_keypoint(gt_kps_3d.cpu().numpy(),gt_color )
            open3d_model.add_keypoint(coarse_kp_3d, coarse_color)
            open3d_model.add_keypoint(fine_kp_3d.cpu().detach().numpy(), fine_color)
            open3d_model.add_points()
            open3d_model.show()
            
            
            
            
               
                

        
       
       

def launch_inference(dataset : ModelParams, opt: OptimizationParams,  pipeline: PipelineParams,   args): 
    
     #Load the feature point cloud
    feat_pc = FeatPointCloud()
    feat_pc.load_ply(os.path.join(dataset.model_path,"feature_point_cloud",
                                                      "iteration_"+ str(args.iteration),
                                                      "feature_point_cloud.ply"))  
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.training_setup(opt)
    
    localize_set(dataset, scene.getTestCameras(), gaussians, pipeline,  args, feat_pc)


if __name__ == "__main__":
# Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--top_k", default=4096, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    launch_inference(model.extract(args), op.extract(args), pipeline.extract(args), args)
    
    

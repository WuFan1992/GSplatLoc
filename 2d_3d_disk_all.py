
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


from warping.warping_loss import *
from warping.warp_utils import *
from utils.loc_utils import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from torch_dimcheck import dimchecked

from submodules.disk.disk.model.disk import DISK
from submodules.disk.disk.common.structs import Features


"""
This file is the complet version of 2d_3d_disk.py that launch direct 2D 3D macthing within all the test image 
 command: 
python 2d_3d_disk_all.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with disk feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

"""

class Image:
    def __init__(self, bitmap: ['C', 'H', 'W'], fname: str, orig_shape=None):
        self.bitmap     = bitmap
        self.fname      = fname
        if orig_shape is None:
            self.orig_shape = self.bitmap.shape[1:]
        else:
            self.orig_shape = orig_shape

    def resize_to(self, shape):
        return Image(
            self._pad(self._interpolate(self.bitmap, shape), shape),
            self.fname,
            orig_shape=self.bitmap.shape[1:],
        )

    @dimchecked
    def to_image_coord(self, xys: [2, 'N']) -> ([2, 'N'], ['N']):
        f, _size = self._compute_interpolation_size(self.bitmap.shape[1:])
        scaled = xys / f

        h, w = self.orig_shape
        x, y = scaled

        mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)

        return scaled, mask

    def _compute_interpolation_size(self, shape):
        x_factor = self.orig_shape[0] / shape[0]
        y_factor = self.orig_shape[1] / shape[1]

        f = 1 / max(x_factor, y_factor)

        if x_factor > y_factor:
            new_size = (shape[0], int(f * self.orig_shape[1]))
        else:
            new_size = (int(f * self.orig_shape[0]), shape[1])

        return f, new_size

    @dimchecked
    def _interpolate(self, image: ['C', 'H', 'W'], shape) -> ['C', 'h', 'w']:
        _f, size = self._compute_interpolation_size(shape)
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    
    @dimchecked
    def _pad(self, image: ['C', 'H', 'W'], shape) -> ['C', 'h', 'w']:
        x_pad = shape[0] - image.shape[1]
        y_pad = shape[1] - image.shape[2]

        if x_pad < 0 or y_pad < 0:
            raise ValueError("Attempting to pad by negative value")

        return F.pad(image, (0, y_pad, 0, x_pad))
    



def getKpDesc(model, img_tensor, img_name ):    
    image_query = Image(img_tensor, img_name)
    
    query_features = model.features(image_query.bitmap[None])

    kps_crop_space = query_features[0].kp.T
    kps_img_space, mask = image_query.to_image_coord(kps_crop_space)
    kps_img_space, mask = kps_img_space.cpu(), mask.cpu()
    
    keypoints   = kps_img_space.numpy().T[mask]
    descriptors = query_features[0].desc.detach().cpu().numpy()[mask]
    #scores      = query_features[0].kp_logp.detach().cpu().numpy()[mask]
    return keypoints, descriptors  


def localize_set(model_path, name, views, gaussians, pipeline, background, args):

        # Keep track of rotation and translation errors for calculation of the median error.
        rErrs = []
        tErrs = []

        prior_rErr = []
        prior_tErr = []
        pnp_p = []
        inliers = []


        gaussian_pcd = gaussians.get_xyz
        gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
        

        
        default_model_path = os.path.split(os.path.abspath(__file__))[0] + '/submodules/disk/depth-save.pth'
        state_dict = torch.load(default_model_path, map_location='cpu')
        DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        CPU   = torch.device('cpu')
        # compatibility with older model saves which used the 'extractor' name
        if 'extractor' in state_dict:
            weights = state_dict['extractor']
        elif 'disk' in state_dict:
            weights = state_dict['disk']
        else:
            raise KeyError('Incompatible weight file!')
        model = DISK(window=8, desc_dim=128)
        model.load_state_dict(weights)
        model = model.to(DEV)

    
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            #Get the image name and image itself
            query_name = view.image_name
            query_img = view.original_image[0:3, :, :]
            query_seq = view.seq_num
            #Get the image R and t and the reference K
            query_R, query_t = view.R, view.T
        

    
            # Get the query image key points and descriptor
            kp_query, desc_query = getKpDesc(model, query_img, query_name)
            desc_query = torch.tensor(desc_query).to("cuda")
            kp_query = torch.tensor(kp_query).to("cuda")
            

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
                    kp_query,
                    desc_query,
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
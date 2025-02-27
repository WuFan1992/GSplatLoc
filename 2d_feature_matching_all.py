import torch


import cv2
import numpy as np
import time
import torch
import torch.optim as optim

########## Image #############
from PIL import Image
from torchvision.transforms import PILToTensor
##############################


from utils.loss_utils import l1_loss

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
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import collections as mplcollections
from matplotlib import colors as mcolors

import torch.nn as nn



class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)


def pixel2ndc(pixel, S):
    return (((pixel/0.5)+1.0)/S)-1.0

def getOrigPoint(point_in_render_img, W, H, ProjMatrix):
    pixel_x, pixel_y, proj_z, p_w = point_in_render_img[1], point_in_render_img[0], point_in_render_img[2], point_in_render_img[3]
    p_proj_x, p_proj_y = pixel2ndc(pixel_x, W), pixel2ndc(pixel_y, H)
    p_hom_x, p_hom_y, p_hom_z = p_proj_x/p_w, p_proj_y/p_w, proj_z/p_w
    p_hom_w = 1/p_w
    p_hom = np.array([p_hom_x, p_hom_y, p_hom_z,p_hom_w])
    origP = np.matmul(p_hom, np.linalg.inv(ProjMatrix), dtype=np.float32)
    origP = origP[:3]
    print("ori = ", point_in_render_img[4], point_in_render_img[5],  point_in_render_img[6] )
    return origP

def getAllOrigPoints(points_in_render_img, W,H,ProjMatrix):
    match_3d = []
    for piri in points_in_render_img:
        origP = getOrigPoint(piri, W,H,ProjMatrix)
        match_3d.append(origP)
    return match_3d

def getXY(points):
    res = []
    for p in points:
        res.append([p[0],p[1]])
    return res
    
    

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    dst_points_xy = dst_points[:, [0,1]]
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
   # img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
   #                               matchColor=(0, 255, 0), flags=2)
    
    #clean the keypoint with mask
    ref_points_valid = []
    match_3d_points = []
    for i in range(len(mask)):
        if mask[i]:
            ref_points_valid.append(ref_points[i])
            match_3d_points.append(dst_points[i, [4,5,6]])

    return  ref_points_valid, match_3d_points




def getIntrinsic(view):
    K = np.eye(3)
    focal_length = fov2focal(view.FoVx, view.image_width)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2] = view.image_width / 2
    K[1, 2] = view.image_height / 2
    return K

def getRefImg(query_name):
    #Get the image number
    query_index = int(query_name.split("-")[1])
    if query_index + 15 > 1000:
        ref_index = query_index - 15
    else:
        ref_index = query_index + 15
    if ref_index > 99:
        ref_index = "000" + str(ref_index)
    else:
        ref_index = "0000"+ str(ref_index)
    ref_name = "frame-" + str(ref_index)
    return  ref_name

  


def localize_set(model_path, name, scene, gaussians, pipeline, background, args):

    views_test = scene.getTestCameras()
    views_train = scene.getTrainCameras()

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    prior_rErr = []
    prior_tErr = []
    pnp_p = []
    inliers = []


    gaussian_pcd = gaussians.get_xyz
    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
        
    xfeat = XFeat(top_k=4096)
    
    for _, view in enumerate(tqdm(views_test, desc="Rendering progress")):
        
        #Get the image name and image itself
        query_name = view.image_name
        query_img = view.original_image[0:3, :, :]
        query_seq = view.seq_num
        #Get the reference image name 
        ref_name = getRefImg(query_name)
        #load reference image
        ref_view = [view for view in views_train if view.image_name == ref_name and view.seq_num == query_seq]
        ref_img = ref_view[0].original_image[0:3, :, :]
        #Get the image R and t and the reference K
        query_R, query_t, K_query = view.R, view.T, getIntrinsic(ref_view[0])
        

    
        # Extract sparse features    
        # # [1,C,H,W] = [1,3,480,640]
        query_keypoints, _, query_feature = xfeat.detectAndCompute(query_img[None], 
                                                                 top_k=4096)[0].values()   #ref_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate
        
        render_pkg = render(ref_view[0], gaussians, pipeline, background)
    
        #-------------------------------------------------#
        #----- points_in_image size [4,Number of points]--#
        #----- feature_map size [C,H,W] = [64,480,640]----#
        #------points_in_render_image [7,N] --------------#
        #-------------------------------------------------#
        feature_map, points_in_render_image,  depth_map = render_pkg["feature_map"], render_pkg["points_in_render_images"], render_pkg["depth"] 
    

        # Get the length of all the projected points
        proj_p_number = (points_in_render_image.shape[1] - torch.sum(points_in_render_image[0].eq(-1))).item()
    
        #Copy the points_in_render_image to avoid the grandient descent
        points = points_in_render_image.clone().detach()
        
        #The colom is full zero if the 3D points project outside the image area
        non_zero_dim = torch.any(points != 0, dim=0)
        
        #Get the non zero indice and then remove all rhe colom 
        non_zero_indices = torch.nonzero(non_zero_dim)
        proj_p_xyzw = points[:,non_zero_indices.squeeze()]

        proj_xy = proj_p_xyzw[:2].transpose(0,1)
        interpolator = InterpolateSparse2d('bicubic')
        
        # Avoid to load all the keypoint coordinate at a time otherwise the CUDA will out of memory
        chunck_size = 10000
        chunck = proj_xy[0: chunck_size]
        proj_p_feature  = interpolator(feature_map[None], chunck[None], 480, 640).squeeze()
        for part in range(chunck_size,proj_xy.shape[0], chunck_size ):
            chunck = proj_xy[part: part + chunck_size]
            proj_p_feature_temp  = interpolator(feature_map[None], chunck[None], 480, 640).squeeze()
            proj_p_feature = torch.cat((proj_p_feature, proj_p_feature_temp), 0)

        proj_p_xyzw = proj_p_xyzw.T
    
        idxs0, idxs1 = xfeat.match(query_feature.to("cpu"), proj_p_feature.to("cpu"), min_cossim=0.82 )
        mkpts_0, mkpts_1 = query_keypoints[idxs0].cpu().numpy(), proj_p_xyzw[idxs1].cpu().numpy()

        #Transform the query and ref img to opencv format
        query_img = query_img.permute(1,2,0).cpu().numpy()
        ref_img = ref_img.permute(1,2,0).cpu().numpy()
        query_points_valid, match_3d = warp_corners_and_draw_matches(mkpts_0, mkpts_1, query_img, ref_img)
        
        num_match = len(match_3d)
   
        _, R, t, inl = cv2.solvePnPRansac(np.array(match_3d), np.array(query_points_valid), 
                                                      K_query, 
                                                      distCoeffs=None, 
                                                      flags=cv2.SOLVEPNP_ITERATIVE, 
                                                      iterationsCount=args.ransac_iters,
                                                      reprojectionError = 3.0
                                                      )
        R, _ = cv2.Rodrigues(R) 
    
        #print("R = ", R  , "  t= ", t)
        #print("query R = ", query_R, "query t = ", query_t)
        rotError, transError = calculate_pose_errors(query_R, query_t, R.T, t)

        # Print the errors
        print(f"Rotation Error: {rotError} deg")
        print(f"Translation Error: {transError} cm")
        print("query name = ", query_name,  " ref name = ", ref_view[0].image_name )
        print(f"Rotation Error moyen: {np.mean(prior_rErr)} deg")
        print(f"Translation Error moyen: {np.mean(prior_tErr)} cm")
        if inl is not None:
            prior_rErr.append(rotError)
            prior_tErr.append(transError)
            inliers.append(len(inl))
            pnp_p.append(num_match)
        print(f"Rotation Error moyen: {np.mean(prior_rErr)} deg")
        print(f"Translation Error moyen: {np.mean(prior_tErr)} cm")
        print(f"Mean Pnp points : {np.mean(pnp_p)}  ")
        print(f"Mean inliers : { np.mean(inliers) } cm ")
    
    err_mean_rot =  np.mean(prior_rErr)
    err_mean_trans = np.mean(prior_tErr)
    mean_pnp_p = np.mean(pnp_p)
    mean_inliers = np.mean(inliers) 
    print(f"Rotation Average Error: {err_mean_rot} deg ")
    print(f"Translation Average Error: {err_mean_trans} cm ")
    print(f"Mean Pnp points : {mean_pnp_p}  ")
    print(f"Mean inliers : {mean_inliers} cm ")
    

    

def launch_inference(dataset : ModelParams, pipeline : PipelineParams, args): 
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1]*64 if dataset.white_background else [0]*64
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    localize_set(dataset.model_path, "test", scene, gaussians, pipeline, background, args)


if __name__ == "__main__":
# Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--top_k", default=1_000, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    parser.add_argument("--warp_lr", default=0.0005, type=float)
    parser.add_argument("--warp_iters", default=251, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    args.eval = True

    launch_inference(model.extract(args), pipeline.extract(args), args)












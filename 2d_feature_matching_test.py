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
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)
    
    #clean the keypoint with mask
    ref_points_valid = []
    dst_points_valid = []
    for i in range(len(mask)):
        if mask[i]:
            ref_points_valid.append(ref_points[i])
            dst_points_valid.append(dst_points[i])

    return img_matches, ref_points_valid, dst_points_valid


def pixel_to_world(u, v, depth, K, R, t):
    """
    Converts pixel coordinates (u, v) to world coordinates (X, Y, Z).
    
    Args:
    - u, v: Pixel coordinates in image space.
    - depth: Depth (Z) of the pixel in the camera frame.
    - K: Camera intrinsic matrix (3x3 tensor).
    - R: Camera rotation matrix (3x3 tensor).
    - t: Camera translation vector (3x1 tensor).
    
    Returns:
    - X, Y, Z: World coordinates corresponding to the pixel (u, v).
    """
    # Step 1: Convert pixel (u, v) to normalized camera coordinates (x_c, y_c)
    uv_homogeneous = torch.tensor([u, v, 1.0], dtype=torch.double)  # Homogeneous coordinates
    uv_normalized = torch.matmul(torch.inverse(K), uv_homogeneous)
    
    x_c, y_c = uv_normalized[0], uv_normalized[1]  # Camera coordinates (normalized)

    # Step 2: Convert to world coordinates
    # Camera coordinates (x_c, y_c, Z) where Z is given by the depth
    camera_coordinates = torch.tensor([x_c * depth, y_c * depth, depth], dtype=torch.double)

    # Step 3: Apply the rotation and translation
    world_coordinates = torch.matmul(R, camera_coordinates) + t

    return world_coordinates



def getIntrinsic(view):
    K = np.eye(3)
    focal_length = fov2focal(view.FoVx, view.image_width)
    K[0, 0] = K[1, 1] = focal_length
    K[0, 2] = view.image_width / 2
    K[1, 2] = view.image_height / 2
    return K


def getWorldCoordinates(list_pixels, list_depth, K, R, t):
    P_N = len(list_pixels)
    output = torch.zeros(P_N, 3)
    list_depth = list_depth.detach().numpy()
    for index in range(len(list_pixels)):
        X,Y,Z = pixel_to_world(list_pixels[index][0], list_pixels[index][1], list_depth[index], K, R, t)
        output[index] = torch.tensor([X,Y,Z])
    return output
        
        

        
    

    


def localize_set(model_path, name, views, gaussians, pipeline, background, args):

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    prior_rErr = []
    prior_tErr = []


    gaussian_pcd = gaussians.get_xyz
    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
        
    xfeat = XFeat(top_k=100)

    #Load image
    img_dir = "./datasets/images/"
    query_img_name = "frame-000005.color.png"
    query_img_name_noext = "frame-000005"
    ref_img_name_noext = "frame-000005"
    ref_img_name = "frame-000005.color.png"

    query_img_path = os.path.join(img_dir, query_img_name)
    query_img = cv2.imread(query_img_path) # [H,W,C] = [480,640,3]

    
    #==========================
    
    ref_img_path = os.path.join(img_dir, ref_img_name)
    ref_img = cv2.imread(ref_img_path)
    tensor_ref_img = xfeat.parse_input(ref_img) # [1,C,H,W] = [1,3,480,640]
    ref_keypoints, _, ref_feature = xfeat.detectAndCompute(tensor_ref_img, 
                                                                 top_k=4096)[0].values()  
    #====================================
    # Get the reference img pose

    
    # Extract sparse features
    tensor_query_img = xfeat.parse_input(query_img) # [1,C,H,W] = [1,3,480,640]
    query_keypoints, _, query_feature = xfeat.detectAndCompute(tensor_query_img, 
                                                                 top_k=4096)[0].values()  #query_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate
    # Get the reference img pose
    ref = [ view for view in views if view.image_name == ref_img_name_noext]
    que = [ view for view in views if view.image_name == query_img_name_noext]
    # Get the reference R and t
    K_ref, K_query = getIntrinsic(ref[0]), getIntrinsic(que[0])


    
    render_pkg = render(ref[0], gaussians, pipeline, background)
    
    #-------------------------------------------------#
    #----- points_in_image size [4,Number of points]--#
    #----- feature_map size [C,H,W] = [64,480,640]----#
    #------points_in_render_image [7,N] --------------#
    #-------------------------------------------------#
    feature_map, points_in_render_image,  depth_map = render_pkg["feature_map"], render_pkg["points_in_render_images"], render_pkg["depth"] 
    


    # Get the proj pixel 
    points_x, points_y, points_z, pw = points_in_render_image[0].tolist(), points_in_render_image[1].tolist(), points_in_render_image[2].tolist(), points_in_render_image[3].tolist()
    X,Y,Z = points_in_render_image[4].tolist(), points_in_render_image[5].tolist(), points_in_render_image[6].tolist()
    points_xyzw = [(x, y, z, w, xx,yy,zz) for x,y,z,w,xx,yy,zz in zip(points_x, points_y, points_z, pw, X,Y,Z)]
    # Get the length of all the projected points
    proj_p_number = (points_in_render_image.shape[1] - torch.sum(points_in_render_image[0].eq(-1))).item()

    #proj_p_feature = torch.zeros(proj_p_number, 64)
    index = 0
    
    points = points_in_render_image.clone().detach()
    non_zero_dim = torch.any(points != 0, dim=0)
    non_zero_indices = torch.nonzero(non_zero_dim)
    proj_p_xyzw = points[:,non_zero_indices.squeeze()]

    proj_xy = proj_p_xyzw[:2].transpose(0,1)
    interpolator = InterpolateSparse2d('bicubic')
    chunck_size = 10000
    chunck = proj_xy[0: chunck_size]
    proj_p_feature  = interpolator(feature_map[None], chunck[None], 480, 640).squeeze()
    for part in range(chunck_size,proj_xy.shape[0], chunck_size ):
        chunck = proj_xy[part: part + chunck_size]
        proj_p_feature_temp  = interpolator(feature_map[None], chunck[None], 480, 640).squeeze()
        proj_p_feature = torch.cat((proj_p_feature, proj_p_feature_temp), 0)

    proj_p_xyzw = proj_p_xyzw.T
    
    """
    for xyzw in points_xyzw: 
        if index > 1499:
            break
        if xyzw[0] !=-1 and xyzw[1] != -1 and xyzw[0] < 640 and xyzw[1] < 480 and xyzw[0]>0 and xyzw[1]>0:
            proj_p_feature[index] = feature_map[:,int(xyzw[1]), int(xyzw[0])]
            proj_p_xyzw[index, 0], proj_p_xyzw[index, 1], proj_p_xyzw[index, 2], proj_p_xyzw[index, 3] = xyzw[0], xyzw[1], xyzw[2], xyzw[3]
            proj_p_xyzw[index, 4], proj_p_xyzw[index, 5], proj_p_xyzw[index, 6] = xyzw[4], xyzw[5], xyzw[6]
            index = index + 1    
    
    """
    idxs0, idxs1 = xfeat.match(query_feature.to("cpu"), proj_p_feature.to("cpu"), min_cossim=0.82 )
    mkpts_0, mkpts_1 = query_keypoints[idxs0].cpu().numpy(), proj_p_xyzw[idxs1].cpu().numpy()
    canvas, query_points_valid, proj_points_valid = warp_corners_and_draw_matches(mkpts_0, mkpts_1, query_img, ref_img)
    
 

    
    match_3d = []
    for hello in proj_points_valid:
        match_3d.append([hello[4], hello[5], hello[6]])
    print("match_3d = ", match_3d)

    
    _, R, t, _ = cv2.solvePnPRansac(np.array(match_3d), np.array(query_points_valid), 
                                                  K_query, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=args.ransac_iters
                                                  )
    R, _ = cv2.Rodrigues(R) 
    
    print("R = ", R  , "  t= ", t)
    print("query R = ", que[0].R, "query t = ", que[0].T)
    rotError, transError = calculate_pose_errors(que[0].R, que[0].T, R.T, t)

    # Print the errors
    print(f"Rotation Error: {rotError} deg")
    print(f"Translation Error: {transError} cm")
    
    plt.figure(figsize=(12,12))
    plt.imshow(canvas[..., ::-1]), plt.show()
    
    

def launch_inference(dataset : ModelParams, pipeline : PipelineParams, args): 
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1]*64 if dataset.white_background else [0]*64
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    localize_set(dataset.model_path, "test", scene.getTrainCameras(), gaussians, pipeline, background, args)


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

    launch_inference(model.extract(args), pipeline.extract(args), args)












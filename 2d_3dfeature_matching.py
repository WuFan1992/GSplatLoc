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
        res.append([p[1],p[0]])
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
    keypoints_matched = keypoints[..., :2].cpu().numpy().astype(np.float64)
    
    return point_vis, keypoints_matched

        

def localize_set(model_path, name, scenes, gaussians, pipeline, background, args):

    test_views = scenes.getTestCameras()

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    prior_rErr = []
    prior_tErr = []


    gaussian_pcd = gaussians.get_xyz
    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
        
    xfeat = XFeat(top_k=10)

    #Load image
    img_dir = "./datasets/wholehead/images/seq-01"
    query_img_name = "frame-000005.color.png"
    query_img_name_noext = "frame-000005"
    

    query_img_path = os.path.join(img_dir, query_img_name)
    query_img = cv2.imread(query_img_path) # [H,W,C] = [480,640,3]

    # Extract sparse features
    tensor_query_img = xfeat.parse_input(query_img) # [1,C,H,W] = [1,3,480,640]
    query_keypoints, _, query_feature = xfeat.detectAndCompute(tensor_query_img, 
                                                                 top_k=10)[0].values()  #query_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate

    # Get the reference img pose
    que = [ view for view in test_views if view.image_name == query_img_name_noext and view.seq_num == "seq-01"]
    # Get the reference R and t
    K_query = getIntrinsic(que[0])

     # Find initial pose prior via 2D-3D matching
    with torch.no_grad():
        matched_3d, matched_2d = find_2d3d_correspondences(
                    query_keypoints,
                    query_feature,
                    gaussian_pcd,
                    gaussian_feat
                )
        
    print("match 2d = ", matched_2d)
    print("match 3d = ", matched_3d)
    
    
    ##########################
    pcd_idx = torch.where(torch.all(torch.tensor(matched_3d[5]).to("cuda") == gaussian_pcd, dim=1))[0].item()
    p_feature = gaussian_feat[pcd_idx].to("cpu")
    
    points_idx = torch.where(torch.all(torch.tensor(matched_2d[5]).to("cuda") == query_keypoints, dim=1))[0].item()
    q_feature = query_feature[points_idx].to("cpu")
    
    output = torch.cosine_similarity(q_feature, p_feature, dim=0)
    print(output)
    ############################

    gt_R = que[0].R
    gt_t = que[0].T

    _, R, t, _ = cv2.solvePnPRansac(matched_3d, matched_2d, 
                                                  K_query, 
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

    launch_inference(model.extract(args), pipeline.extract(args), args)












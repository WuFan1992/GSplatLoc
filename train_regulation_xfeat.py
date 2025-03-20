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

# Regulation package
from reguler.helper.utils import *

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



""""
This file is used to run the whole test cases and get the average error for xfeat feature
It first gets the test image from the Scene. Apply the Xfeat detector to get the keypoints and descriptors
The test image will then sent to NetVlad to get its global descriptor. This global descriptor will then be
used to find its most similar image in training dataset (each image in training dataset are considered as 
reference image)
Once query image and reference image are found, we get the extrinsic and intrinsic parameters of reference image and then 
Use these two paramters to project(rasterize) the 3DGS-Xfeat. We record each pixel in the projected feature map that correspond with a 
3DGS point. Theses pixel with its feature are then used to match with the keypoint of query image.
Once the matching is done, for each matching points in  projected image, we have its 3D coordinate so that we can directly apply PnP RANSAC 
to 2D(query)-3D (3DGS)        
The methods return the average of all the test image's pose error 

command: 
python 2d_feature_xfeat_all.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

If we want to use the netvlad to do the image retrieval, we must launch the getdes.py. Make sure that in the netvlad.py, 
from netvlad.base_model import BaseModel must be 
from base_model import BaseModel

python getdes.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

Then after get the global descriptor, change the 
from base_model import BaseModel
back to  
from netvlad.base_model import BaseModel 
before runing the 2d_feature_disk_all.py
"""


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import collections as mplcollections
from matplotlib import colors as mcolors

import torch.nn as nn

from reguler.helper.utils import *
from reguler.network import *



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
    
def calculate_3d_coordinates(intrinsic_matrix, extrinsic_matrix, depth_map, pixel_coord):
    """
    Calculate the 3D coordinates from the pixel coordinate, depth map, intrinsic and extrinsic matrices.
    
    Args:
    - intrinsic_matrix (torch.Tensor): 3x3 camera intrinsic matrix.
    - extrinsic_matrix (torch.Tensor): 4x4 camera extrinsic matrix.
    - depth_map (torch.Tensor): A 2D depth map with depth values for each pixel.
    - pixel_coord (tuple): (u, v) pixel coordinate for which we want the 3D coordinate.
    
    Returns:
    - world_coord (torch.Tensor): 3D world coordinates of the point (x, y, z).
    """
    
    # Step 1: Invert the intrinsic matrix
    intrinsic_inv = torch.inverse(intrinsic_matrix)
    
    # Step 2: Get the pixel coordinates (u, v)
    u, v = pixel_coord
    
    # Step 3: Get the depth from the depth map at the pixel coordinate
    depth = depth_map[int(v)][int(u)]
    
    # Step 4: Convert 2D pixel to normalized camera coordinates (x_cam, y_cam, z_cam)
    # Apply inverse intrinsic matrix to the pixel coordinate (u, v)
    pixel_homogeneous = torch.tensor([u, v, 1], dtype=torch.double).to("cuda")
    normalized_cam_coords = torch.matmul(intrinsic_inv, pixel_homogeneous)
    
    # Step 5: Scale the normalized camera coordinates by depth to get the 3D camera coordinates
    x_cam = normalized_cam_coords[0] * depth
    y_cam = normalized_cam_coords[1] * depth
    z_cam = depth
    
    # Step 6: Convert to world coordinates using the extrinsic matrix
    # Extrinsic matrix defines the transformation from camera coordinates to world coordinates
    camera_coords = torch.tensor([x_cam, y_cam, z_cam, 1], dtype=torch.float32).to("cuda")  # Homogeneous coordinates
    
    # Apply the extrinsic matrix to get world coordinates
    world_coords = torch.matmul(extrinsic_matrix, camera_coords)
    
    return world_coords[:3]  # Return (x, y, z) world coordi


def find_nearest_point(coord, points):
    # Convert the input coordinate to a tensor (if it isn't already)
    coord_tensor = torch.tensor(coord, dtype=torch.float32)
    
    # Calculate the Euclidean distance between the given coordinate and each point in the tensor
    distances = torch.norm(points - coord_tensor.view(2, 1), dim=0)
    
    # Find the index of the minimum distance
    nearest_index = torch.argmin(distances)
    
    # Return the nearest point
    nearest_point = points[:, nearest_index]
    
    return nearest_point

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

def get_match_gt(all_2d, matched_2d, all_feature, all_3d, device="cuda"):
    
    diff = all_2d.unsqueeze(0).to(device) - matched_2d.unsqueeze(1).to(device)  # Reshape for broadcasting
    # Check where the difference is zero (i.e., exact match)
    match_mask = torch.all(diff == 0, dim=2)  # Check equality along the last dimension (x and y)
    # Get the indices of the matches
    matching_indices_list = match_mask.nonzero(as_tuple=True)[1].to("cpu").numpy() if device=="cuda" else match_mask.nonzero(as_tuple=True)[1].numpy()
    return all_3d[matching_indices_list], all_feature[matching_indices_list]

"""
def get_match_gt_features(all_2d, matched_2d, all_feature, device="cuda"):

    matching_indices = torch.isin(all_2d.to(device), matched_2d.to(device))
    matching_indices_list = torch.nonzero(matching_indices.all(dim=1), as_tuple=False).squeeze().tolist()
    return all_feature[matching_indices_list]
"""
def get_match_3d_ranges(matched_2d, xy_to_3d_ranges, device="cuda"):

    xys = xy_to_3d_ranges.t()[:,:2]
    ranges = xy_to_3d_ranges.t()[:,[2,3]]
    
    diff = xys.unsqueeze(0).to(device) - matched_2d.unsqueeze(1).to(device)  # Reshape for broadcasting
    # Check where the difference is zero (i.e., exact match)
    match_mask = torch.all(diff == 0, dim=2)  # Check equality along the last dimension (x and y)
    # Get the indices of the matches
    matching_indices_list = match_mask.nonzero(as_tuple=True)[1].to("cpu").numpy() if device=="cuda" else match_mask.nonzero(as_tuple=True)[1].numpy()
    return ranges[matching_indices_list]



def diff_tensor(network_output, gt, device="cuda"):
    diff = network_output.to(device) - gt.to(device)
    return torch.nn.functional.normalize(diff, dim=1)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


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
    
    config = Config()
    refiner = Refiner(config)
    optimizer = torch.optim.SGD(refiner.parameters(), lr=0.001, momentum=0.9)

        
    xfeat = XFeat(top_k=10)
        
    for _, view in enumerate(tqdm(views_train, desc="Rendering progress")):
        
        #Get the image name and image itself

        query_img = view.original_image[0:3, :, :]
        #Get the image R and t and the reference K
        query_R, query_t = view.R, view.T
        query_K = getIntrinsic(view)
        
        # Extract sparse features    
        # # [1,C,H,W] = [1,3,480,640]
        query_keypoints, _, query_feature = xfeat.detectAndCompute(query_img[None], 
                                                                 top_k=10)[0].values()   #ref_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate
        
        render_pkg = render(view, gaussians, pipeline, background)
    
        #-------------------------------------------------#
        #----- points_in_image size [4,Number of points]--#
        #----- feature_map size [C,H,W] = [64,480,640]----#
        #------points_in_render_image [7,N] --------------#
        #-------------------------------------------------#
        feature_map, points_in_render_image,  depth_map = render_pkg["feature_map"], render_pkg["points_in_render_images"], render_pkg["depth"] 
        xy_to_3d_ranges = render_pkg["xy_to_3D_ranges"].detach().to("cpu")
        np.savetxt("hello.txt", xy_to_3d_ranges.t().cpu().numpy())
                
        query_keypoints_3d = [calculate_3d_coordinates(torch.tensor(query_K).to("cuda"), view.world_view_transform, depth_map.squeeze().detach(), kp) for kp in query_keypoints]
        query_keypoints_3d = torch.stack(query_keypoints_3d, dim=0)
        with torch.no_grad():
            matched_2d, matched_3d, match_3d_feature = find_2d3d_correspondences(
                    query_keypoints,
                    query_feature,
                    gaussian_pcd,
                    gaussian_feat
                )
        #matched_2d, matched_3d, match_3d_feature = matched_2d, matched_3d.numpy(), match_3d_feature.numpy()

        matched_gt_3d, matched_gt_feature = get_match_gt(query_keypoints, torch.tensor(matched_2d), query_feature,  query_keypoints_3d)
        gt_diff_3d = matched_gt_3d - torch.tensor(matched_3d).to("cuda")
        
        diff_feature = diff_tensor(matched_gt_feature, torch.tensor(match_3d_feature))  # input 1 feature distance
        
        # Get the range (min point index to max point index) with given 2D point
        ranges = get_match_3d_ranges(torch.tensor(matched_2d), xy_to_3d_ranges)

        # Get a list of points correspondant with ranges
        points = get_whole_points_from_ranges(ranges, gaussian_pcd)
 
        
        #Calculate the mass center of each range (correspondant to a series of points that raster a pixel)
        mass_centers = get_whole_mass_center(points)
        mass_centers = torch.stack(mass_centers, dim=0)

        dist = torch.tensor(matched_3d).to("cuda")- mass_centers
        #shift = torch.linalg.norm(dist, dim=1, ord=2)

        #calculate the mass density of each range
         
        mass_densities = get_whole_mass_density(query_keypoints_3d, points)

        #Normalization density
        mass_densities = torch.stack(mass_densities, dim=0)
        mass_densities= normalize_density(mass_densities)
        
        optimizer.zero_grad()
        
        pred_shift = refiner(diff_feature.cpu(), dist.cpu().to(torch.float32))
        loss = l1_loss(pred_shift, gt_diff_3d.cpu())
        loss.backward()
        optimizer.step()
        

        
        
        ############ Show image #################
        print("matched 2d [1] = ", matched_2d[0])
        torch.save(points[0].cpu(),"points.pt")
        query_img = query_img.permute(1,2,0).cpu().numpy()
        imgplot = plt.imshow(query_img)
        plt.show()
        
        
        
    
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
            # Very rare case: the proj_p_feature_temps have only one feature with dim=64 so is [[64]] instead of [N,64]
            if proj_p_feature_temp.dim()==1:
                proj_p_feature_temp = proj_p_feature_temp[None]
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

        if inl is not None:
            prior_rErr.append(rotError)
            prior_tErr.append(transError)
            inliers.append(len(inl))
            pnp_p.append(num_match)

    
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












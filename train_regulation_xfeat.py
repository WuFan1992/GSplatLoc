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
from reguler.helper.transformer import *

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
from random import randint

# For the log 
import uuid
from argparse import Namespace
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



"""
python train_regulation_xfeat.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

If we train with disk, we need to point out in the dataset_reader.py where to find the pre-extract disk feature

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
def get_match_mass_center_density(matched_2d, xy_mass_center, device="cuda"):

    xys = xy_mass_center.t()[:,:2]
    massy_center = xy_mass_center.t()[:,[2,3,4]]
    density = xy_mass_center.t()[:,5]
    
    diff = xys.unsqueeze(0).to(device) - matched_2d.unsqueeze(1).to(device)  # Reshape for broadcasting
    # Check where the difference is zero (i.e., exact match)
    match_mask = torch.all(diff == 0, dim=2)  # Check equality along the last dimension (x and y)
    # Get the indices of the matches
    matching_indices_list = match_mask.nonzero(as_tuple=True)[1].to("cpu").numpy() if device=="cuda" else match_mask.nonzero(as_tuple=True)[1].numpy()
    return massy_center[matching_indices_list], density[matching_indices_list]



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

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def localize_set(model_path, name, scene, gaussians, pipeline, background, args):
   
    tb_writer = prepare_output_and_logger(args)
    # constant iteration 
    total_iter = 3000
    
    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    prior_rErr = []
    prior_tErr = []
    pnp_p = []
    inliers = []
    
    gaussian_pcd = gaussians.get_xyz
    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
    
    lftr = LocalFeatureTransformer()
    config = Config()
    refiner = Refiner(config)
    optimizer = torch.optim.SGD(refiner.parameters(), lr=1.e-6)

        
    xfeat = XFeat(top_k=4096)
    
    
    # For the progress bar 
    first_iter = 0
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, total_iter), desc="Training progress")
    first_iter += 1
        
    #for _, view in enumerate(tqdm(views_train, desc="Rendering progress")):
    for iteration in range(first_iter, total_iter):
           
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        query_img = viewpoint_cam.original_image[0:3, :, :]
        query_img_name = viewpoint_cam.image_name
        #Get the training camera view intrinsic
        query_K = getIntrinsic(viewpoint_cam)
        
        # Extract sparse features    
        # # [1,C,H,W] = [1,3,480,640]
        query_keypoints, _, query_feature = xfeat.detectAndCompute(query_img[None], 
                                                                 top_k=4096)[0].values()   #ref_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate
       
        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
    
        #-------------------------------------------------#
        #----- points_in_image size [4,Number of points]--#
        #----- feature_map size [C,H,W] = [64,480,640]----#
        #------points_in_render_image [7,N] --------------#
        #-------------------------------------------------#
        depth_map = render_pkg["depth"] 
        xy_mass_center = render_pkg["xy_to_3D_ranges"].detach().to("cpu")

                        
        query_keypoints_3d = [calculate_3d_coordinates(torch.tensor(query_K).to("cuda"), viewpoint_cam.world_view_transform, depth_map.squeeze().detach(), kp) for kp in query_keypoints]
        query_keypoints_3d = torch.stack(query_keypoints_3d, dim=0)
        with torch.no_grad():
            
            query_feature = torch.squeeze(lftr(query_feature[None].cpu())).to("cuda")
            query_feature = torch.nn.functional.normalize(query_feature,dim=0)

            matched_2d, matched_3d, match_3d_feature = find_2d3d_correspondences(
                    query_keypoints,
                    query_feature,
                    gaussian_pcd,
                    gaussian_feat
                )
        #matched_2d, matched_3d, match_3d_feature = matched_2d, matched_3d.numpy(), match_3d_feature.numpy()

        matched_gt_3d, matched_gt_feature = get_match_gt(query_keypoints, torch.tensor(matched_2d), query_feature,  query_keypoints_3d)
        gt_diff_3d = matched_gt_3d - torch.tensor(matched_3d).to("cuda")
        
        #diff_feature = diff_tensor(matched_gt_feature, torch.tensor(match_3d_feature))  # input 1 feature distance
        
        # Get the mass center with given 2D point
        mass_centers,density = get_match_mass_center_density(torch.tensor(matched_2d), xy_mass_center)

        mass_center_density = torch.cat([mass_centers, torch.transpose(density[None], 0,1)], dim=1)


        #dist = torch.tensor(matched_3d)- mass_centers

        #shift = torch.linalg.norm(dist, dim=1, ord=2)

        #calculate the mass density of each range
         
        #mass_densities = get_whole_mass_density(query_keypoints_3d, points)

        #Normalization density
        #mass_densities = torch.stack(mass_densities, dim=0)
        #mass_densities= normalize_density(mass_densities)
        
        optimizer.zero_grad()

        pred_shift, gen_feature = refiner(torch.tensor(match_3d_feature), torch.tensor(mass_center_density).to(torch.float32))

        loss = 0.6*l1_loss(pred_shift, gt_diff_3d.cpu()) + 0.4*l1_loss(gen_feature, matched_gt_feature.cpu())
        tb_writer.add_scalar("Loss/train", loss, iteration)
        print("loss = ", loss)
        loss.backward()
        optimizer.step()
        
        
        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == total_iter:
                progress_bar.close()

            # Log and save 
            if (iteration == total_iter):
                print("\n[ITER {}] Saving Regulation".format(iteration))
                torch.save(refiner.state_dict(), scene.model_path + "/regulation_" + str(iteration) + ".pth")
  
        
        tb_writer.flush()
        
        
        ############ Show image #################
        """
        print("matched 2d [1] = ", matched_2d[0])
        torch.save(points[0].cpu(),"points.pt")
        query_img = query_img.permute(1,2,0).cpu().numpy()
        imgplot = plt.imshow(query_img)
        plt.show()
        """
        
    

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












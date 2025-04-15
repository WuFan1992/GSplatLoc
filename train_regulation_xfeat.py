import torch


import cv2
import numpy as np
import torch


# Regulation package


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


import torch.nn as nn


from diffestimator.model import PosExtractNet, Config



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

def mse_loss(network_output, gt):
    return torch.mean((network_output-gt)**2)


"""
Use to get the gt 3d coord from query keypoint
"""
def new_calculate_ndc2camera(proj_matrix, xndc, yndc, depth):
    a1 = proj_matrix[0,0]
    a2 = proj_matrix[0,1]
    a3 = proj_matrix[0,2]
    a4 = proj_matrix[0,3]
    
    a5 = proj_matrix[1,0]
    a6 = proj_matrix[1,1]
    a7 = proj_matrix[1,2]
    a8 = proj_matrix[1,3]
    
    
    a13 = proj_matrix[3,0]
    a14 = proj_matrix[3,1]
    a15 = proj_matrix[3,2]
    a16 = proj_matrix[3,3]
    
    A1 = a1-xndc*a13
    B1 = a2-xndc*a14
    C1 = (a3-xndc*a15)*depth+a4-xndc*a16
    
    A2 = a5-yndc*a13
    B2 = a6-yndc*a14
    C2 = (a7-yndc*a15)*depth+a8-yndc*a16
    
    X = (-C1*B2+C2*B1)/(A1*B2-A2*B1)
    Y = (-A1*C2+A2*C1)/(A1*B2-A2*B1)
    
    return X, Y

def pixel2ndc(pixel, S):
    return (((pixel/0.5)+1.0)/S)-1.0


def getGTXYZ(camera2ndc, view2camera, point_2d, depth_map):
    
    #Get the depth value
    depth_map = depth_map.detach().squeeze(0)
    depth = depth_map[point_2d[:,1].int().to("cpu"), point_2d[:,0].int().to("cpu")] 
    X, Y = new_calculate_ndc2camera(camera2ndc.transpose(0,1), pixel2ndc(point_2d[:,0], 640), pixel2ndc(point_2d[:,1], 480), depth)
    ones = torch.tensor([1.0]).repeat(point_2d.size(0)).to("cuda")
        
    cam_coord_inv = torch.stack([X, Y, depth, ones], dim=1)
    output = torch.matmul(cam_coord_inv.double(), torch.inverse(view2camera).double())
    return output[:, :3]




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
    total_iter = 5000
        
    gaussian_pcd = gaussians.get_xyz
    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
    
    config = Config()
    posenet = PosExtractNet(config)
    optimizer = torch.optim.SGD(posenet.parameters(), lr=1.e-6)

        
    xfeat = XFeat(top_k=4096)
    saving_itr = np.arange(100,total_iter+100,100)
    
    # For the progress bar 
    first_iter = 0
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, total_iter), desc="Training progress")
    first_iter += 1
        
    #for _, view in enumerate(tqdm(views_train, desc="Rendering progress")):
    for iteration in range(first_iter, total_iter+1):
        
         # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        #get batch viewpoint_cam batch size = 16:
        batch_viewpoint_cam = []
        for _ in range(16):
            if len(viewpoint_stack) == 0:
                break
            batch_viewpoint_cam.append(viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)))
            
        loss = 0   
        for viewpoint_cam in batch_viewpoint_cam:
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
        
            #For each keypoint detected in query image, find its coordinate in 3DGS
            query_keypoints_3d = getGTXYZ(viewpoint_cam.projection_matrix, viewpoint_cam.world_view_transform, query_keypoints, depth_map)

            
            with torch.no_grad():
                matched_2d, _, match_3d_feature = find_2d3d_correspondences(
                        query_keypoints,
                        query_feature,
                        gaussian_pcd,
                        gaussian_feat
                )
              
            _, matched_gt_feature = get_match_gt(query_keypoints, torch.tensor(matched_2d), query_feature,  query_keypoints_3d)
            match_3d_feature = torch.tensor(match_3d_feature).to("cuda")
            pred_R, pred_t = posenet(match_3d_feature.to("cpu")[None], matched_gt_feature.to("cpu")[None])
            gt_R = torch.tensor(viewpoint_cam.R)
            gt_t = torch.tensor(viewpoint_cam.T)
            loss += mse_loss(gt_R, pred_R) + mse_loss(gt_t, pred_t)
            
        optimizer.zero_grad()
        loss = loss /(len(batch_viewpoint_cam))
        tb_writer.add_scalar("Loss/train_regulation", loss, iteration)
        print("batch loss = ", loss)
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
            if (iteration in saving_itr):
                print("\n[ITER {}] Saving Regulation".format(iteration))
                torch.save(posenet.state_dict(), scene.model_path + "/regulation_" + str(iteration) + ".pth")
  
        
        tb_writer.flush()
                
    

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












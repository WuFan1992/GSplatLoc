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

from warping.warping_loss import *
from warping.warp_utils import *
from utils.loc_utils import *
import torch.nn.functional as F

# // For the netvlad global descriptor
from netvlad.netvlad import NetVLAD

###############################
# python 2d_feature_matching_all.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000  
################################

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import collections as mplcollections
from matplotlib import colors as mcolors

import torch.nn as nn

from torch_dimcheck import dimchecked

from submodules.disk.disk.model.disk import DISK
from submodules.disk.disk.common.structs import Features

""""
This file is used to run the whole test cases and get the average error for disk feature
It first gets the test image from the Scene. Apply the Disk detector to get the keypoints and descriptors
The test image will then sent to NetVlad to get its global descriptor. This global descriptor will then be
used to find its most similar image in training dataset (each image in training dataset are considered as 
reference image)
Once query image and reference image are found, we get the extrinsic and intrinsic parameters of reference image and then 
Use these two paramters to project(rasterize) the 3DGS-Disk. We record each pixel in the projected feature map that correspond with a 
3DGS point. Theses pixel with its feature are then used to match with the keypoint of query image.
Once the matching is done, for each matching points in  projected image, we have its 3D coordinate so that we can directly apply PnP RANSAC 
to 2D(query)-3D (3DGS)        
The methods return the average of all the test image's pose error 

command: 
python 2d_feature_disk_all.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with disk feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
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


    

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    dst_points_xy = dst_points[:, [0,1]]
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    #warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    """
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners
    """
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



def createNetVlad():
    conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NetVLAD(conf).eval().to(device)
    
    return model

def imageRetrieval(query_img, netvlad_model,global_desc_names):
    
    global_desc, names = torch.squeeze(global_desc_names[0]), global_desc_names[1]
    query_global_desc = netvlad_model(query_img[None])["global_descriptor"]
    
    similarity = torch.mm(query_global_desc, global_desc.t().cuda())
    _, idx = similarity.max(dim=1)
    
    num_seq = names[idx].split("/")[0]
    img_name = names[idx].split("/")[1]
    
    return img_name, num_seq
    
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

def disk_match(feats1, feats2, min_cossim = 0.82):
    
    cossim = feats1 @ feats2.t()
    cossim_t = feats2 @ feats1.t()	
    _, match12 = cossim.max(dim=1)
    _, match21 = cossim_t.max(dim=1)
    idx0 = torch.arange(len(match12), device=match12.device)
    mutual = match21[match12] == idx0
    
    
    if min_cossim > 0:
        cossim, _ = cossim.max(dim=1)
        good = cossim > min_cossim
        idx0 = idx0[mutual & good]
        idx1 = match12[mutual & good]
    else:
        idx0 = idx0[mutual]
        idx1 = match12[mutual]
        
    return idx0, idx1

  


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
    
    #Load the global descriptor
    global_desc_names = torch.load("./netvlad/global_desc.pt")
    netvlad_model = createNetVlad()
    
    
        

        
    for _, view in enumerate(tqdm(views_test, desc="Rendering progress")):
        
        #Get the image name and image itself
        query_name = view.image_name
        query_img = view.original_image[0:3, :, :]
        query_seq = view.seq_num
        #Get the reference image name 
        ref_name, ref_seq = imageRetrieval(query_img, netvlad_model,global_desc_names)
        #load reference image
        ref_view = [view for view in views_train if view.image_name == ref_name and view.seq_num == ref_seq]
        ref_img = ref_view[0].original_image[0:3, :, :]
        #Get the image R and t and the reference K
        query_R, query_t, K_query = view.R, view.T, getIntrinsic(ref_view[0])
        

    
        # Get the query image key points and descriptor
        kp_query, desc_query = getKpDesc(model, query_img, query_name)
        
        
        render_pkg = render(ref_view[0], gaussians, pipeline, background)
    
        #-------------------------------------------------#
        #----- points_in_image size [4,Number of points]--#
        #----- feature_map size [C,H,W] = [64,480,640]----#
        #------points_in_render_image [7,N] --------------#
        #-------------------------------------------------#
        feature_map, points_in_render_image,  depth_map = render_pkg["feature_map"], render_pkg["points_in_render_images"], render_pkg["depth"] 
        feature_map = torch.nn.functional.normalize(feature_map,dim=0)

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
    
        idxs0, idxs1 = disk_match(torch.tensor(desc_query), proj_p_feature.to("cpu"), min_cossim=0.82 )
        mkpts_0, mkpts_1 = kp_query[idxs0], proj_p_xyzw[idxs1].cpu().numpy()
        
        if len(mkpts_0) < 7:
            continue
        #Transform the query and ref img to opencv format
        query_img = query_img.permute(1,2,0).cpu().numpy()
        ref_img = ref_img.permute(1,2,0).cpu().numpy()

        query_points_valid, match_3d = warp_corners_and_draw_matches(mkpts_0, mkpts_1, query_img, ref_img)
        
        
        num_match = len(match_3d)
        if num_match < 1:
            continue

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












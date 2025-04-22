import torch


import cv2
import numpy as np
import torch


from utils.loss_utils import l1_loss


from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from encoders.XFeat.modules.xfeat import XFeat

from warping.warping_loss import *
from warping.warp_utils import *
from utils.loc_utils import *
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import matplotlib.pyplot as plt

import torch.nn as nn



""""
This file is used to show the similarity between the query keypoint and 3DGS feature directly without loading the Scene (save time )

command: 
python show_kp_3dgs_similarity.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

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
before runing the 2d_feature_disk_one.py


"""


def find_2d3d_matching(keypoints, image_features, gaussian_pcd, gaussian_feat, chunk_size=10000):
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


    
def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2, ref_feat, dst_feat):
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
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)
    
    #clean the keypoint with mask
    ref_points_valid = []
    dst_points_valid = []
    ref_feat_valid = []
    dst_feat_valid = []
    for i in range(len(mask)):
        if mask[i]:
        #if True:
            ref_points_valid.append(ref_points[i])
            dst_points_valid.append(dst_points[i])
            ref_feat_valid.append(ref_feat[i])
            dst_feat_valid.append(dst_feat[i])

    return img_matches, ref_points_valid, dst_points_valid, ref_feat_valid, dst_feat_valid

def find_row_index(tensor_2d, target_row):
    matches = torch.all(tensor_2d == target_row, dim=1)
    indices = torch.nonzero(matches, as_tuple=True)[0]
    return indices.item() if indices.numel() > 0 else -1



def localize_set(gaussians,args):


    # Keep track of rotation and translation errors for calculation of the median error.
    gaussian_pcd = gaussians.get_xyz

    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
    top_k = 500
    xfeat = XFeat(top_k=top_k)
    
    #Load image
    img_dir = "./datasets/wholehead/images/seq-01"
    query_img_name = "frame-000000.color.png"
    query_img_name_2 = "frame-000080.color.png" 
    ref_img_name = "frame-000617.color.png"


    
    query_img_path = os.path.join(img_dir, query_img_name)
    query_img = cv2.imread(query_img_path) # [H,W,C] = [480,640,3]
    query_img_tensor = torch.tensor(query_img).permute(2,0,1).cuda() # [C,H,W]
    query_img_tensor = query_img_tensor / 255.0
    
    
    
    query_img_path_2 = os.path.join(img_dir, query_img_name_2)
    query_img_2 = cv2.imread(query_img_path_2) # [H,W,C] = [480,640,3]
    query_img_tensor_2 = torch.tensor(query_img_2).permute(2,0,1).cuda() # [C,H,W]
    query_img_tensor_2 = query_img_tensor_2 / 255.0
    
    
    ref_img_path = os.path.join(img_dir, ref_img_name)
    ref_img = cv2.imread(ref_img_path) # [H,W,C] = [480,640,3]
    ref_img_tensor = torch.tensor(ref_img).permute(2,0,1).cuda() # [C,H,W]
    ref_img_tensor = ref_img_tensor / 255.0
    
    
    # Extract sparse features
    tensor_query_img = xfeat.parse_input(query_img) # [1,C,H,W] = [1,3,480,640]
    query_keypoints, _, query_feature = xfeat.detectAndCompute(tensor_query_img, 
                                                                 top_k=top_k)[0].values()  #query_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate
    
    
      # Extract sparse features
    tensor_query_img_2 = xfeat.parse_input(query_img_2) # [1,C,H,W] = [1,3,480,640]
    query_keypoints_2, _, query_feature_2 = xfeat.detectAndCompute(tensor_query_img_2, 
                                                                 top_k=top_k)[0].values()  #query_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate
    
    
    tensor_ref_img = xfeat.parse_input(ref_img) # [1,C,H,W] = [1,3,480,640]
    ref_keypoints, _, ref_feature = xfeat.detectAndCompute(tensor_ref_img, 
                                                                 top_k=top_k)[0].values()  #query_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate

    index = find_row_index(ref_keypoints, torch.tensor([288,69]).to("cuda"))
    
    print("index = ", index)

    """ 
    for point in ref_keypoints:
            cv2.circle(ref_img, (int(point[0].item()), int(point[1].item())), radius=5, color=(0, 255, 0), thickness=1)  # Green filled circles
    

    cv2.imshow("Coordinates", ref_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    """
    
    idxs0, idxs1 = xfeat.match(query_feature.to("cpu"), query_feature_2.to("cpu"), min_cossim=0.82)
    mkpts_0, mkpts_1 = query_keypoints[idxs0].cpu().numpy(), query_keypoints_2[idxs1].cpu().numpy()
    feat_0, feat_1 = query_feature[idxs0].cpu().numpy(), query_feature_2[idxs1].cpu().numpy()
    canvas, query_points_valid, ref_points_valid, query_feat_valid, ref_feat_valid = warp_corners_and_draw_matches(mkpts_0, mkpts_1, query_img, query_img_2, feat_0, feat_1)

    
    pp_idx = 2
    with torch.no_grad():
        matched_2d, matched_3d, matched_3d_feature = find_2d3d_matching(
                    #torch.tensor(ref_points_valid[pp_idx][None]).to("cuda"),
                    #torch.tensor(ref_feat_valid[pp_idx][None]).to("cuda"),
                    ref_keypoints[index][None],
                    ref_feature[index][None],
                    gaussian_pcd,
                    gaussian_feat
                )
    
    
    
    output = torch.cosine_similarity(torch.tensor(ref_feat_valid[pp_idx]), torch.tensor(query_feat_valid[pp_idx]) , dim=0)
    print("image 0 -> 80 similarity = ", output)
    print("matched 3d coord = ", matched_3d)
    
    output = torch.cosine_similarity(torch.tensor(ref_feature[index]), torch.tensor(query_feat_valid[2]).to("cuda"), dim=0)
    print("image 0 -> image 617 similarity = ", output)
    print("matched 3d coord = ", matched_3d)
    
    output = torch.cosine_similarity(torch.tensor(ref_feature[index]), torch.tensor(ref_feat_valid[2]).to("cuda"), dim=0)
    print("image 80 -> image 617 similarity = ", output)
    print("matched 3d coord = ", matched_3d)
    
    
    """
    output = torch.cosine_similarity(torch.tensor(matched_3d_feature[0]).to("cuda"), torch.tensor(query_feat_valid[2]).to("cuda"), dim=0)
    print("image 0 -> image 873 3dgs similarity = ", output)
    print("matched 3d coord = ", matched_3d)
    
    output = torch.cosine_similarity(torch.tensor(matched_3d_feature[0]).to("cuda"), torch.tensor(ref_feat_valid[2]).to("cuda"), dim=0)
    print("image 80 -> image 873 3dgs similarity = ", output)
    print("matched 3d coord = ", matched_3d)
    
    """
    
    
    plt.figure(figsize=(12,12))
    plt.imshow(canvas[..., ::-1]), plt.show()
    
    
 

def launch_inference(dataset : ModelParams, pipeline : PipelineParams, args): 
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(os.path.join(dataset.model_path,
                                    "point_cloud",
                                    "iteration_" + str(args.iteration),
                                    "point_cloud.ply"))
    
    localize_set(gaussians, args)


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












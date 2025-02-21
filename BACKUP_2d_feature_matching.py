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



def find_2d2d_correspondences(query_keypoints, query_feature, proj_feature, proj_keypoints, chunk_size=10000):
    
    f_N, feat_dim = query_feature.shape
    P_N = proj_feature.shape[0]
    
    # Normalize features for faster cosine similarity computation
    query_feature = F.normalize(query_feature, p=2, dim=1)
    proj_feature = F.normalize(proj_feature, p=2, dim=1)
    
    max_similarity = torch.full((f_N,), -float('inf'), device="cuda")
    max_indices = torch.zeros(f_N, dtype=torch.long, device="cuda")
    
    for part in range(0, P_N, chunk_size):
        chunk = proj_feature[part:part + chunk_size]
        # Use matrix multiplication for faster similarity computation
        similarity = torch.mm(query_feature, chunk.t())
        
        chunk_max, chunk_indices = similarity.max(dim=1)
        
        update_mask = chunk_max > max_similarity
        max_similarity[update_mask] = chunk_max[update_mask]
        max_indices[update_mask] = chunk_indices[update_mask] + part

    point_vis = proj_keypoints[max_indices].cpu().numpy().astype(np.float32)
    keypoints_matched = query_keypoints[..., :2].cpu().numpy().astype(np.float32)
    
    return point_vis, keypoints_matched

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
        
        
"""
        
def find_query_ref_feature_matching(query_keypoints, query_feature, proj_feature, proj_keypoints):
    #reshape the proj_featur from 64x48x64 to 3720x64
    proj_feature = torch.reshape(proj_feature, (3720,64))
    print("reshape feature size = ", proj_feature.shape)
    
    #Normalization
    proj_feature = F.normalize(proj_feature, p=2, dim=1)
    query_feature = F.normalize(query_feature, p=2, dim=1)
    
    #Cosine similarity 
    q_N, feat_dim = query_feature.shape
    max_similarity = torch.full((q_N,), -float('inf'), device=device)
    max_indices = torch.zeros(q_N, dtype=torch.long, device=device)
    
    similarity = torch.mm(proj_feature, query_feature.t())
    max_value, max_indices = similarity.max(dim=1)
    
    
 """   
        
    

    


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
    ref_img_name_noext = "frame-000065"
    ref_img_name = "frame-000065.color.png"

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
    #-------------------------------------------------#
    feature_map, points_in_render_image,  depth_map = render_pkg["feature_map"], render_pkg["points_in_render_images"], render_pkg["depth"] 
    
    #feature_map = F.interpolate(feature_map.unsqueeze(0), size=(60, 80), mode='bilinear', align_corners=True).squeeze(0) #640x480
    #loss = l1_loss(feature_map, que[0].semantic_feature)
    #print("l1 loss = ", loss)

    # Get the proj pixel 
    points_x, points_y, points_z, pw = points_in_render_image[0].tolist(), points_in_render_image[1].tolist(), points_in_render_image[2].tolist(), points_in_render_image[3].tolist()
    points_xyzw = [(x, y, z, w) for x,y,z,w in zip(points_x, points_y, points_z, pw)]
    proj_p_number = len(points_x) - points_x.count(-1)
    proj_p_feature = torch.zeros(proj_p_number, 64)
    proj_p_xyzw = torch.zeros(proj_p_number,4)
    index = 0
    
    """
    for xyzw in points_in_render_image.T:
        if index > 499:
            break
        p_x, p_y = xyzw[0].item(), xyzw[1].item()
        if p_x < 480 and p_y < 640 and p_x > 0 and p_y > 0:
            proj_p_feature[index] = feature_map[:,int(xyzw[0]), int(xyzw[1])]  # feature map = [C, H, W], x-->W,  y-->H
            print(index)
            index = index + 1
            
    """
    
    for xyzw in points_xyzw:
        if xyzw[0] !=-1 and xyzw[1] != -1 and xyzw[0] < 480 and xyzw[1] < 640 and xyzw[0]>0 and xyzw[1]>0:
            proj_p_feature[index] = feature_map[:,int(xyzw[0]), int(xyzw[1])]
            proj_p_xyzw[index, 0], proj_p_xyzw[index, 1], proj_p_xyzw[index, 2], proj_p_xyzw[index, 3] = xyzw[1], xyzw[0], xyzw[2], xyzw[3]
            index = index + 1      
          
    """
    #get the validate point number
    points_x , points_y= points_in_render_image[0].tolist(), points_in_render_image[1].tolist()
    points_xy = [(x, y) for x,y in zip(points_x, points_y)]
    proj_p_number = len(points_x) - points_x.count(-1)
    proj_p_feature = torch.zeros(proj_p_number, 64)
    proj_p_xy = torch.zeros(proj_p_number, 2)
    #update the proj_p_feature (this take almost 1 min)
    index = 0
    for xy in points_xy:
        if xy[0] !=-1 and xy[1] != -1 and xy[0] < 480 and xy[1] < 640 and xy[0]>0 and xy[1]>0:
            proj_p_feature[index] = feature_map[:,int(xy[0]), int(xy[1])]
            proj_p_xy[index, 0], proj_p_xy[index, 1] = xy[1], xy[0]
            index = index + 1
    """

    #proj_points_match, query_points_match = find_2d2d_correspondences(query_keypoints, query_feature, proj_p_feature.to("cuda"), proj_p_xy.to("cuda"))
    ############## Visualize the matching ########################
    idxs0, idxs1 = xfeat.match(query_feature.to("cpu"), proj_p_feature, min_cossim=0.82 )
    mkpts_0, mkpts_1 = query_keypoints[idxs0].cpu().numpy(), proj_p_xyzw[idxs1].cpu().numpy()

    
    canvas, query_points_valid, proj_points_valid = warp_corners_and_draw_matches(mkpts_0, mkpts_1, query_img, ref_img)


    
    #plt.figure(figsize=(12,12))
    #plt.imshow(canvas[..., ::-1]), plt.show()
    
    
    ##############################################################
    
    
    """
    proj_depth_match = torch.zeros(len(proj_points_valid), 1)
    index = 0
    for xy in proj_points_valid:
        proj_depth_match[index] = depth_map[:,int(xy[1]), int(xy[0])]
        index = index + 1
    """

    match_3d = getAllOrigPoints(proj_points_valid, 640,480, ref[0].full_proj_transform.to("cpu"))
    #match_3d = getWorldCoordinates(proj_points_valid, proj_depth_match, torch.from_numpy(K_query), torch.from_numpy(ref[0].R), torch.from_numpy(ref[0].T))

    _, R, t, _ = cv2.solvePnPRansac(np.array(match_3d), np.array(query_points_valid), 
                                                  K_query, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=args.ransac_iters
                                                  )
    R, _ = cv2.Rodrigues(R) 
    
    print("R = ", R  , "  t= ", t)
    print("ref R = ", ref[0].R, "que t = ", ref[0].T)
    rotError, transError = calculate_pose_errors(ref[0].R, ref[0].T, R.T, t)

    # Print the errors
    print(f"Rotation Error: {rotError} deg")
    print(f"Translation Error: {transError} cm")
    
    plt.figure(figsize=(12,12))
    plt.imshow(canvas[..., ::-1]), plt.show()
    
    #F, mask = cv2.findFundamentalMat(proj_points_match,query_points_match,cv2.FM_8POINT)
    #E = np.transpose(K_ref) @ F @ K_query
    
        
        
    
    
    """
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            gt_im = view.original_image[0:3, :, :]

            # Extract sparse features
            gt_keypoints, _, gt_feature = xfeat.detectAndCompute(gt_im[None], 
                                                                 top_k=args.top_k)[0].values()

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
                    gt_keypoints,
                    gt_feature,
                    gaussian_pcd,
                    gaussian_feat
                )

            gt_R = view.R
            gt_t = view.T

            print(f"Match speed: {time.time() - start}")
            _, R, t, _ = cv2.solvePnPRansac(matched_3d, matched_2d, 
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

            prior_rErr.append(rotError)
            prior_tErr.append(transError)

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












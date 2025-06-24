
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import cv2

"""
Introduction: 
     The refinement problem can be consider as an optimization problem 
     Two variable to be optimized: SFM 3D matched point and Pose
     We know that if both the 3D matched point and Pose is correct. The projection of 3D matched point 
     will be exactly at the same position as the query 2D keypoint. 
     
     Now we have a Coarse pose given by 2D 3D matching
     The refinement will be composed as the following step:
        * For each iteration:
             * For each matching 3D point,  project it into pixel space    
             Optimize the Pose (fix the 3D position): 
                 * Get the 8x8 neighbor pixel feature A of matched query keypoint (2D)
                 * Project the whole SFM with given pose and get a sparse projection feature map. 
                 * Project the matched 3D SFM into the same sparse feature map and for each projection get the 32 cloest points 
                   (each point correspond a point in SFM) and its feature B
                 * Dense matching A and B then using PnP to calculate the updated pose P 
        
              Optimize the 3D matched position (fix the pose):
                  * Project the matched 3D position into pixel space, calculate the loss between p
                    projected position adn query matched keypoint(2D), Use this loss to update the
                    position of 3D and repeat it several times
    
    The optimization process is like Bundle ajustment(BA). The difference is that we alternatively optimize the pose and 3D position instead 
    of optimize them together jointly as BA does. For updating the pose, we use the neighbor feature instead of purely geometry as BA.
    
    This optimization face the local minimum problem so we add the random inside the process. When optimize the pose, for each projected macthed 3D, we firstly choose 64 
    cloest points and then randomly select 32 to have its feature 

"""

"""
Full projection Function
"""
def ndc2pixel(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def fullproj(point_3d, full_proj_matrix, W, H):
    """
    Project the 3D point cloud into pixel space
    Using the 3DGS projection methods: World_coord --> Camera_coord 
    --> NDC_coord--> Pixel_coord 
    
    """
    hom = torch.matmul(point_3d, full_proj_matrix)
    weight = 1.0/(hom[:,3] + 0.000001)
    return ndc2pixel(hom[:,0]*weight, W), ndc2pixel(hom[:,1]*weight, H)



def project_and_filter(points_3d, points_feat,P, W, H):
    """
    Project the 3D points into 2D pixel spaces with given projection matrix
    The pixel that is projected out of the pixel space [0:W, 0:H] will be rejected
    
    Input:  
         points_3d: [N, 3] Tensor
         points_feat: [N,C] Tensor: the feature associated with each 3D points 
         P: [3, 4] Projection Matrix Tensor

    Return :
         mask 
         result: [M, 5] Tensor, Each line  [x, y, X, Y, Z] contains its pixel coordinates (x, y) and its asscociated 
                 3D point cloud coordinates (X, Y , Z)   
         points_feat_filter: [M, C] the feature of all the projected keypoint that is inside the pixel space 
    """
    N = points_3d.shape[0]
    
    # Construct homogeneous coordinates [X, Y, Z, 1]
    ones = torch.ones((N, 1), dtype=points_3d.dtype, device=points_3d.device)
    points_homogeneous = torch.cat([points_3d, ones], dim=1)  # [N, 4]
    
    # Project 3D points into pixel space 
    x,y = fullproj(points_homogeneous, P, 640, 480)
    
    # Keep only the projected pixel that is inside the pixel space ：x ∈ (0, 640), y ∈ (0, 480)
    mask = (x > 0) & (x < W) & (y > 0) & (y < H)
    
    x_filtered = x[mask]
    y_filtered = y[mask]
    points_3d_filtered = points_3d[mask]  # [M, 3]
    points_feat_filtered = points_feat[mask]

    # Concetenate to [M, 5]： [x, y, X, Y, Z]
    result = torch.cat([x_filtered.unsqueeze(1), 
                        y_filtered.unsqueeze(1), 
                        points_3d_filtered], dim=1)   
    
    return mask, result, points_feat_filtered


def get_cloest_3d_indice(midpixels, full_proj):
    """
    Objectif: For each midpixel, find its cloest pixel that correspond to a 3D points
              in SFM. This indice will then be used to find its correspondance 3D coordinates
              
    Input:
              midpixels : [N, 2]
              full_proj: [M, 2] All the proj pixel from 3D points
    Output:
              indice [N] indice in full_proj that macth with each query keypoints
    
    """
    # Get the index for each midpoints
    # A: [N, 1, 2], B: [1, M, 2] =>  [N, M, 2]
    diff = midpixels[:, None, :] - full_proj[None, :, :]  # pairwise differences
    
    # Euclidean distance squared (without sqrt for efficiency)
    dists = (diff ** 2).sum(dim=2)  # shape [N, M]

    # Indices of minimum values per row, corresponding to nearest B point for each A point
    nearest_indices = torch.argmin(dists, dim=1)  # shape [N]
    
    return nearest_indices.cpu().numpy() 
    





def generate_mask_matrix(rows, cols, num_true, device='cuda'):
    # Step 1: Randomly sample indices of shape [rows, num_true]
    # For each row, generate all column indices, shuffle them, and pick the first num_true
    all_indices = torch.rand(rows, cols, device=device).argsort(dim=1)
    true_indices = all_indices[:, :num_true]  # [rows, num_true]

    # Step 2: Prepare batch indices [0, 1, ..., rows-1], repeated num_true times for scatter
    row_indices = torch.arange(rows, device=device).unsqueeze(1).expand(-1, num_true)

    # Step 3: Create an all-False mask, then scatter to set True at specific positions
    mask = torch.zeros((rows, cols), dtype=torch.bool, device=device)
    mask[row_indices, true_indices] = True

    return mask


def knn(A, B, B_3D, F, k=32):
    """
    For each point A[i], find the k nearest neighbors from B[j], and extract their corresponding coordinates and features.
    Here I use a random trick. That firstly get 2*k nearest neighbors and then randomly select k neigbors. This is proved to
    be more accurate 
    Parameters:
         A: [N,2] Query points
         B: [M,2] All point coordinates
         B_3D: [M,3] All 3D coordinates
         F: [M,64][M,64] All point features
         k: int Number of nearest neighbors to return for each query point

    Return:
        B_coords_selected: [N, k, 2]
        B_feats_selected:  [N, k, 64]
        B_3d_coords : [N, K, 3]
    """
    device = A.device
    N, M = A.shape[0], B.shape[0]
    
    # Generate the random mask
    mask = generate_mask_matrix(N,2*k, k)

    dists = torch.cdist(A, B, p=2)  # Euclidean distance [N, M]
    
    # Find the indices of the k smallest distances
    knn_dists, knn_indices = torch.topk(dists, k=2*k, dim=1, largest=False)


    # Retrieve the corresponding coordinates (2D and 3D) and features from B 
    B_coords_selected = B[knn_indices*mask]    # [N, k, 2]
    B_feats_selected = F[knn_indices*mask]     # [N, k, 64]
    B_3D_selected = B_3D[knn_indices*mask]
    
    return B_coords_selected, B_feats_selected, B_3D_selected




def mnn_match(corr_matrix):
    """
    mutual nearest neighbor
    corr_matrix: [N, 64, 64] 
    Returns:
        padded_a: (N, K) long tensor of A indices (padded)
        padded_b: (N, K) long tensor of B indices (padded)
        mask: (N, K) bool tensor where mask[i] = True means (padded_a[i], padded_b[i]) is valid MNN
    """
    N, A, B = corr_matrix.shape
    device = corr_matrix.device

    # A -> B
    A_to_B = torch.argmax(corr_matrix, dim=2)  # [N, A]
    # B -> A
    B_to_A = torch.argmax(corr_matrix, dim=1)  # [N, B]

    # Create index grid: [N, A] index of A
    a_idx = torch.arange(A, device=device).unsqueeze(0).expand(N, A)
    b_idx = A_to_B  # [N, A]
    
    # Use b_idx to look up B_to_A: [N, A] -> get a' = B_to_A[b]
    bta = B_to_A.gather(1, b_idx)  # [N, A]

    # MNN condition: B_to_A[b] == a
    mnn_mask = (bta == a_idx)  # [N, A] bool

    # Now, filter only matched (a, b)
    matched_a = a_idx[mnn_mask]  # 1D tensor
    matched_b = b_idx[mnn_mask]  # 1D tensor
    matched_batch = torch.arange(N, device=device).unsqueeze(1).expand(N, A)[mnn_mask]  # [K]

    # Pack into fixed-size output: e.g., pad to max per batch
    # Count number of matches per batch
    from torch.nn.utils.rnn import pad_sequence

    # Split by batch
    a_list = [matched_a[matched_batch == i] for i in range(N)]
    b_list = [matched_b[matched_batch == i] for i in range(N)]

    # Pad to same length (max MNN per batch item)
    padded_a = pad_sequence(a_list, batch_first=True, padding_value=-1)  # shape (N, max_K)
    padded_b = pad_sequence(b_list, batch_first=True, padding_value=-1)
    mask = (padded_a != -1)  # valid mask

    return padded_a, padded_b, mask
        

def dual_softmax(corr_matrix, temp=1):
    corr_matrix = corr_matrix / temp
    corr_matrix = F.softmax(corr_matrix, dim=-2) * F.softmax(corr_matrix, dim=-1)
    return corr_matrix


def extract_patch_features_with_coords(A, fmap, patch_size=8):
    """
   Extract 8x8 patch features and pixel coordinates centered at each point in A.
   Parameters:
        A: [N, 2]  tensor of pixel coordinates (x, y)
        fmap: [C, H, W] feature map tensor
        patch_size: size of the patch, default is 8

    Return:
        patch_feats:  [N, 64, C]  - patch features for each point
        patch_coords: [N, 64, 2]  - pixel coordinates (x, y) of each path
    """
    device = fmap.device
    N = A.shape[0]
    C, H, W = fmap.shape
    half = patch_size // 2

    # Normalize the coordinates of A to the range [-1, 1] for use with grid_sample
    norm_x = (A[:, 0] / (W - 1)) * 2 - 1
    norm_y = (A[:, 1] / (H - 1)) * 2 - 1

    # Construct an 8x8 grid of pixel offsets
    offset = torch.linspace(-half + 0.5, half - 0.5, steps=patch_size).to(device)
    dy, dx = torch.meshgrid(offset, offset, indexing='ij')  # [8, 8]
    grid_offsets = torch.stack((dx, dy), dim=-1).view(1, patch_size * patch_size, 2)  # [1, 64, 2]
    grid_offsets = grid_offsets.expand(N, -1, -1)  # [N, 64, 2]

    # Actual pixel coordinates (unnormalized)
    A_pixel = A.unsqueeze(1)  # [N, 1, 2]
    patch_coords = A_pixel + grid_offsets  # [N, 64, 2]

    # Construct normalized coordinates for grid_sample
    norm_offsets = grid_offsets / torch.tensor([W - 1, H - 1], device=device) * 2
    centers = torch.stack([norm_x, norm_y], dim=1).unsqueeze(1)  # [N, 1, 2]
    sample_grid = centers + norm_offsets  # [N, 64, 2]
    sample_grid = sample_grid.view(N, patch_size, patch_size, 2).float()  # [N, 8, 8, 2]

    # Extract features using grid_sample
    fmap = fmap.unsqueeze(0).expand(N, -1, -1, -1)  # [N, C, H, W]
    patch = F.grid_sample(fmap, sample_grid, mode='bilinear', align_corners=True)  # [N, C, 8, 8]
    patch_feats = patch.permute(0, 2, 3, 1).reshape(N, patch_size * patch_size, C)  # [N, 64, C]

    return patch_feats, patch_coords  # [N, 64, C], [N, 64, 2]

def get_query_coord_from_index(idx_tensor,coords_tensor, dim=2):
    """
    Get the matching pixel coords in query image, given by coords_tensor.
    idx_tensor : [N, K] N is the number of pixels and K is the number of index for each pixels
                It is -1 padding 
    coords_tensor: [N, 64, 2]: pixel coordinates within a 8x8 regional patch 
    
    return:
           q_coords: [N, K, 2]
    
    """
    # Step 1: Replace invalid indices (-1) with 0 because gather doesn't support negative indices
    safe_idx = idx_tensor.clone()
    safe_idx[safe_idx == -1] = 0  # shape: [B, K]

    # Step 2: Expand indices to [B, K, 1] to match coordinate tensor shape for gather
    safe_idx_expanded = safe_idx.unsqueeze(-1)  # [B, K, 1]

    # Step 3: Gather coordinates along dim=1 using the prepared indices
    # coords_tensor shape: [B, 64, 2] → 从 dim=1 上索引
    gathered_coords = torch.gather(coords_tensor, dim=1, index=safe_idx_expanded.expand(-1, -1, dim))  # [B, K, 2]

    # Step 4: Zero out coordinates at positions corresponding to original -1 indices
    mask = (idx_tensor == -1).unsqueeze(-1).expand(-1, -1, dim)  # [B, K, 2]

    # Step 5 : Set coordinates to zero at positions where original indices were -1
    gathered_coords = gathered_coords.masked_fill(mask, 0.0)  
    
    return gathered_coords

def remove_invalid(mat):
    """
    Remove the coordinate that has 0 (which means invalid coordinates)
    """
    mask = mat.any(dim=-1)
    return mat[mask]

    

def get_refine_2d3d(matched_3d_proj,  pixel_pc, pixel_feat,  query_neigbor_pts, query_neigbor_feats, mask):
    """
      Get the neigbor region feature for both query matched 2D keypoint and projected 3D matched point
      The matching is valid when they are mutual nearest matching (MNN)
      
      Input:
         matched_3d_proj: [N, 2] the pixel coordinate of all the matched 3D points
         pixel_pc : [M, 2] all the projected points from SFM , keep only the points that project inside the image 
         pixel_feat: [M, C]:  all the projected points feature (the same, inside the image)from SFM 
         query_neigbor_pts: tensor [N, 64, 2] For each keypoint(pixel) in query feature map, gets its 8x8 neigbor pixels' coordinates 
         query_neigbor_feats : tensor [N, 64, C] For each keypoint(pixel) in query feature map, gets its 8x8 neigbor pixels' coordinates
         mask: keep only the matched 3D points whose projection is inside the image 
       Output:
          q_pixel: updated query neigbor matched corrdinate
          proj_3d: update sfm neigbor matched coordinate 
    
    """    
     # Get the neigbor pixel of projected 3D matched points
    _, proj_neigbor_feats, proj_neigbor_3d =  knn(matched_3d_proj[:,:2], pixel_pc[:,:2], pixel_pc[:,2:], pixel_feat)
    

    # Normalize the feature 
    proj_neigbor_feats = F.normalize(proj_neigbor_feats, dim=2) 
    

    # Get the correlation matrix
    fine_corr_matrix = torch.matmul(
            proj_neigbor_feats, query_neigbor_feats[mask].transpose(-2, -1)
        )
    
    fine_corr_matrix = dual_softmax(
            fine_corr_matrix, temp=0.1
        )


    L_A2B, L_B2A, _ = mnn_match(
            fine_corr_matrix
        )
    
    
    # Get the query image 2D pixel coords
    q_pixel = get_query_coord_from_index(L_B2A, query_neigbor_pts[mask])  # [N, K, 2]
    
    # Get the proj mid neigbor pixel
    proj_3d = get_query_coord_from_index(L_A2B, proj_neigbor_3d, dim=3) # [N,K,3]

         
    # return the 2D pixels and 3D points matching and the new midpoint 3D position
    q_pixel = remove_invalid(q_pixel)
    proj_3d = remove_invalid(proj_3d)
    
    
    return  q_pixel, proj_3d

def optimize_pose(matched_3d_proj,  pixel_pc, pixel_feat, query_neigbor_pts, query_neigbor_feats, mask, K):
    """ Optimize the Pose
    """
    
    # Get the update 2D 3D pairs using its neighbor pixel
    pnp_2d, pnp_3d = get_refine_2d3d(matched_3d_proj,  pixel_pc, pixel_feat, query_neigbor_pts, query_neigbor_feats, mask)
    
    # Update the Pose
    _, fine_R, fine_t, inl = cv2.solvePnPRansac(pnp_3d.cpu().numpy(), pnp_2d.cpu().numpy(), 
                                                  K, 
                                                  distCoeffs=None, 
                                                  flags=cv2.SOLVEPNP_ITERATIVE, 
                                                  iterationsCount=20000
                                                  )
                
    
    return fine_R, fine_t, inl
    
def optimize_3D(X_3D, x_2d, full_proj, steps=100):
    """ Optimize the 3D position
    """
    N = X_3D.shape[0]
    x_2d = torch.tensor(x_2d).cuda()
    # Construct homogeneous coordinates [X, Y, Z, 1]
    ones = torch.ones((N, 1), dtype=X_3D.dtype, device=X_3D.device)
    X_3D = torch.cat([X_3D, ones], dim=1)  # [N, 4]
    X_3D.requires_grad = True
    optimizer = optim.NAdam([X_3D], lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad()
        x, y = fullproj(X_3D, full_proj,640, 480)
        x_proj = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1)
        loss = ((x_proj - x_2d) ** 2).mean()
        loss.backward()
        optimizer.step()
    return X_3D[:, :3]


def refiner(matched_2d, matched_3d, matched_3d_feature, view, feat_pcd, feat_feat, query_neigbor_pts, query_neigbor_feats,K):
    """
    Refine the coarse pose
    Input:
         matched_2d: numpy array [N,2] the matched query feature map keypoint(pixel) coordinates
         matched_3d: numpy array [N,3] the matched 3D SFM keypoint(3d) coordinates
         matched_3d_feature: tenosr [N, C] the matched 3D SFM keypoint(3d) feature
         view: an Camera Object that contains the Pose informations 
         feat_pcd: tensor [M, 3]the SFM point cloud coordinates
         feat_feat: tensor [M, C] the SFM point cloud feature
         query_neigbor_pts: tensor [N, 64, 2] For each keypoint(pixel) in query feature map, gets its 8x8 neigbor pixels' coordinates 
         query_neigbor_feats : tensor [N, 64, C] For each keypoint(pixel) in query feature map, gets its 8x8 neigbor pixels' coordinates
         K: camera intrinsic
    
    """
    
    # Get the projection matrix
    full_proj_matrix = view.full_proj_transform
    
    # Project the matched 3D into pixel space 
    # only the pixel that is inside the image
    matched_3d, matched_3d_feature = torch.tensor(matched_3d).cuda().float(), torch.tensor(matched_3d_feature).cuda().float()
    mask, matched_3d_proj, _ = project_and_filter(matched_3d, matched_3d_feature, full_proj_matrix, 640, 480)
    mask = mask.cpu().numpy()
    
    # Project the whole point cloud to 2D pixel space and keep 
    # only the pixel that is inside the image
    _, pixel_pc, pixel_feat =  project_and_filter(feat_pcd, feat_feat, full_proj_matrix, 640, 480)
    
    
    # Refine the pose using neighbor feature 
    updated_R, updated_t, inl = optimize_pose(matched_3d_proj,  pixel_pc, pixel_feat, query_neigbor_pts, query_neigbor_feats, mask, K)
    
    # Update the Pose
    updated_R, _ = cv2.Rodrigues(updated_R)  
    view.update_RT(updated_R.T, updated_t[:,0])
    # Get the updated projection matrix 
    full_proj_matrix = view.full_proj_transform
    
    # Refine the 3D position
    updated_3D = optimize_3D(matched_3d, matched_2d, full_proj_matrix)
    
    
    return view, updated_3D, updated_R, updated_t, inl





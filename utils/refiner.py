
import torch
import torch.nn.functional as F
import numpy as np


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
    



def get_midpoints(query_kp, proj_kp, device='cuda'):
    """
    Get the midpoints coordinates.
    After matching 2D query feature keypoint with 3D SFM points, We project the matching 3D SFM points into 
    pixel space. We firstly connect each project pixel with its matching pixel in query feature map. If the matching points 
    is correct, all the pixel(query)-pixel(proj) "connection" will be parellel.
    So according to each connection, we use the optimization process to get this parellel line that begins at each project pixel 
    with less angle difference sum with its original connection. Then the midpoint will be found along side each parellel line
    that takes half of the original connection distance
    
    Objectif: Push the projected pixel into query pixel. Similar to Gradient descent, The parellel line work as the moving direction and 
              the distance work as the step    
    
    query_kp : tensor [N, 2]
    proj_kp: tensor [N, 2]
    """
    # Set devices 
    query_kp, proj_kp = query_kp.to(device).float(), proj_kp.to(device).float()
        
    # Initialize a random direction vector d and make it differentiable
    d = torch.randn(2, device=device, requires_grad=True)
    
    optimizer = torch.optim.Adam([d], lr=0.05)
    
    previous_loss = 0

    for step in range(1000):
        optimizer.zero_grad()

        d_norm = d / torch.norm(d)  # Unit vector direction 

        # Calculate the difference vector between each pair of query keypoint and matched project keypoint in pixel space
        seg_vecs = query_kp - proj_kp
        seg_lengths = torch.norm(seg_vecs, dim=1)

        # Compute the cosine similarity
        cos_angles = torch.sum(seg_vecs * d_norm, dim=1) / seg_lengths

        # Optimization objective: minimize the total angle, i.e., maximize the cosine. 
        # The total loss is defined as (1 - cos).
        loss = torch.sum(1 - cos_angles)
        if abs(loss.item() - previous_loss) < 0.001:
            break

        loss.backward()
        optimizer.step()
        previous_loss = loss.item()
        """
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
        """
    # Final direction vector (normalized)
    final_d = d / torch.norm(d)
    # final midpoints
    midpoits = proj_kp + 0.5 * seg_lengths.view(-1, 1) * final_d.view(1, 2)
    
    return midpoits

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


def knn(A, B, B_3D, F, k=64):
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

    
    
    
"""
Main function 

"""
def refiner(matched_2d, matched_3d, matched_3d_feature,  full_proj_matrix, feat_pcd, feat_feat, query_neigbor_pts, query_neigbor_feats):
    """
    Refine the coarse pose
    Input:
         matched_2d: numpy array [N,2] the matched query feature map keypoint(pixel) coordinates
         matched_3d: numpy array [N,3] the matched 3D SFM keypoint(3d) coordinates
         matched_3d_feature: tenosr [N, C] the matched 3D SFM keypoint(3d) feature
         full_proj_matrix: the world-to-pixel projection matrix 
         feat_pcd: tensor [M, 3]the SFM point cloud coordinates
         feat_feat: tensor [M, C] the SFM point cloud feature
         query_neigbor_pts: tensor [N, 64, 2] For each keypoint(pixel) in query feature map, gets its 8x8 neigbor pixels' coordinates 
         query_neigbor_feats : tensor [N, 64, C] For each keypoint(pixel) in query feature map, gets its 8x8 neigbor pixels' coordinates
    
    return value:
         q_pixel : [L,2]  all the neighbor pixels that has a match with 3D SFM point, which is then use to calculate PnP
         proj_3d: [L,3]  all the 3D SFM point that match with q_pixel, which is then use to calculate PnP
         temp_kp_3d : [Z, 3] the 3D coords of each midpoints
         temp_kp_3d_feat : [Z, C] the feature of midpoints
         matched_2d[mask]: updated (reject all the proj pixel that is outside of the space) matched query 2D keypoint (pixel) 
         query_neigbor_pts[mask]: updated query keypoints neighbor pixles 
         query_neigbor_feats[mask]: updated query keypoint neighbor pixels features
         
         updated_matched_3d : [N, 3]
    """
    # Project the matched 3d into 2D pixel space and keep
    # only the pixel that is inside the image
    matched_3d, matched_3d_feature = torch.tensor(matched_3d).cuda().float(), torch.tensor(matched_3d_feature).cuda().float()
    mask, matched_3d_proj, _ = project_and_filter(matched_3d, matched_3d_feature, full_proj_matrix, 640, 480)
    mask = mask.cpu().numpy()
    
    # Project the whole point cloud to 2D pixel space and keep 
    # only the pixel that is inside the image
    _, pixel_pc, pixel_feat =  project_and_filter(feat_pcd, feat_feat, full_proj_matrix, 640, 480)
    
    
    # Get the midpoints
    """
    Two choices: 
    1. Use the midpoints, which means that the refinement follows the updated position with each step  
    """
    midpoints = get_midpoints(torch.tensor(matched_2d[mask]), matched_3d_proj[:,:2])
    
    """
    2. Keep the refinement only in the neiborhood of project pixel without updating its position.
       This is proved to be more accurate the choice 1
    """
    #midpoints = matched_3d_proj[:,:2]
    
    
    # Get the cloest pixel that match a 3D points in SFM
    matches = get_cloest_3d_indice(midpoints, pixel_pc[:,:2])
    temp_kp_3d = pixel_pc[:,2:][matches]
    temp_kp_3d_feat = pixel_feat[matches]
    
    # Get the neigbor pixel of midpoints
    _, mid_neigbor_feats, mid_neigbor_3d =  knn(midpoints, pixel_pc[:,:2], pixel_pc[:,2:], pixel_feat)
    

    # Normalize the feature 
    mid_neigbor_feats = F.normalize(mid_neigbor_feats, dim=2) 
    

    # Get the correlation matrix
    fine_corr_matrix = torch.matmul(
            mid_neigbor_feats, query_neigbor_feats[mask].transpose(-2, -1)
        )
    
    fine_corr_matrix = dual_softmax(
            fine_corr_matrix, temp=0.1
        )


    L_A2B, L_B2A, mask_mnn = mnn_match(
            fine_corr_matrix
        )
    
    
    # Get the query image 2D pixel coords
    q_pixel = get_query_coord_from_index(L_B2A, query_neigbor_pts[mask])  # [N, K, 2]
    
    # Get the proj mid neigbor pixel
    proj_3d = get_query_coord_from_index(L_A2B, mid_neigbor_3d, dim=3) # [N,K,3]

         
    # return the 2D pixels and 3D points matching and the new midpoint 3D position
    q_pixel = remove_invalid(q_pixel)
    proj_3d = remove_invalid(proj_3d)
    
    
    return  q_pixel, proj_3d, temp_kp_3d, temp_kp_3d_feat, matched_2d[mask], query_neigbor_pts[mask], query_neigbor_feats[mask]





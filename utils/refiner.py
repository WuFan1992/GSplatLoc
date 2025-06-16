
import torch
import torch.nn.functional as F


"""
Full projection Function
"""
def ndc2pixel(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def fullproj(point_3d, full_proj_matrix, W, H):
    hom = torch.matmul(point_3d, full_proj_matrix)
    weight = 1.0/(hom[:,3] + 0.000001)
    return ndc2pixel(hom[:,0]*weight, W), ndc2pixel(hom[:,1]*weight, H)



def project_and_filter(points_3d, points_feat,P, W, H):
    """
    points_3d: [N, 3] Tensor
    P: [3, 4] 投影矩阵 Tensor
    points_feat : [N, 64]: 
    返回:
         mask 
         result: [M, 5] Tensor, 每行是 [x, y, X, Y, Z]
         points_feat_filter: [M, 64]
    """
    N = points_3d.shape[0]
    
    # 添加齐次坐标列 [X, Y, Z, 1]
    ones = torch.ones((N, 1), dtype=points_3d.dtype, device=points_3d.device)
    points_homogeneous = torch.cat([points_3d, ones], dim=1)  # [N, 4]
    
    x,y = fullproj(points_homogeneous, P, 640, 480)
    
    
    # 筛选条件：x ∈ (0, 640), y ∈ (0, 480)
    mask = (x > 0) & (x < W) & (y > 0) & (y < H)
    

    # 保留满足条件的点
    x_filtered = x[mask]
    y_filtered = y[mask]
    points_3d_filtered = points_3d[mask]  # [M, 3]
    points_feat_filtered = points_feat[mask]

    # 拼接成 [M, 5]： [x, y, X, Y, Z]
    result = torch.cat([x_filtered.unsqueeze(1), 
                        y_filtered.unsqueeze(1), 
                        points_3d_filtered], dim=1)   
    
    return mask, result, points_feat_filtered


def get_cloest_3d_indice(midpixels, full_proj):
    """
    Objectif: For each midpixel, find its cloest pixel that correspond to a 3D points
              in SFM
    Input:
              midpixels : [N, 2]
              full_proj: [M, 2] All the proj pixel from 3D points
    Output:
              indice [N] indice in full_proj that macth with each query keypoints
    
    """
    # Get the index for each midpoints
    # A: [N, 1, 2], B: [1, M, 2] => 广播得到 [N, M, 2]
    diff = midpixels[:, None, :] - full_proj[None, :, :]  # pairwise differences
    
    # 欧几里得距离（不取 sqrt 保持效率）
    dists = (diff ** 2).sum(dim=2)  # shape [N, M]

    # 每行最小值的索引，即 A 中每个点最近的 B 中的点的索引
    nearest_indices = torch.argmin(dists, dim=1)  # shape [N]
    
    return nearest_indices.cpu().numpy() 
    


def get_updated_3d_indice(query_kp, query_feats, render_kp, render_feats, device= 'cuda'):
    """
    query_kp : tensor [N, 2]
    render_kp: tensor [N, 2]
    """
    # Find the middle coordinate
    midpoints = get_midpoints(query_kp, render_kp)
    
    # Get the 64 neibors pixels of midpoints for refinement
    mid_neigbor_pts, mid_neigbor_feats = knn(midpoints, render_kp, render_feats) 
    
    # Get the 64 neibors pixel of query keypoints (one time)
    query_neigbor_pts, query_neigbor_feats = knn(query_kp, render_kp, render_feats)
    
    # Get the  
    
    # Get the index for each midpoints
    # A: [N, 1, 2], B: [1, M, 2] => 广播得到 [N, M, 2]
    diff = midpoints[:, None, :] - projected_points[None, :, :]  # pairwise differences

    # 欧几里得距离（不取 sqrt 保持效率）
    dists = (diff ** 2).sum(dim=2)  # shape [N, M]

    # 每行最小值的索引，即 A 中每个点最近的 B 中的点的索引
    nearest_indices = torch.argmin(dists, dim=1)  # shape [N]
    
    return nearest_indices.cpu().numpy()


def get_midpoints(query_kp, render_kp, device='cuda'):
    """
    query_kp : tensor [N, 2]
    render_kp: tensor [N, 2]
    """
    # Set devices 
    query_kp, render_kp = query_kp.to(device).float(), render_kp.to(device).float()
    

    
    # 初始化一个随机方向向量 d，并让它可导
    d = torch.randn(2, device=device, requires_grad=True)
    

    optimizer = torch.optim.Adam([d], lr=0.05)
    
    previous_loss = 0

    for step in range(1000):
        optimizer.zero_grad()

        d_norm = d / torch.norm(d)  # 单位向量方向

        # 每个 ai-Ai 向量
        seg_vecs = query_kp - render_kp
        seg_lengths = torch.norm(seg_vecs, dim=1)

        # 计算余弦夹角的 cos 值
        cos_angles = torch.sum(seg_vecs * d_norm, dim=1) / seg_lengths

        # 优化目标：最小化总夹角，即最大化 cos，总损失设为 (1 - cos)
        loss = torch.sum(1 - cos_angles)
        if abs(loss.item() - previous_loss) < 0.001:
            break

        loss.backward()
        optimizer.step()
        previous_loss = loss.item()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    # 最终方向向量（单位化）
    final_d = d / torch.norm(d)
    # final midpoints
    midpoits = render_kp + 0.5 * seg_lengths.view(-1, 1) * final_d.view(1, 2)
    
    return midpoits




def knn(A, B, F, k=64):
    """
    对每个 A[i] 找出最近的 k 个 B[j]，并提取对应坐标与特征

    参数:
        A: [N, 2] 查询点
        B: [M, 2] 所有点坐标
        F: [M, 64] 所有点特征
        k: int, 每个查询点返回最近的 k 个点

    返回:
        B_coords_selected: [N, k, 2]
        B_feats_selected:  [N, k, 64]
    """
    device = A.device
    N, M = A.shape[0], B.shape[0]

    dists = torch.cdist(A, B, p=2)  # 欧氏距离 [N, M]
    

    # 找到前 k 个最小距离的索引
    knn_dists, knn_indices = torch.topk(dists, k=k, dim=1, largest=False)

    # 获取对应的 B 坐标和特征
    B_coords_selected = B[knn_indices]    # [N, k, 2]
    B_feats_selected = F[knn_indices]     # [N, k, 64]

    return B_coords_selected, B_feats_selected




def mnn_match(corr_matrix):
    """
    corr_matrix: [N, 64, 64] on CUDA
    Returns:
        A_indices: (N, K) long tensor of A indices (padded)
        B_indices: (N, K) long tensor of B indices (padded)
        mask: (N, K) bool tensor where mask[i, j] = True means (A_indices[i, j], B_indices[i, j]) is valid MNN
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
    提取以 A 中每个点为中心的 8x8 patch 特征和像素坐标。

    参数:
        A: [N, 2] Tensor，坐标为 (x, y) 像素坐标
        fmap: [C, H, W] Tensor，特征图
        patch_size: patch 大小，默认 8

    返回:
        patch_feats:  [N, 64, C]  - 每个点的 patch 特征
        patch_coords: [N, 64, 2]  - 每个点的 patch 像素坐标 (x, y)
    """
    device = fmap.device
    N = A.shape[0]
    C, H, W = fmap.shape
    half = patch_size // 2

    # 归一化 A 的坐标到 [-1, 1] 用于 grid_sample
    norm_x = (A[:, 0] / (W - 1)) * 2 - 1
    norm_y = (A[:, 1] / (H - 1)) * 2 - 1

    # 构建 8x8 网格偏移（像素单位）
    offset = torch.linspace(-half + 0.5, half - 0.5, steps=patch_size).to(device)
    dy, dx = torch.meshgrid(offset, offset, indexing='ij')  # [8, 8]
    grid_offsets = torch.stack((dx, dy), dim=-1).view(1, patch_size * patch_size, 2)  # [1, 64, 2]
    grid_offsets = grid_offsets.expand(N, -1, -1)  # [N, 64, 2]

    # 实际像素坐标（未归一化）
    A_pixel = A.unsqueeze(1)  # [N, 1, 2]
    patch_coords = A_pixel + grid_offsets  # [N, 64, 2]

    # 构造 grid_sample 使用的归一化坐标
    norm_offsets = grid_offsets / torch.tensor([W - 1, H - 1], device=device) * 2
    centers = torch.stack([norm_x, norm_y], dim=1).unsqueeze(1)  # [N, 1, 2]
    sample_grid = centers + norm_offsets  # [N, 64, 2]
    sample_grid = sample_grid.view(N, patch_size, patch_size, 2).float()  # [N, 8, 8, 2]

    # 使用 grid_sample 提取特征
    fmap = fmap.unsqueeze(0).expand(N, -1, -1, -1)  # [N, C, H, W]
    patch = F.grid_sample(fmap, sample_grid, mode='bilinear', align_corners=True)  # [N, C, 8, 8]
    patch_feats = patch.permute(0, 2, 3, 1).reshape(N, patch_size * patch_size, C)  # [N, 64, C]

    return patch_feats, patch_coords  # [N, 64, C], [N, 64, 2]
    
    
"""
Main function 

"""
def refiner(matched_2d, matched_3d, matched_3d_feature,  full_proj_matrix, feat_pcd, feat_feat, query_neigbor_pts, query_neigbor_feats):
    """
    return value:
         matched_2d : [N,2]
         updated_matched_3d : [N, 3]
    """
    # Project the matched 3d into 2D pixel space and keep
    # only the pixel that is inside the image
    matched_3d, matched_3d_feature = torch.tensor(matched_3d).cuda().float(), torch.tensor(matched_3d_feature).cuda().float()
    mask, matched_3d_proj, _ = project_and_filter(matched_3d, matched_3d_feature, full_proj_matrix, 640, 480)
    mask = mask.cpu().numpy()
    
    # Project the point cloud to 2D pixel space and keep 
    # only the pixel that is inside the image
    _, pixel_pc, pixel_feat =  project_and_filter(feat_pcd, feat_feat, full_proj_matrix, 640, 480)
    
    
    # Get the midpoints
    midpoints = get_midpoints(torch.tensor(matched_2d[mask]), matched_3d_proj[:,:2])
    
    # Get the cloest pixel that match a 3D points in SFM
    matches = get_cloest_3d_indice(midpoints, pixel_pc[:,:2])
    temp_kp_3d = pixel_pc[:,2:][matches]
    
    # Get the neigbor pixel of midpoints
    mid_neigbor_pts, mid_neigbor_feats = knn(midpoints, pixel_pc[:,:2], pixel_feat)
    

    # Normalize the feature 
    mid_neigbor_feats = F.normalize(mid_neigbor_feats, dim=2) 
    

    
    # Get the correlation matrix
    fine_corr_matrix = torch.matmul(
            mid_neigbor_feats, query_neigbor_feats[mask].transpose(-2, -1)
        )
    
    fine_corr_matrix = dual_softmax(
            fine_corr_matrix, temp=0.1
        )


    L_A2B, L_B2A, mask = mnn_match(
            fine_corr_matrix
        )
    
    print("L_A2B = ", L_A2B)
    print("mask = ", mask)

    
    

    
    # update the 3D position of each keypoint
    #matches = get_updated_3d_indice(torch.tensor(matched_2d[mask]), matched_3d_proj[:,:2], pixel_pc[:,:2])
    
    #return matched_2d[mask], pixel_pc[:,2:][matches]    



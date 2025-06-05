
import torch


"""
Full projection Function
"""
def ndc2pixel(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def fullproj(point_3d, full_proj_matrix, W, H):
    hom = torch.matmul(point_3d, full_proj_matrix)
    weight = 1.0/(hom[:,3] + 0.000001)
    return ndc2pixel(hom[:,0]*weight, W), ndc2pixel(hom[:,1]*weight, H)



def project_and_filter(points_3d, P, W, H):
    """
    points_3d: [N, 3] Tensor
    P: [3, 4] 投影矩阵 Tensor
    返回: [M, 5] Tensor, 每行是 [x, y, X, Y, Z]
    """
    N = points_3d.shape[0]
    
    # 添加齐次坐标列 [X, Y, Z, 1]
    ones = torch.ones((N, 1), dtype=points_3d.dtype, device=points_3d.device)
    points_homogeneous = torch.cat([points_3d, ones], dim=1)  # [N, 4]
    
    x,y = fullproj(points_homogeneous, P, 640, 480)
    
    
    # 筛选条件：x ∈ (0, 640), y ∈ (0, 480)
    mask = (x > 0) & (x < W) & (y > 0) & (y < H)
    
    print("mask shape = ", mask.shape)

    # 保留满足条件的点
    x_filtered = x[mask]
    y_filtered = y[mask]
    points_3d_filtered = points_3d[mask]  # [M, 3]

    # 拼接成 [M, 5]： [x, y, X, Y, Z]
    result = torch.cat([x_filtered.unsqueeze(1), 
                        y_filtered.unsqueeze(1), 
                        points_3d_filtered], dim=1)
    
    print("result shape = ", result.shape)
    
    return result  # [M, 5]
    


"""
Greedy algorithm to find the matching point
"""
def greedy_weighted_match(fixed_points, projected_points, alpha=1.0, beta=1.0, max_iter=10, device='cuda'):
    """
    固定点: (N, 2)
    投射点: (M, 2)
    返回: list of matched index pairs
    """
    fixed_points = fixed_points.to(device)
    projected_points = projected_points.to(device)

    N = fixed_points.shape[0]
    M = projected_points.shape[0]
    assert M >= N, "投射点数量必须不小于固定点数量"

    # 距离矩阵（垂直距离 + 欧氏距离）
    y_fixed = fixed_points[:, 1].unsqueeze(1)  # (N, 1)
    y_proj = projected_points[:, 1].unsqueeze(0)  # (1, M)
    vertical_dist = torch.abs(y_fixed - y_proj)  # (N, M)

    # 欧几里得距离矩阵
    fixed_exp = fixed_points.unsqueeze(1).expand(-1, M, -1)  # (N, M, 2)
    proj_exp = projected_points.unsqueeze(0).expand(N, -1, -1)  # (N, M, 2)
    euclidean_dist = torch.norm(fixed_exp - proj_exp, dim=2)  # (N, M)

    # 组合代价矩阵
    total_cost = alpha * vertical_dist + beta * euclidean_dist  # (N, M)
    print("total cost shape = ", total_cost.shape)

    # 贪心匹配初始化
    matched_indices = -torch.ones(N, dtype=torch.long, device=device)
    used_proj = torch.zeros(M, dtype=torch.bool, device=device)

    for i in range(100):
        dists = total_cost[i]
        print(i)
        sorted_indices = torch.argsort(dists)
        for idx in sorted_indices:
            if not used_proj[idx]:
                matched_indices[i] = idx
                used_proj[idx] = True
                break
    """
    # 匹配代价函数
    def compute_total_cost(matches):
        vertical = torch.abs(fixed_points[:, 1] - projected_points[matches, 1])
        euclid = torch.norm(fixed_points - projected_points[matches], dim=1)
        return (alpha * vertical + beta * euclid).sum()

    best_cost = compute_total_cost(matched_indices)

    # 局部交换优化（2-opt）
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for j in range(i + 1, N):
                new_matches = matched_indices.clone()
                new_matches[i], new_matches[j] = matched_indices[j], matched_indices[i]
                if len(torch.unique(new_matches)) < N:
                    continue
                new_cost = compute_total_cost(new_matches)
                if new_cost < best_cost:
                    matched_indices = new_matches
                    best_cost = new_cost
                    improved = True
        if not improved:
            break
    """
    return [(int(i), int(matched_indices[i].item())) for i in range(N)]


"""
Main function 

"""
def refiner(matched_2d, full_proj_matrix, feat_pcd):
    """
    return value:
         matched_2d : [N,2]
         updated_matched_3d : [N, 3]
    """
    
    # Project the point cloud to 2D pixel space and keep 
    # only the pixel that is inside the image
    pixel_pc =  project_and_filter(feat_pcd, full_proj_matrix, 640, 480)
    print("project point = ", pixel_pc[:,:2])
    
    # update the 3D position of each keypoint
    matches = greedy_weighted_match(torch.tensor(matched_2d), pixel_pc[:,:2], alpha=1.0, beta=0.5)
    
    mask_2d, mask_3d = torch.tensor(matches)[:,0], torch.tensor(matches)[:,1]
    
    return matched_2d[mask_2d], pixel_pc[mask_3d][:,2:].cpu().numpy()



"""
# 示例点 (可替换为实际数据)
fixed = torch.tensor([[10, 100], [30, 200], [50, 300]], dtype=torch.float).cuda()
projected = torch.tensor([[9, 110], [31, 190], [70, 250], [48, 310]], dtype=torch.float).cuda()

# 设置垂直距离和整体距离的权重
matches = greedy_weighted_match(fixed, projected, alpha=1.0, beta=0.5)
mask = torch.tensor(matches)[:, 1]



for i, j in matches:
    print(f"固定点 {i} ({fixed[i].tolist()}) -> 投射点 {j} ({projected[j].tolist()})") 

"""

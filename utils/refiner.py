
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
    

    # 保留满足条件的点
    x_filtered = x[mask]
    y_filtered = y[mask]
    points_3d_filtered = points_3d[mask]  # [M, 3]

    # 拼接成 [M, 5]： [x, y, X, Y, Z]
    result = torch.cat([x_filtered.unsqueeze(1), 
                        y_filtered.unsqueeze(1), 
                        points_3d_filtered], dim=1)   
    
    return mask, result  # [M, 5]
    


def get_updated_3d_indice(query_kp, render_kp,projected_points, device= 'cuda'):
    """
    query_kp : tensor [N, 2]
    render_kp: tensor [N, 2]
    """
    # Find the middle coordinate
    query_kp, render_kp = query_kp.to(device), render_kp.to(device)
    midpoints = (query_kp + render_kp)/2
    
    # Get the index for each midpoints
    # A: [N, 1, 2], B: [1, M, 2] => 广播得到 [N, M, 2]
    diff = midpoints[:, None, :] - projected_points[None, :, :]  # pairwise differences

    # 欧几里得距离（不取 sqrt 保持效率）
    dists = (diff ** 2).sum(dim=2)  # shape [N, M]

    # 每行最小值的索引，即 A 中每个点最近的 B 中的点的索引
    nearest_indices = torch.argmin(dists, dim=1)  # shape [N]
    
    return nearest_indices.cpu().numpy()
    
    
"""
Main function 

"""
def refiner(matched_2d, matched_3d, full_proj_matrix, feat_pcd):
    """
    return value:
         matched_2d : [N,2]
         updated_matched_3d : [N, 3]
    """
    # Project the matched 3d into 2D pixel space and keep
    # only the pixel that is inside the image
    mask, matched_3d_proj = project_and_filter(torch.tensor(matched_3d).cuda().float(), full_proj_matrix, 640, 480)
    mask = mask.cpu().numpy()

    
    # Project the point cloud to 2D pixel space and keep 
    # only the pixel that is inside the image
    _, pixel_pc =  project_and_filter(feat_pcd, full_proj_matrix, 640, 480)

    
    # update the 3D position of each keypoint
    matches = get_updated_3d_indice(torch.tensor(matched_2d[mask]), matched_3d_proj[:,:2], pixel_pc[:,:2])
    
    return matched_2d[mask], pixel_pc[:,2:][matches]    



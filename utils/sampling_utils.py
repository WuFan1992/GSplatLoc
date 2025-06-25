import torch


def sample_random_points(height=480, width=640, cell_size=8, device='cuda'):
    """
			Divide the feature map [480,640] into several cells, each cell contains 
            8x8 tensors. Randomly select a tensor in each cell 

			input:
				height : 480
                width : 640
                cell size : 8
                device: cuda or cpu
			return:
				coordinate [4800, 2]
    """
    cell_rows = height // cell_size  # 60
    cell_cols = width // cell_size   # 80
    total_cells = cell_rows * cell_cols  # 4800

    # Get the the top left tensor coordinate
    y_starts = torch.arange(0, height, cell_size, device=device).repeat_interleave(cell_cols)
    x_starts = torch.arange(0, width, cell_size, device=device).repeat(cell_rows)

    # In each cell, randomly offset from 0 to 7 
    offset = torch.randint(0, cell_size, size=(total_cells, 2), device=device)

    # Final coordinate = top left + offset 
    coords = torch.stack([x_starts, y_starts], dim=1) + offset  # shape: (4800, 2)

    return coords.int()  # keep the int type for coordinate


def sample_features(feature: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    # feature shape: (C, H, W)
    C, H, W = feature.shape
    N = coords.shape[0]
    

    x = coords[:, 0].long()
    y = coords[:, 1].long()
    
    x = x.clamp(0, W - 1)
    y = y.clamp(0, H - 1)
    
    flat_indices = y * W + x  # shape: (N,)
    feature_flat = feature.view(C, -1)  # shape: (64, 480*640)
    sampled = feature_flat[:, flat_indices]  # shape: (64, N)
    
    # Transpose(N, 64)
    return sampled.t()


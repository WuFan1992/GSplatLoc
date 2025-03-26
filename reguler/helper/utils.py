import torch 
import numpy as np
from typing import Callable
from torch.autograd import Function
import torch.nn.functional as F



# Get the mass center of a list of points 
# Points:  Tensor List ([[1,2,3], [14,25,36]...]) 
def calculate_mass_center(points):
    return torch.mean(points, dim=0)

def calculate_distance(point_a, point_b):
    """
    Calculate the Euclidean distance between two 3D points using PyTorch tensors.
    
    Parameters:
    point_a (torch.Tensor): A tensor of shape (3,) representing point A.
    point_b (torch.Tensor): A tensor of shape (3,) representing point B.
    
    Returns:
    torch.Tensor: The distance between point A and point B.
    """
    return torch.sqrt(torch.sum((point_b - point_a) ** 2))

def calculate_radius(point_a, list_of_points_b):
    """
    Find the maximum distance between point A and a list of points B using PyTorch.
    
    Parameters:
    point_a (torch.Tensor): A tensor of shape (3,) representing point A.
    list_of_points_b (torch.Tensor): A tensor of shape (N, 3), where each row represents a point B.
    
    Returns:
    torch.Tensor: The maximum distance between point A and any point in B.
    """
    distances = torch.norm(list_of_points_b - point_a, dim=1)
    return torch.max(distances)
    

def calculate_mass_density(center, points):

    radius = calculate_radius(center, points)
    
    # Compute the squared Euclidean distances between the points and the center
    distances = torch.norm(points - center, dim=1)
    
   # Count the number of points within the radius
    num_points_within_radius = torch.sum(distances <= radius).item()
    
    # Calculate the volume of a sphere with the given radius
    sphere_volume = (4/3) * np.pi * radius**3
    
    # Calculate the mass density as the number of points within the radius divided by the volume
    mass_density = num_points_within_radius / sphere_volume
    
    return mass_density

def get_points_from_ranges(range, gaussian_pcd):
    return gaussian_pcd[int(range[0]): int(range[1])]

def get_whole_points_from_ranges(ranges, gaussian_pcd):
    return [get_points_from_ranges(range, gaussian_pcd).detach() for range in ranges]
    
def get_whole_mass_center(all_points):
    return [calculate_mass_center(points) for points in all_points]

def get_whole_mass_density(centers, points):
    return [calculate_mass_density(centers[idx], points[idx]) for idx in range(len(centers))]  

def normalize_density(densities):
    den_max, den_min = torch.max(densities), torch.min(densities)
    range = den_max-den_min
    return [(den-den_min)/range for den in densities]  



class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))
trunc_exp = _TruncExp.apply

def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    elif name == "relu":
        return lambda x: torch.relu(x)
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")
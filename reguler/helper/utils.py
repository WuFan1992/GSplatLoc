import torch 
import numpy as np

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
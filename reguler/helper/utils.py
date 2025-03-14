import torch 

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
    # Calculate the area of the circle (π * radius^2)
    area = torch.pi * radius ** 2
    
    # Compute the squared Euclidean distances between the points and the center
    distances = torch.norm(points - center, dim=1)
    
    # Count the number of points within the given radius (distance <= radius)
    count_within_radius = torch.sum(distances <= radius).item()
    
    # Calculate density
    density = count_within_radius / area.item()
    return density
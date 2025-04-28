import torch
import numpy as np
import os
import torch.nn as nn
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p

from plyfile import PlyData, PlyElement
from .dataset_readers import fetchPly



class FeatPointCloud:
    
    def __init__(self):
        self._xyz = torch.empty(0).to("cuda")
        self.xyz_gradient_accum = torch.empty(0)
        self._semantic_feature = torch.empty(0).to("cuda") 
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_semantic_feature(self):
        return self._semantic_feature 
    
    def init_feat_pc(self, source_path, semantic_feature_size : int):
        """
        fetch the point cloud 
        """
        # Get the point cloud path
        pc_path = os.path.join(source_path,"sparse/0/points3D.ply")
        # Load the pcd
        pcd = fetchPly(pc_path)
        # set the initial xyz 
        self._xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # initialize the feature
        self._semantic_feature = torch.zeros(self._xyz.shape[0], semantic_feature_size, 1).float().cuda()
        
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._semantic_feature.shape[1]):  
            l.append('semantic_{}'.format(i))
        return l
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        semantic_feature = self._semantic_feature.detach().flatten(start_dim=1).contiguous().cpu().numpy() 
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, semantic_feature), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        self._xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        
        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
        semantic_feature = np.stack([np.asarray(plydata.elements[0][f"semantic_{i}"]) for i in range(count)], axis=1) 
        self._semantic_feature = np.expand_dims(semantic_feature, axis=-1) 

    
    def update_ply(self, kps: torch.Tensor, kp_feat: torch.Tensor):
        """
            For each matched 3D points, find its closest point in point cloud and update its feature 
            kp : 3D point associated with keypoint detected in query image [N, 3]
            kp_feat: 3D point feature [N, 64] 
        
        """
        kp_idx = 0 
        for kp in kps:
            distance = torch.norm(self._xyz - kp, dim=1)
            closest_index = torch.argmin(distance).item()
            self._semantic_feature[closest_index] = kp_feat[kp_idx].unsqueeze(-1)
            kp_idx = kp_idx + 1
            
            
    

    
    
    
    
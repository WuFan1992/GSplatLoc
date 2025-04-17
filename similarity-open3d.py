import torch


import cv2
import numpy as np

import torch




import matplotlib.pyplot as plt

from scene import Scene
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2, fov2focal

from encoders.XFeat.modules.xfeat import XFeat

from warping.warping_loss import *
from warping.warp_utils import *
from utils.loc_utils import *
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 




""""
This file is used to get disk feature of a query image and show its matching with another image Ref via xfeat feature.
The methods to get the Ref image can be achieved by NetVlad or mannuelly decided its index 

command: 
python similarity-open3d.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with disk feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

If we want to use the netvlad to do the image retrieval, we must launch the getdes.py. Make sure that in the netvlad.py, 
from netvlad.base_model import BaseModel must be 
from base_model import BaseModel

python getdes.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

Then after get the global descriptor, change the 
from base_model import BaseModel
back to  
from netvlad.base_model import BaseModel 
before runing the 2d_feature_disk_one.py


"""

import open3d 

import argparse
import collections
import os
import struct


"""

"""
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras"  + ext)) and \
       os.path.isfile(os.path.join(path, "images"   + ext)) and \
       os.path.isfile(os.path.join(path, "points3D" + ext)):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras
def read_points3D_binary(colors, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for idx in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            #rgb = np.array(binary_point_line_properties[4:7])
            rgb = colors[idx]
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D
def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D

def read_model(colors, path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        points3D = read_points3D_binary(colors, os.path.join(path, "points3D") + ext)
    return points3D

class Model:
    def __init__(self):
        self.points3D = []
        self.__vis = None
        self.xyz = []
        self.rgb = []

    def read_model(self, colors, path, ext=""):
        self.points3D = read_model(colors,path, ext)

    def add_points(self, min_track_len=3, remove_statistical_outlier=True):
        pcd = open3d.geometry.PointCloud()


        for point3D in self.points3D.values():
            track_len = len(point3D.point2D_idxs)
            if track_len < min_track_len:
                continue
            self.xyz.append(point3D.xyz)
            self.rgb.append(point3D.rgb / 255)
        pcd.points = open3d.utility.Vector3dVector(self.xyz)
        pcd.colors = open3d.utility.Vector3dVector(self.rgb)

        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()
        
    def add_keypoints(self, keypoints):
        for kp in keypoints:
            self.xyz.append(np.array(kp))
            self.rgb.append(np.array((255,0,0))/255)
        
    # Fan WU #######
    def add_keypoint(self, keypoints, rgb):
        for kp in keypoints:
            self.xyz.append(np.array(kp))
            self.rgb.append(np.array((rgb))/255)
        
        
    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()


def similarity(kp_feat,  gaussian_feat, chunk_size= 10000):
    P_N = gaussian_feat.shape[0]
    
    # Normalize features for faster cosine similarity computation
    kp_feat = F.normalize(kp_feat[None], p=2, dim=1)
    gaussian_feat = F.normalize(gaussian_feat, p=2, dim=1)
    
    similarity = torch.tensor([]).to("cuda")
    for part in range(0, P_N, chunk_size):
        chunk = gaussian_feat[part:part + chunk_size]
        # Use matrix multiplication for faster similarity computation
        similarity_temp = torch.mm(kp_feat, chunk.t())
        similarity = torch.cat([similarity, similarity_temp], 1)
    min_sim = torch.min(similarity)
    max_sim = torch.max(similarity)
    
    similarity_norm = (similarity-min_sim)/(max_sim-min_sim)
    return similarity_norm
    

def localize_set(model_path, name, views, gaussians, pipeline, background, args):


    # Set the point cloud path and format
    pointcloud_path = "./datasets/wholehead/sparse/0/"
    pointcloud_format = ".bin" 
    

    # Keep track of rotation and translation errors for calculation of the median error.

    gaussian_pcd = gaussians.get_xyz
    gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
    top_k = 50
    xfeat = XFeat(top_k=top_k)
    
    #Load image
    img_dir_query = "./datasets/wholehead/images/seq-01"
    query_img_name = "frame-000000.color.png"
    query_img_name_noext = "frame-000000"

    
    query_img_path = os.path.join(img_dir_query, query_img_name)
    query_img = cv2.imread(query_img_path) # [H,W,C] = [480,640,3]
    query_img_tensor = torch.tensor(query_img).permute(2,0,1).cuda() # [C,H,W]
    query_img_tensor = query_img_tensor / 255.0

      
    # Extract sparse features
    tensor_query_img = xfeat.parse_input(query_img) # [1,C,H,W] = [1,3,480,640]
    query_keypoints, _, query_feature = xfeat.detectAndCompute(tensor_query_img, 
                                                                 top_k=top_k)[0].values()  #query_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate
    

    #draw all the keypoint in red circle in image
    #for keypoint in query_keypoints:
   
    #Get the reference keypoint and its feature 
    index = 0
    ref_keypoint, ref_kp_feature = query_keypoints[index].to("cpu"), query_feature[index]
    print("keypoint = ", ref_keypoint)
    
    cv2.circle(query_img, (int(ref_keypoint[0].item()), int(ref_keypoint[1].item()) ), radius=10, color=(0, 0, 255), thickness=2)
    
    cv2.imshow('image', query_img)
    cv2.waitKey(0)
    #Get its similarity with all the 3DGS points
    sim = similarity(ref_kp_feature, gaussian_feat)
    
     #Initialize the open3d model 
    model = Model()
    
    
    cmap = plt.get_cmap('jet')
    colors = cmap(sim.to("cpu").detach().numpy()).squeeze()[:,:3]
    
    
    model.read_model(255*colors, pointcloud_path, pointcloud_format)
    
     # display using Open3D visualization tools
    model.create_window()
    
    model.add_points()
    model.show()
    
    
   

def launch_inference(dataset : ModelParams, pipeline : PipelineParams, args): 
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1]*64 if dataset.white_background else [0]*64
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    localize_set(dataset.model_path, "test", scene.getTrainCameras(), gaussians, pipeline, background, args)


if __name__ == "__main__":
# Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--top_k", default=1_000, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    parser.add_argument("--warp_lr", default=0.0005, type=float)
    parser.add_argument("--warp_iters", default=251, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    launch_inference(model.extract(args), pipeline.extract(args), args)












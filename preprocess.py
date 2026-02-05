# import some common libraries
import sys

sys.path.append(".")
sys.path.append("submodules/Mask2Former")

import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

setup_logger()
setup_logger(name="mask2former")
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")


from mask2former import add_maskformer2_config
"""
Preprocess only used for cambridge dataset
1. Use colmap_from_nvm to convert nvm to  sfm point cloud bin (the same format used by 7 scene)
2. [ONLY IN UBUNTU ENVIRONMENT !!!!!!!] If you use Windows, download the WSL 2, and install conda and all the necessary envrionment for mask2former
   Following the proces in https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md
   
   When finish "sh make.sh"  we need to add mask2former to the python environment because mask2former is not a lib but in this file 
   we use it as a lib 
   export PYTHONPATH=$PYTHONPATH:/path/to/Mask2Former
   
3. run this file in UBUNTU console :
     python -m preprocess --source_path datasets/cambridge/OldHospital --output_folder processed
     
    Use -m to run it will make it as a lib that avoid all the relative dependences problem  

"""

import numpy as np
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
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
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

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

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors


def read_points3D_nvm(nvm_file, threshold=1000):
    """
    Formats of nvm file:
        <Number of cameras>   <List of cameras>
        <Number of 3D points> <List of points>
        <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
        <Measurement> = <Image index> <Feature Index> <xy>
    """
    cams = []       # List image frames 
    cam_points = {} # Map key: index of frame, value: list of indices of 3d points that are visible to this frame.
    points = []     # List of 3d points in the reconstruction model
    rgb = []

    print('Read 3D points from {}'.format(nvm_file))
    with open(nvm_file, 'r') as f:
        next(f)    # Skip headding lines
        next(f)
        
        # Load images
        cam_num = int(next(f).split()[0])
        for i in range(cam_num):
            line = next(f) 
            frame = line.split()[0]
            cams.append(frame)
            cam_points[frame] = []
            
        next(f)  # Skip the separation line
        point_num = int(next(f).split()[0])
        for i in range(point_num):
            line = next(f)
            cur = line.split()
            points.append([float(x) for x in cur[0:3]])
            rgb.append([int(x) for x in cur[3:6]])
            measure_num = int(cur[6])
            for j in range(measure_num):
                idx = int(cur[7+j*4])
                frame = cams[idx]
                cam_points[frame].append(i)
    print('Loading finished: camera frames {}, total 3d points {}'.format(len(cam_points), len(points)))

    filtered_points = []
    filtered_colors = []
    for point, color in zip(points, rgb):
        if all(abs(coord) <= threshold for coord in point):
            filtered_points.append(point)
            filtered_colors.append(color)
    
    return np.stack(filtered_points), np.stack(filtered_colors)



def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def get_resolution_from_longest_edge(height, width, longest_edge=640):
    if height > width:
        scale = longest_edge / height
        new_height = longest_edge
        new_width = int(width * scale)
    else:
        scale = longest_edge / width
        new_width = longest_edge
        new_height = int(height * scale)
    return new_height, new_width




def hist_equalize(image):
    r, g, b = cv2.split(image)
    #
    clahe_b = hist_equalizer.apply(b)
    clahe_g = hist_equalizer.apply(g)
    clahe_r = hist_equalizer.apply(r)

    # merge
    clahe_image_rgb = cv2.merge((clahe_r, clahe_g, clahe_b))
    return clahe_image_rgb


class stuff_masker(torch.nn.Module):
    def __init__(self):
        super(stuff_masker, self).__init__()
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(
            "submodules/Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
        )
        cfg.MODEL.WEIGHTS = "submodules/Mask2Former/model_final_f07440.pkl"
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        cfg.freeze()

        predictor = DefaultPredictor(cfg)
        self.predictor = predictor

    def get_stuff_mask(self, image):
        # BGR
        outputs = self.predictor(image)
        stuff_mask = torch.zeros_like(outputs["panoptic_seg"][0])
        for info in outputs["panoptic_seg"][1]:
            if info["isthing"] is False:
                stuff_mask[outputs["panoptic_seg"][0] == info["id"]] = 1
        return stuff_mask

    def get_stuff_and_sky_mask(self, image):
        # BGR
        outputs = self.predictor(image)
        stuff_mask = torch.ones_like(outputs["panoptic_seg"][0], dtype=torch.bool)
        sky_mask = torch.ones_like(outputs["panoptic_seg"][0], dtype=torch.bool)
        for info in outputs["panoptic_seg"][1]:
            if info["isthing"]:
                stuff_mask[outputs["panoptic_seg"][0] == info["id"]] = False
            # mask sky
            if info["category_id"] == 119:
                sky_mask[outputs["panoptic_seg"][0] == info["id"]] = False
        return stuff_mask, sky_mask

    def forward(self, image):
        return self.get_stuff_and_sky_mask(image)


def undistort(distorted_image, camera_matrix, distortion_coeffs):
    # read
    if distorted_image is None:
        raise ValueError("distorted_image is None")
    # undistort
    undistorted_image = cv2.undistort(distorted_image, camera_matrix, distortion_coeffs)
    return undistorted_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="")
    parser.add_argument("--images", type=str, default="")
    parser.add_argument("--longest_edge", type=int, default=640)
    parser.add_argument("--output_folder", type=str, default="processed")

    args = parser.parse_args()

    colmap_path = os.path.join(args.source_path, "sparse", "0")
    extrinsics = read_extrinsics_binary(os.path.join(colmap_path, "images.bin"))

    output_path = os.path.join(args.source_path, args.output_folder)
    os.makedirs(output_path, exist_ok=True)

    try:
        cameras_extrinsic_file = os.path.join(colmap_path, "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(colmap_path, "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    masker = stuff_masker()
    hist_equalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    masks = {}

    for key in tqdm(cam_extrinsics, desc="Prepocessing"):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        image_name = extr.name
        image_output_path = os.path.join(
            args.source_path, args.output_folder, image_name
        )
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)

        if intr.model == "SIMPLE_RADIAL":
            camera_matrix = np.array(
                [
                    [intr.params[0], 0, intr.params[1]],
                    [0, intr.params[0], intr.params[2]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )  # focal length and principal point
            distortion_coeffs = np.array([intr.params[3], 0, 0, 0], dtype=np.float32)

        # read image
        image_path = os.path.join(args.source_path, args.images, image_name)
        distorted_image = cv2.imread(image_path)

        # CLAHE hist equalization
        hist_equalized_image = hist_equalize(distorted_image)

        # undistort
        undistorted_image = undistort(
            hist_equalized_image, camera_matrix, distortion_coeffs
        )

        # save processed image
        cv2.imwrite(image_output_path, undistorted_image)

        # generate masks
        image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1)

        undistort_mask = torch.max(image, dim=0, keepdim=True)[0] > 0

        mask_resolution = get_resolution_from_longest_edge(
            image.shape[1], image.shape[2], args.longest_edge
        )

        image = F.interpolate(
            image.unsqueeze(dim=0).float(),
            size=mask_resolution,
            mode="bilinear",
            align_corners=False,
        ).squeeze(dim=0)
        undistort_mask = (
            F.interpolate(
                undistort_mask.unsqueeze(dim=0).float(),
                size=mask_resolution,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            > 0.5
        )
        stuff_mask, sky_mask = masker(image.permute(1, 2, 0).numpy()[:, :, ::-1])
        mask = (stuff_mask, sky_mask, undistort_mask)
        masks[image_name] = mask

    pickle.dump(masks, open(os.path.join(output_path, "masks.pkl"), "wb"))
    print("Masks saved to", os.path.join(output_path, "masks.pkl"))

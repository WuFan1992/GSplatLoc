import torch, h5py, imageio, os, argparse
import numpy as np
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_dimcheck import dimchecked

import cv2
import matplotlib.pyplot as plt

from disk import DISK, Features



import warnings, imageio

from torch_dimcheck import dimchecked

from disk.geom import distance_matrix

MAX_FULL_MATRIX = 10000**2


@dimchecked
def _binary_to_index(binary_mask: ['N'], ix2: ['M']) -> [2, 'M']:
    return torch.stack([
        torch.nonzero(binary_mask, as_tuple=False)[:, 0],
        ix2
    ], dim=0)

@dimchecked
def _ratio_one_way(dist_m: ['N', 'M'], rt) -> [2, 'K']:
    val, ix = torch.topk(dist_m, k=2, dim=1, largest=False)
    ratio = val[:, 0] / val[:, 1]
    passed_test = ratio < rt
    ix2 = ix[passed_test, 0]

    return _binary_to_index(passed_test, ix2)

@dimchecked
def _match_chunkwise(ds1: ['N', 'F'], ds2: ['M', 'F'], rt) -> [2, 'K']:
    chunk_size = MAX_FULL_MATRIX // ds1.shape[0]
    matches = []
    start = 0

    while start < ds2.shape[0]:
        ds2_chunk = ds2[start:start+chunk_size]
        dist_m = distance_matrix(ds1, ds2_chunk)
        one_way = _ratio_one_way(dist_m, rt)
        one_way[1] += start
        matches.append(one_way)
        start += chunk_size

    return torch.cat(matches, dim=1)
    
@dimchecked
def _match(ds1: ['N', 'F'], ds2: ['M', 'F'], rt) -> [2, 'K']:
    size = ds1.shape[0] * ds2.shape[0]

    fwd = _match_chunkwise(ds1, ds2, rt)
    bck = _match_chunkwise(ds2, ds1, rt)
    bck = torch.flip(bck, (0, ))

    merged = torch.cat([fwd, bck], dim=1)
    unique, counts = torch.unique(merged, dim=1, return_counts=True)

    return unique[:, counts == 2]

def match(desc_1, desc_2, rt=1., u16=False):
    matched_pairs = _match(desc_1, desc_2, rt)
    matches = matched_pairs.cpu().numpy()

    if u16:
        matches = matches.astype(np.uint16)

    return matches

def brute_match(descriptors, hdf):
    keys = sorted(list(descriptors.keys()))

    n_total = (len(keys) * (len(keys) - 1)) // 2
    saved = 0
    pbar = tqdm(total=n_total)

    for i, key_1 in enumerate(keys):
        desc_1 = descriptors[key_1].to(DEV)
        group  = hdf.require_group(key_1)
        for key_2 in keys[i+1:]:
            if key_2 in group.keys():
                continue

            desc_2 = descriptors[key_2].to(DEV)
            
            try:
                matches = match(desc_1, desc_2, rt=args.rt, u16=args.u16)
                n = matches.shape[1]

                if n >= args.save_threshold:
                    group.create_dataset(key_2, data=matches)
                    saved += 1
            except RuntimeError:
                print('Error, skipping...')
                n = 0

            pbar.update(1)
            pbar.set_postfix(left=str(key_1), s=saved, n=n)

    pbar.close()
    

class MatcherWrapper:
    class InnerWrapper:
        def __init__(self):
            if args.rt is None:
                self._cycle_matcher = CycleMatcher()
            else:
                self._cycle_matcher = CycleRatioMatcher(args.rt)

        @dimchecked
        def raw_mle_match_pair(self, ds1: ['N', 'F'], ds2: ['M', 'F']) -> [2, 'K']:
            dist = distance_matrix(ds1, ds2, normalized=True)
            return self._cycle_matcher(dist)

    def __init__(self):
        self.matcher = MatcherWrapper.InnerWrapper()

class Image:
    def __init__(self, bitmap: ['C', 'H', 'W'], fname: str, orig_shape=None):
        self.bitmap     = bitmap
        self.fname      = fname
        if orig_shape is None:
            self.orig_shape = self.bitmap.shape[1:]
        else:
            self.orig_shape = orig_shape

    def resize_to(self, shape):
        return Image(
            self._pad(self._interpolate(self.bitmap, shape), shape),
            self.fname,
            orig_shape=self.bitmap.shape[1:],
        )

    @dimchecked
    def to_image_coord(self, xys: [2, 'N']) -> ([2, 'N'], ['N']):
        f, _size = self._compute_interpolation_size(self.bitmap.shape[1:])
        scaled = xys / f

        h, w = self.orig_shape
        x, y = scaled

        mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)

        return scaled, mask

    def _compute_interpolation_size(self, shape):
        x_factor = self.orig_shape[0] / shape[0]
        y_factor = self.orig_shape[1] / shape[1]

        f = 1 / max(x_factor, y_factor)

        if x_factor > y_factor:
            new_size = (shape[0], int(f * self.orig_shape[1]))
        else:
            new_size = (int(f * self.orig_shape[0]), shape[1])

        return f, new_size

    @dimchecked
    def _interpolate(self, image: ['C', 'H', 'W'], shape) -> ['C', 'h', 'w']:
        _f, size = self._compute_interpolation_size(shape)
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    
    @dimchecked
    def _pad(self, image: ['C', 'H', 'W'], shape) -> ['C', 'h', 'w']:
        x_pad = shape[0] - image.shape[1]
        y_pad = shape[1] - image.shape[2]

        if x_pad < 0 or y_pad < 0:
            raise ValueError("Attempting to pad by negative value")

        return F.pad(image, (0, y_pad, 0, x_pad))
    

    
def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    
    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)
    
    #clean the keypoint with mask
    ref_points_valid = []
    dst_points_valid = []
    for i in range(len(mask)):
        if mask[i]:
            ref_points_valid.append(ref_points[i])
            dst_points_valid.append(dst_points[i])

    return  img_matches,ref_points_valid, dst_points_valid

    
def disk_match(feats1, feats2, min_cossim = 0.82):
    
    cossim = feats1 @ feats2.t()
    cossim_t = feats2 @ feats1.t()	
    _, match12 = cossim.max(dim=1)
    _, match21 = cossim_t.max(dim=1)
    idx0 = torch.arange(len(match12), device=match12.device)
    mutual = match21[match12] == idx0
    
    if min_cossim > 0:
        cossim, _ = cossim.max(dim=1)
        good = cossim > min_cossim
        idx0 = idx0[mutual & good]
        idx1 = match12[mutual & good]
    else:
        idx0 = idx0[mutual]
        idx1 = match12[mutual]
    
   
    return idx0, idx1
    

def getKpDesc(model, dir, img_name):
    img_path = os.path.join(dir, img_name)
    img = cv2.imread(img_path) # [H,W,C] = [480,640,3]
    img_tensor = torch.tensor(img).permute(2,0,1).cuda() # [C,H,W]
    img_tensor = img_tensor / 255.0
    
    image_query = Image(img_tensor, img_name.split('.')[0])
    
    query_features = model.features(image_query.bitmap[None])

    
    kps_crop_space = query_features[0].kp.T
    kps_img_space, mask = image_query.to_image_coord(kps_crop_space)
    kps_img_space, mask = kps_img_space.cpu(), mask.cpu()
    
    keypoints   = kps_img_space.numpy().T[mask]
    descriptors = query_features[0].desc.detach().cpu().numpy()[mask]
    #scores      = query_features[0].kp_logp.detach().cpu().numpy()[mask]
    return keypoints, descriptors, img

if __name__ == '__main__':
    
    default_model_path = os.path.split(os.path.abspath(__file__))[0] + '/depth-save.pth'
    state_dict = torch.load(default_model_path, map_location='cpu')
    DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CPU   = torch.device('cpu')
    # compatibility with older model saves which used the 'extractor' name
    if 'extractor' in state_dict:
        weights = state_dict['extractor']
    elif 'disk' in state_dict:
        weights = state_dict['disk']
    else:
        raise KeyError('Incompatible weight file!')
    model = DISK(window=8, desc_dim=128)
    model.load_state_dict(weights)
    model = model.to(DEV)
    
    
    dir = "../../datasets/wholehead/images/seq-01"
    query_img_name = "frame-000005.color.png"
    ref_img_name = "frame-000055.color.png"
    
    kp_query, desc_query, query_img = getKpDesc(model, dir, query_img_name)
    kp_ref, desc_ref, ref_img = getKpDesc(model, dir, ref_img_name)
    
    #matches = match(torch.tensor(desc_query), torch.tensor(desc_ref))
    #n = matches.shape[1]

    idxs0, idxs1 = disk_match(torch.tensor(desc_query), torch.tensor(desc_ref), min_cossim=0.82)
    mkpts_0, mkpts_1 = kp_query[idxs0], kp_ref[idxs1]

    canvas, query_points_valid, ref_points_valid = warp_corners_and_draw_matches(mkpts_0, mkpts_1, query_img, ref_img)
    
    print("number of matching point : ", len(ref_points_valid))
    plt.figure(figsize=(12,12))
    plt.imshow(canvas[..., ::-1]), plt.show()
    
    
    
    
    

    
    
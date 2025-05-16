# from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from torchvision import transforms

import argparse
import os

def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)
    
        
class SuperPointModel(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 256
        
        self.transform = transforms.Grayscale(num_output_channels=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, out_channels,
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)), strict=False)


    def forward(self, x):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.transform(x)
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)                                   # [1,65,60,80]
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]     # [1,64,60,80]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8) # [1,480,640]
        

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1) # [1,256,60,80]
        
        return descriptors, scores

class SuperPoint(nn.Module):
    def __init__(self, top_k, detection_threshold,nms_dist ):
        super(SuperPoint, self).__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = SuperPointModel().to(self.dev).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold
        self.nms_dist = nms_dist
        self.border_remove = 4
    
    def getPtsFromHeatmap(self, heatmap):
        '''
        :param self:
        :param heatmap:
            np (H, W)
        :return:
        '''
        heatmap = heatmap.squeeze()
        # print("heatmap sq:", heatmap.shape)
        H, W = heatmap.shape[0], heatmap.shape[1]
        xs, ys = np.where(heatmap >= self.detection_threshold)  # Confidence threshold.
        self.sparsemap = (heatmap >= self.detection_threshold)
        if len(xs) == 0:
            return np.zeros((3, 0))
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys # abuse of ys, xs
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]  # check the (x, y) here
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        return pts
    
    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds
    
    def heatmap_to_pts(self, heatmap):

        pts_nms_batch = [self.getPtsFromHeatmap(h) for h in heatmap] # [batch, H, W]
        self.pts_nms_batch = pts_nms_batch
        return pts_nms_batch
    
    def preprocess_tensor(self, x):
        """
        Guarantee that image is divisible by 32 to avoid aliasing artifacts.
        """
        if isinstance(x, np.ndarray) and len(x.shape) == 3:
            x = torch.tensor(x).permute(2,0,1)[None]
        x = x.to(self.dev).float()
        
        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W
        
        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw
    
    
    def detectAndCompute(self, x, top_k = None, detection_threshold = None):
        """
			Compute sparse keypoints & descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return:
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
					'scores'       ->   torch.Tensor(N,): keypoint scores
					'descriptors'  ->   torch.Tensor(N, 64): local features
        """
        
        if top_k is None: top_k = self.top_k
        x, rh1, rw1 = self.preprocess_tensor(x)
        
        B, _, _H1, _W1 = x.shape
        #M1, K1, H1 = self.net(x)
        Des, Score  = self.net(x) # Score : [1,480,640], Des : [1,256,60,80]
        Des = F.normalize(Des, dim=1)
        Des = F.interpolate (Des, scale_factor=8, mode='bicubic', align_corners=True)
        Score = F.normalize(Score, dim=1)

       
		#Convert logits to heatmap and extract kpts
        #K1h = torch.unsqueeze(Score, 1) # torch.Size([1, 1, 480, 640])
        heatmap_np = toNumpy(Score)
        
        pts_nms_batch = self.heatmap_to_pts(heatmap_np)
        keypoints = torch.from_numpy(pts_nms_batch[0])[:2,:top_k].to(dtype=torch.int) #[2, top_k]
        
        descriptors = [torch.permute(des[:,keypoints[1], keypoints[0]], (1,0)) for des in Des]
        keypoints = torch.permute(keypoints, (1,0))
        
        return  {
                     'keypoints': keypoints,
                      'descriptor': descriptors,
                      'feature_map': Des[0]       
                }
        
        
        
        
        """

        #mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5) # [B,N,2] N is the key point after NMS
        
		#Interpolate descriptors at kpts positions
        feats = self.interpolator(Des, mkpts, H = _H1, W = _W1)

		#L2-Normalize
        feats = F.normalize(feats, dim=-1)

		#Correct kpt scale
        mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1)
        
        valid = scores > 0
        return [  
				   {'keypoints': mkpts[b][valid[b]],
					'scores': scores[b][valid[b]],
					'descriptors': feats[b][valid[b]]} for b in range(B) 
			   ]
      """
        return
    
    
    

parser = argparse.ArgumentParser(
    description=(
        "Get image embeddings of an input image or directory of images."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where embeddings will be saved. Output will be either a folder "
        "of .pt per image or a single .pt representing image embeddings."
    ),
)


parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


def main(args: argparse.Namespace) -> None:
    print("Loading model...")

    model = SuperPoint().cuda()

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        print(os.listdir(args.input))
        seqs = [f for f in os.listdir(args.input) if "seq" in f and "zip" not in f]
    
    print(seqs)
    os.makedirs(args.output, exist_ok=True)


    for seq in seqs:
        targets = [
            f"{seq}/{f}" for f in os.listdir(os.path.join(args.input, seq)) if "color" in f
        ]
        targets = [os.path.join(args.input, f) for f in targets]

        output_dir = os.path.join(args.output, seq)
        os.makedirs(output_dir, exist_ok=True)

        for t in targets:
            print(f"Processing '{t}'...")
            img_name = t.split(os.sep)[-1]
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            
            tensor_image = torch.from_numpy(np.array(image))
            input_image = tensor_image.to(
                device="cuda", dtype=torch.float32, non_blocking=True
            )
            input_image = input_image / 255.0
            img_features, scores = model(input_image.permute(2, 0, 1)[None])
            # print(scores.shape)

            img_features = img_features.squeeze(0) # (256, 60, 80)
            img_scores = scores[0] # (60, 80)

            torch.save(img_features, os.path.join(output_dir, f"{img_name}_fmap_CxHxW.pt"))
            torch.save(img_scores, os.path.join(output_dir, f"{img_name}_smap_CxHxW.pt"))
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


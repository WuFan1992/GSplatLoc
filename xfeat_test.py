from encoders.XFeat.modules.xfeat import XFeat
import numpy as np
import os
import torch
from torchvision.transforms import PILToTensor
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import collections as mplcollections
from matplotlib import colors as mcolors

import cv2
import numpy as np

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
    '''
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners
    '''

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]
    print("keypoint1 len = ", len(keypoints1))
    print("matches number = ", len(matches))
    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches



class MultiFigure:
    def __init__(
        self,
        image1: ['C', 'H', 'W'],
        image2: ['C', 'H', 'W'],
        grid=None,
        vertical=False,
    ):
        assert image1.shape == image2.shape
        c, h, w = image1.shape
        
        cat_dim = 1 if vertical else 2
        images = torch.cat([image1, image2], dim=cat_dim)
        images =  torch.permute(images,(1,2,0)).int()


        figsize = (20,40) if vertical else (40,20)
        self.fig, self._ax = plt.subplots(
            figsize=figsize,  # unit is inch 
            frameon=False,    # the axis contributs, if the axis frame is shown or not 
            constrained_layout=True  #layout='constrained'  optimize so that no overlapping in axis    
        )
        self._ax.imshow(images)
        xmax = w
        ymax = h
        if vertical:
            ymax *= 2
        else:
            xmax *= 2

        self._ax.set_xlim(0, xmax)
        self._ax.set_ylim(ymax, 0)

        if grid is None:
            self._ax.axis('off')
        else:
            self._ax.set_xticks(np.arange(0, xmax, grid))
            self._ax.set_yticks(np.arange(0, ymax, grid))
            self._ax.grid()

        if vertical:
            self.offset = torch.tensor([0, h])
        else:
            self.offset = torch.tensor([w, 0])

    def mark_xy(
        self,
        xy1: ['N', 2],
        xy2: ['N', 2],
        color='green',
        lines=True,
        marks=True,
        plot_n=None,
        linewidth=None,
        marker_size=None,
    ):
        xy2 = xy2 + self.offset

        xys = torch.stack([xy1, xy2], dim=1)

        if plot_n is not None:
            if xys.shape[0] > plot_n:
                ixs = torch.linspace(0, xys.shape[0]-1, plot_n).to(torch.int64)
                xys = xys[ixs, :]

        if lines:
            if color is not None:
                # LineCollection requires an rgb tuple
                color = mcolors.to_rgb(color)

            # yx convention
            plot = mplcollections.LineCollection(
                xys.numpy(),
                color=color,
                linewidth=linewidth
            )
            self._ax.add_collection(plot)
        else:
            plot = None

        if marks:
            self._ax.scatter(
                xys[:, :, 0].numpy().flatten(),
                xys[:, :, 1].numpy().flatten(),
                marker='o',
                c='white',
                edgecolor='black',
                s=marker_size,
            )
    
        return plot



# initialize the model 
xfeat = XFeat(top_k=4096)

#Load image
img_dir = "./datasets/wholehead/images/seq-01"
img_name_1 = "frame-000005.color.png"
img_name_2 = "frame-000020.color.png"


image_path_1 = os.path.join(img_dir, img_name_1)
image_path_2 = os.path.join(img_dir, img_name_2)

im1, im2 = cv2.imread(image_path_1), cv2.imread(image_path_2)
mkpts_0, mkpts_1 = xfeat.match_xfeat(im1, im2, top_k = 4096)
canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
plt.figure(figsize=(12,12))
plt.imshow(canvas[..., ::-1]), plt.show()



    
        






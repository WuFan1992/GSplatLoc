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
xfeat = XFeat(top_k=)

#Load image
img_dir = "./datasets/images/"
img_name_1 = "frame-000000.color.png"
img_name_2 = "frame-000050.color.png"

image_path_1 = os.path.join(img_dir, img_name_1)
image_path_2 = os.path.join(img_dir, img_name_2)

image_1 = Image.open(image_path_1)   # size = (640, 480)
image_2 = Image.open(image_path_2)


tensor_image_1 = PILToTensor()(image_1).float()
tensor_image_2 = PILToTensor()(image_2).float()

mkpts_0, mkpts_1 = xfeat.match_xfeat(tensor_image_1, tensor_image_2)
mkpts_0 = torch.from_numpy(mkpts_0)
mkpts_1 = torch.from_numpy(mkpts_1)

print("mkpts_0 : ", mkpts_0)
print("mkpts_1 : ", mkpts_1)

fig = MultiFigure(tensor_image_1, tensor_image_2)
fig.mark_xy(mkpts_0, mkpts_1)
plt.show()



    
        






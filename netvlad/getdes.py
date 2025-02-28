import torch
import torch.nn as nn
from torch.autograd import Variable

from netvlad import NetVLAD
from netvlad import EmbedNet
from hard_triplet_loss import HardTripletLoss
from torchvision.models import resnet18
from torchvision.transforms import PILToTensor

import cv2
import numpy as np
import os

import sys
sys.path.insert(0, 'C:/Users/fwu/Documents/PhD_FanWU/MyCode/GSplatLoc/gsplatloc-main')

from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams,PipelineParams, get_combined_args

###  Command #############
# python getdes.py -s ../datasets/wholehead/ -m ../output/ --iteration 15000 
########################################################################



def getNetVladDesc(views, model):   # --->  global descriptor [N, 128] and image name:  [N, 1]
    
    imgs_name = []
    global_desc = []
    
    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        #Get the image
        img_name = view.seq_num + '/' + view.image_name  # ex: seq-1/frame-000455
        image = view.original_image[0:3, :, :][None]
        output = model(image)
        global_desc.append(output.detach().cpu())
        imgs_name.append(img_name)
        
    return global_desc, imgs_name


# Discard layers at the end of base network
encoder = resnet18(pretrained=True)
base_model = nn.Sequential(
    encoder.conv1,
    encoder.bn1,
    encoder.relu,
    encoder.maxpool,
    encoder.layer1,
    encoder.layer2,
    encoder.layer3,
    encoder.layer4,
)
dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)

# Define model for embedding
net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)
model = EmbedNet(base_model, net_vlad).cuda()

# Define loss
criterion = HardTripletLoss(margin=0.1).cuda()

#Prepare the dataset 
parser = ArgumentParser(description="Testing script parameters")
model_params = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--iteration", default=-1, type=int)
args = get_combined_args(parser)
gaussians = GaussianModel(3)
scene = Scene(model_params.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)

global_desc, imgs_name = getNetVladDesc(scene.getTrainCameras(), model)
desc_names_tensor = [torch.stack(global_desc), imgs_name]
torch.save(desc_names_tensor, 'global_desc.pt')




"""
#Get the image 
img_dir_query = "../datasets/wholehead/images/seq-01"   
query_img_name = "frame-000969.color.png"
query_img_path = os.path.join(img_dir_query, query_img_name)
query_img = cv2.imread(query_img_path) # [H,W,C] = [480,640,3]
query_img_tensor = torch.tensor(query_img).permute(2,0,1)[None].cuda() # [C,H,W]

# This is just toy example. Typically, the number of samples in each classes are 4.
#labels = torch.randint(0, 10, (40, )).long()
#x = torch.rand(40, 3, 128, 128).cuda()
output = model(query_img_tensor.to(torch.float))

#triplet_loss = criterion(output, labels)
"""
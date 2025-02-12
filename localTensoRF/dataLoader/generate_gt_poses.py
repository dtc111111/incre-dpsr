import os
import random
import glob 
import logging
from PIL import Image
#from models.geometry.wrappers import Camera
import numpy as np
import math
import torch
import cv2
import re
import collections
import struct
from common import _load_data, recenter_poses, spherify_poses, load_depths_npz, load_gt_depths, poses_avg, normalize
from torchvision import transforms
from joblib import delayed, Parallel
from torch.utils import data
#from torch.utils.data import Dataset
#from utils.utils import decode_flow
import json
import sys
from aa import *

def get_gt_poses(datadir, spherify=True):
    poses, bds, _, _, _, _, _ = _load_data(datadir, factor=None, crop_size=0, load_colmap_poses=True)
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    bd_factor = 0.75
    # Rescale if bd_factor is provided
    
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    poses = recenter_poses(poses)
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)
    input_poses = poses.astype(np.float32)
    hwf = input_poses[0,:3,-1]
    input_poses = input_poses[:,:3,:4]
    H, W, focal = hwf
    H, W = int(H), int(W)
    poses_tensor = torch.from_numpy(input_poses)
    bottom = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
    bottom = bottom.repeat(poses_tensor.shape[0], 1, 1)
    c2ws_colmap = torch.cat([poses_tensor, bottom], 1)
    return c2ws_colmap

if __name__ == "__main__":
    data_dir = "/dataset/Ward/ward3"
    gt_poses = get_gt_poses(datadir=data_dir)
    torch.set_printoptions(threshold=sys.maxsize)
    print('GT poses: {}'.format(gt_poses))
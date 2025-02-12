# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
import random
import glob 
import logging
from PIL import Image
from models.geometry.wrappers import Camera
import numpy as np
import math
import torch
import cv2
import re
import collections
import struct
from dataLoader.common import _load_data, recenter_poses, spherify_poses, load_depths_npz, load_gt_depths, poses_avg, normalize
from torchvision import transforms
from joblib import delayed, Parallel
from torch.utils import data
#from torch.utils.data import Dataset
from utils.utils import decode_flow
import json

from dataLoader.aa import * #for the obtaining and calculation of fov; for reading image to learn pose——wcd 20230720
logger = logging.getLogger(__name__)
####################################################################################################################
############################# get pose configs & data dir——wcd 20230801 ############################################
####################################################################################################################
def get_dataloader(cfg, mode='train',
                   shuffle=True, n_views=None):
    """Return dataloader instance

    Instansiate dataset class and dataloader and 
    return dataloader
    
    Args:
        cfg (dict): imported config for dataloading
        mode (str): tran/eval/render/all
        shuffle (bool): as name
        n_views (int): specify number of views during rendering
    """
    ############################################################
        
    batch_size = cfg['dataloading']['batchsize']
    n_workers = cfg['dataloading']['n_workers']
   
    fields = get_data_fields(cfg, mode)
    if n_views is not None and mode=='render':
        n_views = n_views
    else:
        n_views = fields['img'].N_imgs
    ## get dataset
    dataset = OurDataset(
         fields, n_views=n_views, mode=mode)

    ## dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=n_workers, 
        shuffle=shuffle, pin_memory=True
    )

    return dataloader, fields


def get_data_fields(cfg, mode='train'):
    """Returns the data fields.

    Args:
        cfg (dict): imported yaml config
        mode (str): the mode which is used

    Return:
        field (dict): datafield
    """
    ############################################
    use_DPT = (cfg['depth']['type']=='DPT')
    resize_img_transform = ResizeImage_mvs() # for dpt input images
    fields = {}
    load_ref_img = ((cfg['training']['pc_weight']!=0.0) or (cfg['training']['rgb_s_weight']!=0.0))
    dataset_name = cfg['dataloading']['dataset_name']
    if dataset_name=='any':
        img_field = pose_DataField( 
                model_path=cfg['dataloading']['path'],
                transform=resize_img_transform,
                with_camera=True,
                with_depth=cfg['dataloading']['with_depth'],
                scene_name=cfg['dataloading']['scene'],
                use_DPT=use_DPT, mode=mode,spherify=cfg['dataloading']['spherify'], 
                load_ref_img=load_ref_img, customized_poses=cfg['dataloading']['customized_poses'],
                customized_focal=cfg['dataloading']['customized_focal'],
                resize_factor=cfg['dataloading']['resize_factor'], depth_net=cfg['dataloading']['depth_net'], 
                crop_size=cfg['dataloading']['crop_size'], random_ref=cfg['dataloading']['random_ref'], norm_depth=cfg['dataloading']['norm_depth'],
                load_colmap_poses=cfg['dataloading']['load_colmap_poses'], sample_rate=cfg['dataloading']['sample_rate'])
    else:
        print(dataset_name, 'does not exist')
    fields['img'] = img_field
    return fields
class ResizeImage_mvs(object):
    def __init__(self):
        net_w = net_h = 384
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = transforms.Compose(
                [
                    Resize(
                        net_w,
                        net_h,
                        resize_target=True,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal"
                    ),
                    normalization,
                    PrepareForNet(),
                ]
            )
    def __call__(self, img):
        img = self.transform(img)
        return img




class OurDataset(data.Dataset):
    #Dataset class
    

    def __init__(self,  fields, n_views=0, mode='train'):
        # Attributes
        self.fields = fields
        print(mode,': ', n_views, ' views') 
        self.n_views = n_views

    def __len__(self):
        #Returns the length of the dataset.
        
        return self.n_views

    def __getitem__(self, idx):
        """Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        """
       ######################################
        data = {}
        for field_name, field in self.fields.items():
            field_data = field.load(idx)

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        return data



def collate_remove_none(batch):
    """Collater that puts each data field into a tensor with outer dimension batch size.

    Args:
        batch: batch
    """
   ##################################################################################
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    # Worker init function to ensure true randomness.
    
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


class pose_DataField(object):
    def __init__(self, model_path,
                 transform=None, 
                 with_camera=False, 
                with_depth=False,
                 use_DPT=False, scene_name=[' '], mode='train', spherify=False, 
                 load_ref_img=False,customized_poses=False,
                 customized_focal=False,resize_factor=2, depth_net='dpt',crop_size=0, 
                 random_ref=False,norm_depth=False,load_colmap_poses=True, sample_rate=8, **kwargs):
        """load images, depth maps, etc.
        Args:
            model_path (str): path of dataset
            transform (class, optional):  transform made to the image. Defaults to None.
            with_camera (bool, optional): load camera intrinsics. Defaults to False.
            with_depth (bool, optional): load gt depth maps (if available). Defaults to False.
            DPT (bool, optional): run DPT model. Defaults to False.
            scene_name (list, optional): scene folder name. Defaults to [' '].
            mode (str, optional): train/eval/all/render. Defaults to 'train'.
            spherify (bool, optional): spherify colmap poses (no effect to training). Defaults to False.
            load_ref_img (bool, optional): load reference image. Defaults to False.
            customized_poses (bool, optional): use GT pose if available. Defaults to False.
            customized_focal (bool, optional): use GT focal if provided. Defaults to False.
            resize_factor (int, optional): image downsample factor. Defaults to 2.
            depth_net (str, optional): which depth estimator use. Defaults to 'dpt'.
            crop_size (int, optional): crop if images have black border. Defaults to 0.
            random_ref (bool/int, optional): if use a random reference image/number of neaest images. Defaults to False.
            norm_depth (bool, optional): normalise depth maps. Defaults to False.
            load_colmap_poses (bool, optional): load colmap poses. Defaults to True.
            sample_rate (int, optional): 1 in 'sample_rate' images as test set. Defaults to 8.
        """
        self.transform = transform
        self.with_camera = with_camera
        self.with_depth = with_depth
        self.use_DPT = use_DPT
        self.mode = mode
        self.ref_img = load_ref_img
        self.random_ref = random_ref
        self.sample_rate = sample_rate
        
        load_dir = os.path.join(model_path, scene_name[0])
        if crop_size!=0:
            depth_net = depth_net + '_' + str(crop_size)
        poses, bds, imgs, img_names, crop_ratio, focal_crop_factor, points3D = _load_data(load_dir, factor=resize_factor, crop_size=crop_size, load_colmap_poses=load_colmap_poses)
        if load_colmap_poses:
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
            self.hwf = input_poses[:,:3,:]
            input_poses = input_poses[:,:3,:4]
            H, W, focal = hwf
            H, W = int(H), int(W)
            poses_tensor = torch.from_numpy(input_poses)
            bottom = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(0)
            bottom = bottom.repeat(poses_tensor.shape[0], 1, 1)
            c2ws_colmap = torch.cat([poses_tensor, bottom], 1)
            

        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        imgs = np.transpose(imgs, (0, 3, 1, 2))

        if customized_focal:
            focal_gt = np.load(os.path.join(load_dir, 'intrinsics.npz'))['K'].astype(np.float32)
            if resize_factor is None:
                resize_factor = 1
            fx = focal_gt[0, 0] / resize_factor
            fy = focal_gt[1, 1] / resize_factor
        else:
            fx, fy = focal, focal
        fx = fx / focal_crop_factor
        fy = fy / focal_crop_factor
        
        _, _, h, w = imgs.shape
        self.H, self.W, self.focal = h, w, fx
        self.K = np.array([[2*fx/w, 0, 0, 0], 
            [0, -2*fy/h, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]).astype(np.float32)
        ids = np.arange(imgs.shape[0])
        i_test = ids[int(sample_rate/2)::sample_rate] #test采样的方式和下面保持一致，每10个一组
        #i_train = np.array([i for i in ids if i not in i_test])
        i_train = np.array([i for i in ids])
        self.i_train = i_train
        self.i_test = i_test
        image_list_train = [img_names[i] for i in i_train]
        image_list_test = [img_names[i] for i in i_test]
        print('test set: ', image_list_test)

        if customized_poses: #false
            c2ws_gt = np.load(os.path.join(load_dir, 'gt_poses.npz'))['poses'].astype(np.float32)
            T = torch.tensor(np.array([[1, 0, 0, 0],[0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)) # ScanNet coordinate
            c2ws_gt = torch.from_numpy(c2ws_gt)
            c2ws = c2ws_gt @ T
        else:
            c2ws = c2ws_colmap

        self.points3D = points3D
        
        
        self.N_imgs_train = len(i_train)
        self.N_imgs_test = len(i_test)
        
        pred_depth_path = os.path.join(load_dir, depth_net)
        self.dpt_depth = None
        if mode in ('train','eval_trained', 'render'):
            self.imgs = imgs[i_train]
            self.c2ws = c2ws[i_train]
            self.N_imgs = len(i_train)
            if load_colmap_poses:
                self.c2ws_colmap = c2ws_colmap[i_train]
            if not use_DPT:
                self.dpt_depth = load_depths_npz(image_list_train, pred_depth_path, norm=norm_depth)
            if with_depth:
                self.depth = load_gt_depths(image_list_train, load_dir, crop_ratio=crop_ratio)
            self.img_list = image_list_train
        elif mode=='eval':
            self.imgs = imgs[i_test]
            self.c2ws = c2ws[i_test]
            if load_colmap_poses:
                self.c2ws_colmap = c2ws_colmap[i_test]
            if with_depth:
                self.depth = load_gt_depths(image_list_test, load_dir, crop_ratio=crop_ratio)
            self.N_imgs = len(i_test)
            self.img_list = image_list_test
        elif mode=='all':
            self.imgs = imgs
            self.c2ws = c2ws
            if load_colmap_poses:
                self.c2ws_colmap = c2ws_colmap
            self.N_imgs = len(i_train) + len(i_test)
            if not use_DPT:
                self.dpt_depth = load_depths_npz(img_names, pred_depth_path,  norm=norm_depth)
            if with_depth:
                self.depth = load_gt_depths(img_names, load_dir, crop_ratio=crop_ratio)
            self.img_list = img_names

       

    def load(self, input_idx_img=None):
        #Loads the field.
        
        return self.load_field(input_idx_img)

    def load_image(self, idx, data={}):
        image = self.imgs[idx]
        data['tgt_img'] = image
        data['tgt_img'] = torch.from_numpy(data['tgt_img'])
        if self.use_DPT:
            data_in = {"image": np.transpose(image, (1, 2, 0))}
            data_in = self.transform(data_in)
            data['normalised_img'] = data_in['image']
        data['idx'] = idx
        data['T_tgt'] = torch.from_numpy(self.c2w_to_w2c(self.c2ws[idx]))
    def load_ref_img(self, idx, data={}):
        if self.random_ref:
            if idx==0:
                ref_idx = self.N_imgs-1
            else:
                #ran_idx = random.randint(1, min(self.random_ref, self.N_imgs-idx-1))
                ran_idx = 1
                ref_idx = idx - ran_idx
        image = self.imgs[ref_idx]
        if self.dpt_depth is not None:
            dpt = self.dpt_depth[ref_idx]
            data['ref_dpts'] = dpt
        if self.use_DPT:
            data_in = {"image": np.transpose(image, (1, 2, 0))}
            data_in = self.transform(data_in)
            normalised_ref_img = data_in['image']
            data['normalised_ref_img'] = normalised_ref_img
        if self.with_depth:
            depth = self.depth[ref_idx]
            data['ref_depths'] = depth
        data['ref_imgs'] = image
        data['ref_imgs'] = torch.from_numpy(data['ref_imgs'])
        data['ref_idxs'] = ref_idx
        ##

        data['T_ref'] = torch.from_numpy(self.c2w_to_w2c(self.c2ws[ref_idx]))

    def load_depth(self, idx, data={}):
        depth = self.depth[idx]
        data['depth'] = depth
    def load_DPT_depth(self, idx, data={}):
        depth_dpt = self.dpt_depth[idx]
        data['dpt'] = depth_dpt

    def load_camera(self, idx, data={}):
        data['camera_mat'] = self.K
        data['scale_mat'] = np.array([[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]).astype(np.float32)
        data['idx'] = idx
       ###########################################################
        """
        C = CAMERAS.split()
        name, camera_model, width, height = C[:4]
        params = np.array(C[4:], float)
        camera = Camera.from_colmap(dict(
                model=camera_model, params=params,
                width=int(width), height=int(height)))
        data['camera'] = camera
        """
    def c2w_to_w2c(self,c2w):
        w2c = np.linalg.inv(c2w)
        return w2c

    def load_T_r2q(self, data={}):
        
        """
        def mat_to_Pose(matrix):
            r_vector = matrix[:3, :3]
            t_vector = matrix[:3, 3]
            pose = Pose.from_Rt(r_vector, t_vector)
            return pose
        """

        c2w_ref = torch.from_numpy(self.c2w_to_w2c(data['T_ref']))

        T_r2q_gt = data['T_tgt'] @ c2w_ref
        #T_r2q_gt = mat_to_Pose(T_r2q_gt)
        data['T_r2q_gt'] = T_r2q_gt

    def load_p3d(self, data={}):
        def T_mul_p3d(T, p3d):
            T = T.float()
            p3d = torch.tensor(p3d).float()
            points3D = torch.cat((p3d, torch.ones(p3d.shape[0], 1)), dim=1)
            mul = torch.matmul(T, points3D.t())
            p3D = mul.t()[:, :3]
            return p3D

        def get_valid(p3d):
            eps = 0.001
            c = torch.tensor([[480,270]])
            f = 593.97969
            size = torch.tensor([[960,540]])
            #project
            z = p3d[..., -1]
            visible = z > eps
            z = z.clamp(min=eps)
            p2d = p3d[..., :-1] / z.unsqueeze(-1)
            ##denormalize
            p2d = p2d * f + c
            ##in_image
            in_image = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)

            valid = visible & in_image
            return valid



        p3D = self.points3D['points3D']
        obs = self.points3D['p3D_observed'][data['ref_idxs']]
        valid = get_valid(T_mul_p3d(data['T_ref'], p3D[obs].squeeze()))
        obs = obs[valid.numpy()].squeeze()
        max_num_points3D = 512
        obs = np.random.choice(obs, max_num_points3D)
        data['points3D'] = T_mul_p3d(data['T_ref'], p3D[obs])
        data['points3D'] = torch.unsqueeze(data['points3D'], 0)

   
        
    def load_field(self, input_idx_img=None):
        if input_idx_img is not None:
            idx_img = input_idx_img
        else:
            idx_img = 0
        # Load the data
        data = {}
        if not self.mode =='render':
            self.load_image(idx_img, data)
            if self.ref_img:
                self.load_ref_img(idx_img, data)
            if self.with_depth:
                self.load_depth(idx_img, data)
            if self.dpt_depth is not None:
                self.load_DPT_depth(idx_img, data)
        if self.with_camera:
            self.load_camera(idx_img, data)

        self.load_T_r2q(data)
        self.load_p3d(data)
        
        return data

def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample
        size (tuple): image size

    Returns:
        tuple: new size
    """
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height)."""

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"].astype(bool)

        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std."""

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "disparity" in sample:
            disparity = sample["disparity"].astype(np.float32)
            sample["disparity"] = np.ascontiguousarray(disparity)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample



def concatenate_append(old, new, dim):
    new = np.concatenate(new, 0).reshape(-1, dim)
    if old is not None:
        new = np.concatenate([old, new], 0)

    return new
####################################################################################################################
############################ load dataset for localrf——wcd 20230801 ################################################
####################################################################################################################
class LocalRFDataset(data.Dataset):
    def __init__(
        self,
        datadir,
        pose_cfg,
        split="train",
        frames_chunk=20,
        downsampling=-1,
        load_depth=False,
        load_flow=False,
        with_GT_poses=False,
        n_init_frames=7,
        subsequence=[0, -1],
        test_frame_every=10,
        frame_step=1,
        
    ):
        self.root_dir = datadir
        self.split = split
        self.frames_chunk = max(frames_chunk, n_init_frames) #the number of frames loaded each time,(default:20)
        self.downsampling = downsampling
        self.load_depth = load_depth
        self.load_flow = load_flow
        self.frame_step = frame_step
        self.pose_cfg=pose_cfg
        self.pose_dataloader = get_dataloader(cfg=self.pose_cfg, mode="train", shuffle=False)
        if with_GT_poses:
            with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
                self.transforms = json.load(f)
            self.image_paths = [os.path.basename(frame_meta["file_path"]) for frame_meta in self.transforms["frames"]]
            self.image_paths = sorted(self.image_paths)
            poses_dict = {os.path.basename(frame_meta["file_path"]): frame_meta["transform_matrix"] for frame_meta in self.transforms["frames"]}
            poses = []
            for idx, image_path in enumerate(self.image_paths):
                pose = np.array(poses_dict[image_path], dtype=np.float32)
                poses.append(pose)

            self.rel_poses = []
            for idx in range(len(poses)):
                if idx == 0:
                    pose = np.eye(4, dtype=np.float32)
                else:
                    pose = np.linalg.inv(poses[idx - 1]) @ poses[idx]
                self.rel_poses.append(pose)
            self.rel_poses = np.stack(self.rel_poses, axis=0) 

            scale = 2e-2 / np.median(np.linalg.norm(self.rel_poses[:, :3, 3], axis=-1))
            self.rel_poses[:, :3, 3] *= scale
            self.rel_poses = self.rel_poses[::frame_step]

        else:
            self.image_paths = sorted(os.listdir(os.path.join(self.root_dir, "images")))
            #self.
        if subsequence != [0, -1]:
            self.image_paths = self.image_paths[subsequence[0]:subsequence[1]]

        self.image_paths = self.image_paths[::frame_step] #slice with step length=frame_step(default:1)
        self.all_image_paths = self.image_paths

        self.test_mask = []
        self.test_paths = []
        for idx, image_path in enumerate(self.image_paths):
            fbase = os.path.splitext(image_path)[0]
            index = int(fbase) if fbase.isnumeric() else idx
            if test_frame_every > 0 and index % test_frame_every == 0: #for Francis: index%20=19
                self.test_paths.append(image_path)
                self.test_mask.append(1)
            else:
                self.test_mask.append(0)
        self.test_mask = np.array(self.test_mask)

        if split=="test":
            self.image_paths = self.test_paths
            self.frames_chunk = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self.all_fbases = {os.path.splitext(image_path)[0]: idx for idx, image_path in enumerate(self.image_paths)}

        self.white_bg = False

        self.near_far = [0.1, 1e3] # Dummi
        self.scene_bbox = 2 * torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        self.all_rgbs = None
        self.all_invdepths = None
        self.all_fwd_flow, self.all_fwd_mask, self.all_bwd_flow, self.all_bwd_mask = None, None, None, None
        self.all_loss_weights = None

        self.active_frames_bounds = [0, 0]
        self.loaded_frames = 0
        self.activate_frames(n_init_frames)


    def activate_frames(self, n_frames=1):
        self.active_frames_bounds[1] += n_frames
        self.active_frames_bounds[1] = min(
            self.active_frames_bounds[1], self.num_images
        )

        if self.active_frames_bounds[1] > self.loaded_frames:
            self.read_meta()
            



    def has_left_frames(self):
        return self.active_frames_bounds[1] < self.num_images

    def deactivate_frames(self, first_frame):
        n_frames = first_frame - self.active_frames_bounds[0]
        self.active_frames_bounds[0] = first_frame

        self.all_rgbs = self.all_rgbs[n_frames * self.n_px_per_frame:] 
        if self.load_depth:
            self.all_invdepths = self.all_invdepths[n_frames * self.n_px_per_frame:]
        if self.load_flow:
            self.all_fwd_flow = self.all_fwd_flow[n_frames * self.n_px_per_frame:]
            self.all_fwd_mask = self.all_fwd_mask[n_frames * self.n_px_per_frame:]
            self.all_bwd_flow = self.all_bwd_flow[n_frames * self.n_px_per_frame:]
            self.all_bwd_mask = self.all_bwd_mask[n_frames * self.n_px_per_frame:]
        self.all_loss_weights = self.all_loss_weights[n_frames * self.n_px_per_frame:]



    def read_meta(self):
        def read_image(i):
            image_path = os.path.join(self.root_dir, "images", self.image_paths[i]) #join 3 parts of image path together
            motion_mask_path = os.path.join(self.root_dir, "masks", 
                f"{os.path.splitext(self.image_paths[i])[0]}.png")
            
            
            img = cv2.imread(image_path)[..., ::-1] #BGR to RGB, need RGB as input
            img = img.astype(np.float32) / 255 #normalize RGB values from [0,255] to [0,1]
            if self.downsampling != -1:
                scale = 1 / self.downsampling
                img = cv2.resize(img, None, 
                    fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            if self.load_depth:
                invdepth_path = os.path.join(self.root_dir, "depth", 
                    f"{os.path.splitext(self.image_paths[i])[0]}.png")
                invdepth = cv2.imread(invdepth_path, -1).astype(np.float32) #depth file-->numpy
                invdepth = cv2.resize(
                    invdepth, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA) #match size to image
            else:
                invdepth = None

            if self.load_flow:
                glob_idx = self.all_image_paths.index(self.image_paths[i])
                if glob_idx+1 < len(self.all_image_paths):
                    fwd_flow_path = self.all_image_paths[glob_idx+1]
                else:
                    fwd_flow_path = self.all_image_paths[0]
                if self.frame_step != 1:
                    fwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"fwd_step{self.frame_step}_{os.path.splitext(fwd_flow_path)[0]}.png")
                    bwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"bwd_step{self.frame_step}_{os.path.splitext(self.image_paths[i])[0]}.png")
                else:
                    fwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"fwd_{os.path.splitext(fwd_flow_path)[0]}.png")
                    bwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"bwd_{os.path.splitext(self.image_paths[i])[0]}.png") #set prefix for fwd_flow and bwd_flow paths
                encoded_fwd_flow = cv2.imread(fwd_flow_path, cv2.IMREAD_UNCHANGED)
                encoded_bwd_flow = cv2.imread(bwd_flow_path, cv2.IMREAD_UNCHANGED)
                flow_scale = img.shape[0] / encoded_fwd_flow.shape[0] 
                encoded_fwd_flow = cv2.resize(
                    encoded_fwd_flow, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)
                encoded_bwd_flow = cv2.resize(
                    encoded_bwd_flow, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)            
                fwd_flow, fwd_mask = decode_flow(encoded_fwd_flow)
                bwd_flow, bwd_mask = decode_flow(encoded_bwd_flow)
                fwd_flow = fwd_flow * flow_scale
                bwd_flow = bwd_flow * flow_scale
            else:
                fwd_flow, fwd_mask, bwd_flow, bwd_mask = None, None, None, None

            if os.path.isfile(motion_mask_path):
                mask = cv2.imread(motion_mask_path, cv2.IMREAD_UNCHANGED)
                if len(mask.shape) != 2:
                    mask = mask[..., 0]
                mask = cv2.resize(mask, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA) > 0
            else:
                mask = None

            return {
                "img": img, 
                "invdepth": invdepth,
                "fwd_flow": fwd_flow,
                "fwd_mask": fwd_mask,
                "bwd_flow": bwd_flow,
                "bwd_mask": bwd_mask,
                "mask": mask,
            }

        n_frames_to_load = min(self.frames_chunk, self.num_images - self.loaded_frames) #decide the num of frames to load according to f_c
        all_data = Parallel(n_jobs=-1, backend="threading")(
            delayed(read_image)(i) for i in range(self.loaded_frames, self.loaded_frames + n_frames_to_load) 
        ) #read new frames in parallel, using all available CPUs
        self.loaded_frames += n_frames_to_load
        all_rgbs = [data["img"] for data in all_data] #img: RGB array for images
        all_invdepths = [data["invdepth"] for data in all_data]
        all_fwd_flow = [data["fwd_flow"] for data in all_data]
        all_fwd_mask = [data["fwd_mask"] for data in all_data]
        all_bwd_flow = [data["bwd_flow"] for data in all_data]
        all_bwd_mask = [data["bwd_mask"] for data in all_data]
        all_mask = [data["mask"] for data in all_data]

        all_laplacian = [
                np.ones_like(img[..., 0]) * cv2.Laplacian(
                            cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F
                        ).var()
            for img in all_rgbs
        ]
        all_loss_weights = [laplacian if mask is None else laplacian * mask for laplacian, mask in zip(all_laplacian, all_mask)]

        self.img_wh = list(all_rgbs[0].shape[1::-1])
        self.n_px_per_frame = self.img_wh[0] * self.img_wh[1]

        ############################## obtain fov(field of view) from COLMAP-standard dataset——wcd 20230720 ###############################
        camera_focal = read_cameras_binary(f"{self.root_dir}/sparse/0/cameras.bin")[1].params[0] #focal obtained from dataset binary params
        self.fov_real = 2 * np.arctan(self.img_wh[0] / (2 * camera_focal)) * 180 / math.pi #to be used in LocalTensoRF init in training
        ###################################################################################################################################
        # self.all_rgbs_real = np.stack(all_rgbs, 0) # use as input for learnpose——wcd 20230725
        if self.split != "train":
            self.all_rgbs = np.stack(all_rgbs, 0)
            
            if self.load_depth:
                self.all_invdepths = np.stack(all_invdepths, 0)
            if self.load_flow:
                self.all_fwd_flow = np.stack(all_fwd_flow, 0)
                self.all_fwd_mask = np.stack(all_fwd_mask, 0)
                self.all_bwd_flow = np.stack(all_bwd_flow, 0)
                self.all_bwd_mask = np.stack(all_bwd_mask, 0)
        else:
            self.all_rgbs = concatenate_append(self.all_rgbs, all_rgbs, 3)
            if self.load_depth:
                self.all_invdepths = concatenate_append(self.all_invdepths, all_invdepths, 1)
            if self.load_flow:
                self.all_fwd_flow = concatenate_append(self.all_fwd_flow, all_fwd_flow, 2)
                self.all_fwd_mask = concatenate_append(self.all_fwd_mask, all_fwd_mask, 1)
                self.all_bwd_flow = concatenate_append(self.all_bwd_flow, all_bwd_flow, 2)
                self.all_bwd_mask = concatenate_append(self.all_bwd_mask, all_bwd_mask, 1)
            self.all_loss_weights = concatenate_append(self.all_loss_weights, all_loss_weights, 1)
    #看一下image有没有数据，all_rgbs是什么 self.image
    


    def __len__(self): #not applied
        return int(1e10)

    def __getitem__(self, i):
        raise NotImplementedError #not implemented or applied
        idx = np.random.randint(self.sampling_bound[0], self.sampling_bound[1])

        return {"rgbs": self.all_rgbs[idx], "idx": idx}

    def get_frame_fbase(self, view_id):
        return list(self.all_fbases.keys())[view_id]

    def sample(self, batch_size, is_refining, optimize_poses, n_views=16):
        active_test_mask = self.test_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]
        test_ratio = active_test_mask.mean()
        if optimize_poses:
            train_test_poses = test_ratio > random.uniform(0, 1) #adjust test ratio
        else:
            train_test_poses = False

        inclusion_mask = active_test_mask if train_test_poses else 1 - active_test_mask
        sample_map = np.arange(
            self.active_frames_bounds[0], 
            self.active_frames_bounds[1], 
            dtype=np.int64)[inclusion_mask == 1]
        
        raw_samples = np.random.randint(0, inclusion_mask.sum(), n_views, dtype=np.int64)

        # Force having the last views during coarse optimization
        if not is_refining and inclusion_mask.sum() > 4:
            raw_samples[:2] = inclusion_mask.sum() - 1
            raw_samples[2:4] = inclusion_mask.sum() - 2
            raw_samples[4] = inclusion_mask.sum() - 3
            raw_samples[5] = inclusion_mask.sum() - 4

        view_ids = sample_map[raw_samples]

        idx = np.random.randint(0, self.n_px_per_frame, batch_size, dtype=np.int64) #batch_size: pixels per view
        idx = idx.reshape(n_views, -1)
        idx = idx + view_ids[..., None] * self.n_px_per_frame
        idx = idx.reshape(-1)

        idx_sample = idx - self.active_frames_bounds[0] * self.n_px_per_frame #minus each element

        return {
            "rgbs": self.all_rgbs[idx_sample], 
            "loss_weights": self.all_loss_weights[idx_sample], 
            "invdepths": self.all_invdepths[idx_sample] if self.load_depth else None,
            "fwd_flow": self.all_fwd_flow[idx_sample] if self.load_flow else None,
            "fwd_mask": self.all_fwd_mask[idx_sample] if self.load_flow else None,
            "bwd_flow": self.all_bwd_flow[idx_sample] if self.load_flow else None,
            "bwd_mask": self.all_bwd_mask[idx_sample] if self.load_flow else None,
            "idx": idx,
            "view_ids": view_ids,
            "train_test_poses": train_test_poses,
            # "rgbs_real": self.all_rgbs_real[idx], #——wcd 20230726
        }
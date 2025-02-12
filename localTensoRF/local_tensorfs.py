# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import cv2
from models.tensorBase import AlphaGridMask
from models.checkpoints import CheckpointIO
from models.tensoRF import TensorVMSplit
#import dataLoader.dataloading as dl

from models.poses import LearnPose #位姿估计子网络
from pathlib import Path
from utils.utils import mtx_to_sixD, sixD_to_mtx
from utils.ray_utils import get_ray_directions_lean, get_rays_lean, get_ray_directions_360
from utils.utils import N_to_reso

import models.utils as ut

from dataLoader.common import _load_data, recenter_poses, poses_avg, normalize

import numpy as np

from dataLoader.aa import *

from models.geometry.wrappers import Pose #assessed through LearnPose ——wcd 20230728
from models.geometry.wrappers import Camera, CAMERAS 

CAMERAS = '1 SIMPLE_PINHOLE 960 540 593.9796929180062 480 270' #change according to camera specification of dataset used ——wcd 20230728 camera参数（实际上没用）
#CAMERAS = read_cameras_binary(f"{self.root_dir}/sparse/0/cameras.bin")[1]
def ids2pixel_view(W, H, ids):
    """
    Regress pixel coordinates from ray indices
    """
    col = ids % W
    row = (ids // W) % H
    view_ids = ids // (W * H)
    return col, row, view_ids

def ids2pixel(W, H, ids): #在forward打乱随机采样时，像素点位置和像素点相互转换
    """
    Regress pixel coordinates from ray indices
    """
    col = ids % W
    row = (ids // W) % H
    return col, row

class LocalTensorfs(torch.nn.Module):
    """
    Self calibrating local tensorfs.
    """

    def __init__(
        self,
        fov,
        datadir,
        n_init_frames,
        n_overlap,
        WH,
        n_iters_per_frame,
        n_iters_reg,
        lr_R_init,
        lr_t_init,
        lr_i_init,
        lr_exposure_init,
        rf_lr_init,
        rf_lr_basis,
        lr_decay_target_ratio,
        N_voxel_list,
        update_AlphaMask_list,
        camera_prior,
        device,
        lr_upsample_reset,
        #LearnPose子网络参数
        #*Posenet_args,
        num_cams,
        #n_views,
        learn_R, 
        learn_t,
        cfg,
        init_c2w,
        pose_dataloader,
        **tensorf_args #tensorBase参数（VMSplit/tensoRF没有自己新定义的参数，全盘继承tensorBase父类）
        # —wcd 20230725

        # frontend_cfg,
        # frontend_video,
        # frontend_net,
        # 在LTrfs类中再实例化一个Frontend类、一个Backend类（from frontend/backend.py import Frontend/Backend），参数放在localrf里面——wcd 20240129
        #Frontend的cfg、args在run.py里——wcd 20240131
    ):

        super(LocalTensorfs, self).__init__()
        # pose-net 初始化
        self.fov = fov
        self.root_dir = datadir #自动从dataset读取的数据获得数据集的焦距（fov）、pose配置地址
        self.n_init_frames = n_init_frames
        self.n_overlap = n_overlap
        self.W, self.H = WH
        self.n_iters_per_frame = n_iters_per_frame
        self.n_iters_reg_per_frame = n_iters_reg
        self.lr_R_init, self.lr_t_init, self.lr_i_init, self.lr_exposure_init = lr_R_init, lr_t_init, lr_i_init, lr_exposure_init
        self.rf_lr_init, self.rf_lr_basis, self.lr_decay_target_ratio = rf_lr_init, rf_lr_basis, lr_decay_target_ratio
        self.N_voxel_per_frame_list = N_voxel_list
        self.update_AlphaMask_per_frame_list = update_AlphaMask_list
        self.device = torch.device(device)
        self.camera_prior = camera_prior
        self.tensorf_args = tensorf_args
        #self.Posenet_args = Posenet_args  # should add parameters in train & arg——wcd 20230725
        self.num_cams=num_cams
        #self.n_views=n_views
        self.learn_R=learn_R
        self.learn_t=learn_t
        self.cfg=cfg
        self.init_c2w=init_c2w #上面五个是learnpose参数
        # Backend的参数逐个定义在这，不要用指针形式——wcd 20240129

        self.is_refining = False
        self.lr_upsample_reset = lr_upsample_reset
        self.pose_loss = 0
        self.lr_factor = 1
        self.regularize = True
        self.n_iters_reg = self.n_iters_reg_per_frame
        self.n_iters = self.n_iters_per_frame
        self.update_AlphaMask_list = update_AlphaMask_list
        self.N_voxel_list = N_voxel_list
        self.pose_dataloader = pose_dataloader
        self.pred = {} #tensorRF的参数
        # Setup pose and camera parameters

        # self.frontend_cfg = frontend_cfg

        self.r_c2w, self.t_c2w = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.w2c_list = []
        
        self.exposure = torch.nn.ParameterList() # do not optimize at first
        self.r_optimizers, self.t_optimizers, self.exp_optimizers, self.pose_linked_rf = [], [], [], []
        ################## set up pose optimizer based on GO-SLAM frontend——wcd 20240131 #######################
        # self.posenet = Frontend(video=self.video,
        #                         cfg = self.pose_cfg,  # cfg中要带上frontend的参数——wcd 20240131
        #                         net=self.net,
        #                         config=self.frontend_config,                   #args要逐个看
        #                         device=self.device,
        #                         max_frames=self.frontend_max_frames,
        #                         only_tracking=self.frontend_only_tracking,
        #                         make_video=self.frontend_make_video,
        #                         input_folder=self.frontend_input_folder,
        #                         output_folder=self.frontend_output_folder,
        #                         image_size=self.frontend_image_size,
        #                         calibration_txt=self.calibration_txt,
        #                         mode=self.frontend_mode)
        ########################################################################################################

        ################## set up pose net & load parameters——wcd 20230731 ################# LearnPose子网络的初始化
        self.Posenets = LearnPose(device=self.device, 
                                  num_cams=self.num_cams,
                                  #n_views=self.n_views,
                                  learn_R=self.learn_R,
                                  learn_t=self.learn_t,
                                  cfg=self.cfg,
                                  init_c2w=self.init_c2w) #——wcd 20230725
        #self.pose_ckpts_dir = Path("/data/nope-nerf-ours/checkpoints")
        #self.pose_ckpts_dir = Path("/data/localrf-main-latest/log/ignatius_log_3")
        #self.pose_init_cp = ut.get_cktps(self.pose_ckpts_dir)
        #self.pose_init_cp = torch.load(str(self.pose_init_cp), map_location='cpu')
        #self.pose_model_dict = self.Posenets.state_dict()
        #self.pose_init_cp_model = {k: v for k, v in self.pose_init_cp['model'].items() if k in self.pose_model_dict}
        #self.pose_init_cp_model = {k: v for k, v in self.pose_init_cp['kwargs'].items() if k in self.pose_model_dict}
        #self.pose_model_dict.update(self.pose_init_cp_model)
        #self.Posenets.load_state_dict(self.pose_model_dict)
        self.Posenets.to(self.device)
        #####################################################################################
        ####################### pose optimizers ——wcd 20230727 ###############################
        #params = [(n, p) for n, p in self.Posenets.named_parameters() if p.requires_grad]
        #lr_params = ut.pack_lr_parameters(  
        #    params, base_lr=1e-05, lr_scaling=[[100, ['dampingnet.const']]])
        #self.pose_optimizers=torch.optim.Adam(lr_params, lr=1e-05)   #——wcd 20230727 
        out_dir = '/data/localrf-main-latest/checkpoints/church_ckpt'
        #####单lr
        
        
        ######双lr的ckpt加载
        params = [(n, p) for n, p in self.Posenets.named_parameters() if p.requires_grad]
        lr_params = ut.pack_lr_parameters(params, base_lr=5e-04, lr_scaling=[[100,['dampingnet.const']]])
        optimizer_fn = torch.optim.Adam
        self.pose_optimizers = optimizer_fn(lr_params, lr=5e-04)
        checkpoint_io_pose = CheckpointIO(out_dir, model=self.Posenets, optimizer=self.pose_optimizers)
        
        def lr_fn(it):  # noqa: E306
            if True:
                return 1
        try:
            pose_load_dir = 'model_pose_530000.pt'
            load_dict = checkpoint_io_pose.load(pose_load_dir, load_model_only=cfg['training']['load_ckpt_model_only'])
        except FileExistsError:
            load_dict = dict()
        
        #self.pose_scheduler=torch.optim.lr_scheduler.MultiplicativeLR(self.pose_optimizers, lr_fn) #lr_fn param?
        #self.pose_optimizers.load_state_dict(self.pose_init_cp['optimizer'])
        #self.pose_scheduler.load_state_dict(self.pose_init_cp['lr_scheduler'])    
        ######################################################################################
        
        
        self.blending_weights = torch.nn.Parameter(
            torch.ones([1, 1], device=self.device, requires_grad=False), 
            requires_grad=False,
        ) #不同localrf间重叠帧的权重处理
        for _ in range(n_init_frames):
            self.append_frame() #append initial frames first (before appending the first rf)
            self.optimizer_Posenets_step(self.pose_loss, optimize_poses=True)

        if self.camera_prior is not None:
            focal = self.camera_prior["transforms"]["fl_x"]
            focal *= self.W / self.camera_prior["transforms"]["w"]
        else:
            fov = fov * math.pi / 180
            focal = self.W / math.tan(fov / 2) / 2
        
        self.init_focal = torch.Tensor([focal]).to(self.device)
        
        self.focal_offset = torch.ones(1, device=device)
        self.center_rel = 0.5 * torch.ones(2, device=device) #——wcd 20230721
        
        ################## temporarily cancel the optimization of focal——wcd 20230721 #################### 事实上最后发现focal和exp均优化时，对大多数数据集效果更好
        self.init_focal = torch.nn.Parameter(torch.Tensor([focal]).to(self.device))

        self.focal_offset = torch.nn.Parameter(torch.ones(1, device=device))
        self.center_rel = torch.nn.Parameter(0.5 * torch.ones(2, device=device))
        
        if lr_i_init > 0:
            self.intrinsic_optimizer = torch.optim.Adam([self.focal_offset, self.center_rel], betas=(0.9, 0.99), lr=self.lr_i_init)
        ###################################################################################################
        

        # Setup radiance fields
        self.tensorfs = []
        self.rf_optimizers, self.rf_iter = [], []
        self.world2rf = torch.nn.ParameterList()
        self.append_rf() #针对多个localrf及其对应的优化器，设置空列表

        
    def append_rf(self, n_added_frames=1): #图片加载至离开当前rf时，加载新rf
        self.is_refining = False
        if len(self.tensorfs) > 0:
            n_overlap = min(n_added_frames, self.n_overlap, self.blending_weights.shape[0] - 1)
            weights_overlap = 1 / n_overlap + torch.arange(
                0, 1, 1 / n_overlap
            )
            self.blending_weights.requires_grad = False
            self.blending_weights[-n_overlap :, -1] = 1 - weights_overlap
            new_blending_weights = torch.zeros_like(self.blending_weights[:, 0:1])
            new_blending_weights[-n_overlap :, 0] = weights_overlap
            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, new_blending_weights], dim=1),
                requires_grad=False,
            )
            world2rf = -self.t_c2w[-1].clone().detach()
            self.tensorfs[-1].to(torch.device("cpu"))
            torch.cuda.empty_cache()
        else:
            world2rf = torch.zeros(3, device=self.device)

        self.tensorfs.append(TensorVMSplit(device=self.device, **self.tensorf_args)) #每个tensorf靠一个tensorVMSplit子网络定义；每次iteration优化的也是这部分的参数
        
        self.world2rf.append(torch.nn.Parameter(world2rf.clone().detach()))
        
        self.rf_iter.append(0)

        grad_vars = self.tensorfs[-1].get_optparam_groups(
            self.rf_lr_init, self.rf_lr_basis
        )
        self.rf_optimizers.append(torch.optim.Adam(grad_vars, betas=(0.9, 0.99)))
   
    def append_frame(self): #加gt_pose，第0张从posedata里读，后面的相乘 #加载图片及其相应位姿（由learnpose子网络得到，未优化）
        
        if len(self.r_c2w) == 0: #第一张，直接用gt_pose
            #################### add poses from updated video poses——wcd 20240131 ##########################
            #self.posedata = self.video.poses[0]
            #self.r_c2w.append(torch.nn.Parameter(self.posedata[:, :3]))
            #self.t_c2w.append(torch.nn.Parameter(self.posedata[:, 3]))
            ################################################################################################

            ############# add c2w for each frame to list, obtained from Posenets——wcd 20230809 #############
            self.PoseData = self.pose_dataloader[0].dataset[0]
            self.pred = (self.Posenets(self.PoseData))
            #self.c2w_list.append(self.Posenets.get_T())
            self.w2c_list.append(self.PoseData['img.T_tgt'].to(self.device)) 
            #self.c2w_list.append(torch.inverse(self.w2c_list[0])[:3, :])
            c2w = torch.inverse(self.w2c_list[0])[:3, :]
            c2w_sixD = mtx_to_sixD(c2w[:, :3])
            self.r_c2w.append(torch.nn.Parameter(c2w_sixD))
            self.t_c2w.append(torch.nn.Parameter(c2w[:, 3]))
            print("=> Training pose of the zero-th frame...")
            #self.r_c2w.append(torch.nn.Parameter(torch.eye(3, 2, device=self.device)))
            
            #self.t_c2w.append(torch.nn.Parameter(torch.zeros(3, device=self.device)))

            ################################################################################################
            self.pose_linked_rf.append(0)  
        elif len(self.r_c2w) == 1:  #第二张，对learnpose部分单独优化500代以取得更准确初始位姿
            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, self.blending_weights[-1:, :]], dim=0),
                requires_grad=False,
            )
            print("=> Training pose of the first frame...")
            for _ in range(0, 500):
                PoseData = self.pose_dataloader[0].dataset[1]
                pred = (self.Posenets(PoseData))
                loss = self.Posenets.loss(pred, PoseData)['total']
                self.optimizer_Posenets_step(loss, optimize_poses=True)
            
            self.PoseData = self.pose_dataloader[0].dataset[1]
            self.pred = (self.Posenets(self.PoseData))
            self.w2c_list.append(torch.matmul(self.Posenets.get_T().clone().detach().to(self.device), self.w2c_list[-1])) #nopenerf.training line 252
            c2w = torch.inverse(self.w2c_list[-1])[:3, :]
            c2w_sixD = mtx_to_sixD(c2w[:, :3])
            self.r_c2w.append(torch.nn.Parameter(c2w_sixD))
            self.t_c2w.append(torch.nn.Parameter(c2w[:, 3]))
            rf_ind = int(torch.nonzero(self.blending_weights[-1, :])[0])
            self.pose_linked_rf.append(rf_ind)

        else:
            self.blending_weights = torch.nn.Parameter(
                torch.cat([self.blending_weights, self.blending_weights[-1:, :]], dim=0),
                requires_grad=False,
            )
            
            ############# add c2w for each frame to list, obtained from Posenets——wcd 20230809 #############
            #for _ in range(0, 200):
            #    PoseData = self.pose_dataloader[0].dataset[len(self.pose_linked_rf)]
            #    pred = (self.Posenets(PoseData))
            #    loss = self.Posenets.loss(pred, PoseData)['total']
            #    self.optimizer_Posenets_step(loss, optimize_poses=True)
            self.PoseData = self.pose_dataloader[0].dataset[len(self.pose_linked_rf)]
            self.pred = (self.Posenets(self.PoseData))
            self.w2c_list.append(torch.matmul(self.Posenets.get_T().clone().detach().to(self.device), self.w2c_list[-1])) #nopenerf.training line 252
            c2w = torch.inverse(self.w2c_list[-1])[:3, :]
            c2w_sixD = mtx_to_sixD(c2w[:, :3])
            self.r_c2w.append(torch.nn.Parameter(c2w_sixD))
            self.t_c2w.append(torch.nn.Parameter(c2w[:, 3]))
            #self.r_c2w.append(torch.nn.Parameter(mtx_to_sixD(sixD_to_mtx(self.r_c2w[-1].clone().detach()[None]))[0]))
            #self.t_c2w.append(torch.nn.Parameter(self.t_c2w[-1].clone().detach()))
            ################################################################################################
            rf_ind = int(torch.nonzero(self.blending_weights[-1, :])[0])
            self.pose_linked_rf.append(rf_ind)
        self.pose_loss = self.Posenets.loss(self.pred, self.PoseData)['total']
        self.pose_loss = self.pose_loss.to(self.device)
        self.pose_loss.item() 
       
        self.exposure.append(torch.nn.Parameter(torch.eye(3, 3, device=self.device)))

        if self.camera_prior is not None: #default: None
            idx = len(self.r_c2w) - 1
            rel_pose = self.camera_prior["rel_poses"][idx]
            last_r_c2w = sixD_to_mtx(self.r_c2w[-1].clone().detach()[None])[0]
            self.r_c2w[-1] = last_r_c2w @ rel_pose[:3, :3]
            self.t_c2w[-1].data += last_r_c2w @ rel_pose[:3, 3]
        self.r_optimizers.append(torch.optim.Adam([self.r_c2w[-1]], betas=(0.9, 0.99), lr=self.lr_R_init)) 
        self.t_optimizers.append(torch.optim.Adam([self.t_c2w[-1]], betas=(0.9, 0.99), lr=self.lr_t_init)) 
        self.exp_optimizers.append(torch.optim.Adam([self.exposure[-1]], betas=(0.9, 0.99), lr=self.lr_exposure_init)) 
        
    #def get_pose_loss(self):
    #    return self.pose_loss.clone()
    ######################## cancel the optimization of r/t_c2w ——wcd 20230725 ############################
    
    #pose的优化过程，暂定用Frontend类里自带的update来进行，每次append之后，将目前的图片序列作为video，按Frontend进行更新，poses结果在Frontend里自带——wcd 20240129
    #重要的是把Frontend里的参数形式和输入图片格式对应，然后才能update

    def optimizer_step_poses_only(self, loss): #只优化r、t
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                self.r_optimizers[idx].zero_grad()
                self.t_optimizers[idx].zero_grad()
        
        loss.backward()

        # Optimize poses
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                self.r_optimizers[idx].step()
                self.t_optimizers[idx].step()
        #暂时保留——wcd 20240131
    
    
    ################# separate optimization for poses ——wcd 20230809 ###############
    def optimizer_Posenets_step(self, loss, optimize_poses): #只优化learnpose子网络参数（顺序：梯度清零——损失回传——前向传播，下同）
        if optimize_poses:
            self.pose_optimizers.zero_grad()
            loss.backward()
            self.pose_optimizers.step() 
    # self.posenet()——wcd 20240131
    ################################################################################
    
 
    def optimizer_step(self, loss, optimize_poses): #除learnpose以外全部优化
        if self.rf_iter[-1] == 0:
            self.lr_factor = 1
            self.n_iters = self.n_iters_per_frame
            self.n_iters_reg = self.n_iters_reg_per_frame
            

        elif self.rf_iter[-1] == 1:
            n_training_frames = (self.blending_weights[:, -1] > 0).sum()
            self.n_iters = int(self.n_iters_per_frame * n_training_frames)
            self.n_iters_reg = int(self.n_iters_reg_per_frame * n_training_frames)
            self.lr_factor = self.lr_decay_target_ratio ** (1 / self.n_iters)
            self.N_voxel_list = {int(key * n_training_frames): self.N_voxel_per_frame_list[key] for key in self.N_voxel_per_frame_list}
            self.update_AlphaMask_list = [int(update_AlphaMask * n_training_frames) for update_AlphaMask in self.update_AlphaMask_per_frame_list]

        self.regularize = self.rf_iter[-1] < self.n_iters_reg
        
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                # Poses
                
                if optimize_poses:
                    for param_group in self.r_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    for param_group in self.t_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    self.r_optimizers[idx].zero_grad()
                    self.t_optimizers[idx].zero_grad()

                if self.lr_exposure_init > 0:
                    for param_group in self.exp_optimizers[idx].param_groups:
                        param_group["lr"] *= self.lr_factor
                    self.exp_optimizers[idx].zero_grad()
        
        # Intrinsics
        if (
            self.lr_i_init > 0 and 
            self.blending_weights.shape[1] == 1 and 
            self.is_refining
        ):
            for param_group in self.intrinsic_optimizer.param_groups:
                param_group["lr"] *= self.lr_factor
            self.intrinsic_optimizer.zero_grad()

                
        
        
        # tensorfs
        for optimizer, iteration in zip(self.rf_optimizers, self.rf_iter):
            if iteration < self.n_iters:
                optimizer.zero_grad()
        
        loss.backward()
        # Optimize RFs
        self.rf_optimizers[-1].step()
        if self.is_refining:
            for param_group in self.rf_optimizers[-1].param_groups:
                param_group["lr"] = param_group["lr"] * self.lr_factor

        # Increase RF resolution
        if self.rf_iter[-1] in self.N_voxel_list:
            n_voxels = self.N_voxel_list[self.rf_iter[-1]]
            reso_cur = N_to_reso(n_voxels, self.tensorfs[-1].aabb)
            self.tensorfs[-1].upsample_volume_grid(reso_cur)

            if self.lr_upsample_reset:
                print("reset lr to initial")
                grad_vars = self.tensorfs[-1].get_optparam_groups(
                    self.rf_lr_init, self.rf_lr_basis
                )
                self.rf_optimizers[-1] = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        # Update alpha mask
        if iteration in self.update_AlphaMask_list:
            reso_mask = (self.tensorfs[-1].gridSize / 2).int()
            self.tensorfs[-1].updateAlphaMask(tuple(reso_mask))
        
        for idx in range(len(self.r_optimizers)):
            if self.pose_linked_rf[idx] == len(self.rf_iter) - 1 and self.rf_iter[-1] < self.n_iters:
                # Optimize poses
                if optimize_poses:
                    self.r_optimizers[idx].step()
                    self.t_optimizers[idx].step()
                if self.lr_exposure_init > 0:
                    self.exp_optimizers[idx].step()
        
        if (
            self.lr_i_init > 0 and 
            self.blending_weights.shape[1] == 1 and
            self.is_refining 
        ):
            self.intrinsic_optimizer.step()
        
        
        if self.is_refining: #与pose无关
            self.rf_iter[-1] += 1

        can_add_rf = self.rf_iter[-1] >= self.n_iters - 1
        return can_add_rf
    
    def get_cam2world(self, view_ids=None, starting_id=0): #所有和get_cam2world相关的（train、render）都需要改
        if view_ids is not None:
            r_c2w = torch.stack([self.r_c2w[view_id] for view_id in view_ids], dim=0)
            t_c2w = torch.stack([self.t_c2w[view_id] for view_id in view_ids], dim=0)
            
        else:
            r_c2w = torch.stack(list(self.r_c2w[starting_id:]), dim=0)
            t_c2w = torch.stack(list(self.t_c2w[starting_id:]), dim=0)
           
        return torch.cat([sixD_to_mtx(r_c2w), t_c2w[..., None]], dim = -1)
        
        

    def get_kwargs(self): #get args for LocalTensoRF, excluding those for TensorVMSplit & LearnPose #得到tensorf参数（训练中不用）把Frontend的参数也加进去
        kwargs = {
            "camera_prior": None,
            "fov": self.fov,
            "n_init_frames": self.n_init_frames,
            "n_overlap": self.n_overlap,
            "WH": (self.W, self.H),
            "n_iters_per_frame": self.n_iters_per_frame,
            "n_iters_reg": self.n_iters_reg_per_frame,
            "lr_R_init": self.lr_R_init,
            "lr_t_init": self.lr_t_init,
            "lr_i_init": self.lr_i_init,
            "lr_exposure_init": self.lr_exposure_init,
            "rf_lr_init": self.rf_lr_init,
            "rf_lr_basis": self.rf_lr_basis,
            "lr_decay_target_ratio": self.lr_decay_target_ratio,
            "lr_decay_target_ratio": self.lr_decay_target_ratio,
            "N_voxel_list": self.N_voxel_per_frame_list,
            "update_AlphaMask_list": self.update_AlphaMask_per_frame_list,
            "lr_upsample_reset": self.lr_upsample_reset,
            "num_cams": self.num_cams,
            "learn_R": self.learn_R,
            "learn_t": self.learn_t,
            "cfg": self.cfg,
            "init_c2w": self.init_c2w
        }
        kwargs.update(self.tensorfs[0].get_kwargs())


        return kwargs

    def save(self, path): #保存和加载ckpt（连续训练时不用）
        kwargs = self.get_kwargs()
        ckpt = {"kwargs": kwargs, "state_dict": self.state_dict()}
        torch.save(ckpt, path)

    def load(self, state_dict): #not used in training
        # TODO A bit hacky?
        import re
        n_frames = 0
        for key in state_dict:
            if re.fullmatch(r"r_c2w.[0-9]*", key):
                n_frames += 1
            if re.fullmatch(r"tensorfs.[1-9][0-9]*.density_plane.0", key):
                self.tensorf_args["gridSize"] = [state_dict[key].shape[2], state_dict[key].shape[3], state_dict[f"{key[:-15]}density_line.0"].shape[2]]
                self.append_rf()

        for i in range(len(self.tensorfs)):
            if f"tensorfs.{i}.alphaMask.aabb" in state_dict:
                alpha_volume = state_dict[f'tensorfs.{i}.alphaMask.alpha_volume'].to(self.device)
                aabb = state_dict[f'tensorfs.{i}.alphaMask.aabb'].to(self.device)
                self.tensorfs[i].alphaMask = AlphaGridMask(self.device, aabb, alpha_volume)


        for _ in range(n_frames - len(self.r_c2w)):
            self.append_frame()
        
        self.blending_weights = torch.nn.Parameter(
            torch.ones_like(state_dict["blending_weights"]), requires_grad=False
        )

        self.load_state_dict(state_dict)

    def get_dist_to_last_rf(self): #目前最后一帧到上一个rf终点帧的平移向量
        return torch.norm(self.t_c2w[-1] + self.world2rf[-1])

    def get_reg_loss(self, tvreg, TV_weight_density, TV_weight_app, L1_weight_inital):
        tv_loss = 0
        l1_loss = 0
        if self.rf_iter[-1] < self.n_iters:
            if TV_weight_density > 0:
                tv_weight = TV_weight_density * (self.lr_factor ** self.rf_iter[-1])
                tv_loss += self.tensorfs[-1].TV_loss_density(tvreg).mean() * tv_weight
                
            if TV_weight_app > 0:
                tv_weight = TV_weight_app * (self.lr_factor ** self.rf_iter[-1])
                tv_loss += self.tensorfs[-1].TV_loss_app(tvreg).mean() * tv_weight
    
            if L1_weight_inital > 0:
                l1_loss += self.tensorfs[-1].density_L1() * L1_weight_inital
        return tv_loss, l1_loss

    def focal(self, W): #焦距和图像中心点
        return self.init_focal * self.focal_offset * W / self.W 
    def center(self, W, H):
        return torch.Tensor([W, H]).to(self.center_rel) * self.center_rel

    def forward( #前向传播：根据随机采样结果返回乱序（像素排列）的rgb_map、depth、光线参数等，还需要train中渲染成真正图片
        self,
        ray_ids,
        view_ids,
        W,
        H,
        white_bg=True,
        is_train=True,
        cam2world=None,
        world2rf=None,
        blending_weights=None,
        chunk=16384,
        test_id=False,
        floater_thresh=0,
    ):#绝对位姿还是相对位姿？需要改；确定view_id是否连续
        C = CAMERAS.split()
        name, camera_model, width, height = C[:4]
        params = np.array(C[4:], float)
        camera = Camera.from_colmap(dict(
            model=camera_model, params=params,
            width=int(width), height=int(height))) #set up camera params for Posenets.forward——wcd 20230727
        poses = [] # create empty list to save cam2rfs——wcd 
        #length = int(len(view_ids))
        #poses = torch.zeros(length, dtype=torch.float64)
        i, j = ids2pixel(W, H, ray_ids)
        if self.fov == 360:
            directions = get_ray_directions_360(i, j, W, H)
        else:
            directions = get_ray_directions_lean(i, j, self.focal(W), self.center(W, H))

        if blending_weights is None:
            blending_weights = self.blending_weights[view_ids].clone()
        
        if cam2world is None:
            cam2world = self.get_cam2world(view_ids) #ignore
        if world2rf is None:
            world2rf = self.world2rf
        
        
        #pose=self.pose-net() append到一个list
        # Train a single RF at a time
        if is_train:
            blending_weights[:, -1] = 1
            blending_weights[:, :-1] = 0

        if is_train:
            active_rf_ids = [len(self.tensorfs) - 1]
        else:
            active_rf_ids = torch.nonzero(torch.sum(blending_weights, dim=0))[:, 0].tolist()
        ij = torch.stack([i, j], dim=-1)
        if len(active_rf_ids) == 0:
            print("****** No valid RF")
            return torch.ones([ray_ids.shape[0], 3]), torch.ones_like(ray_ids).float(), torch.ones_like(ray_ids).float(), directions, ij

        cam2rfs = {}
        initial_devices = []
        for rf_id in active_rf_ids:
            
            cam2rf = cam2world.clone()
            cam2rf[:, :3, 3] += world2rf[rf_id]
            cam2rfs[rf_id] = cam2rf

            initial_devices.append(self.tensorfs[rf_id].device)
            if initial_devices[-1] != view_ids.device:
                self.tensorfs[rf_id].to(view_ids.device)

        for key in cam2rfs:
            cam2rfs[key] = cam2rfs[key].repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        blending_weights_expanded = blending_weights.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
        rgbs = torch.zeros_like(directions) 
        depth_maps = torch.zeros_like(directions[..., 0]) 
        N_rays_all = ray_ids.shape[0]
        chunk = chunk // len(active_rf_ids)
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            if chunk_idx != 0:
                torch.cuda.empty_cache()
            directions_chunk = directions[chunk_idx * chunk : (chunk_idx + 1) * chunk]
            blending_weights_chunk = blending_weights_expanded[
                chunk_idx * chunk : (chunk_idx + 1) * chunk
            ]

            for rf_id in active_rf_ids:
                blending_weight_chunk = blending_weights_chunk[:, rf_id]
                cam2rf = cam2rfs[rf_id][chunk_idx * chunk : (chunk_idx + 1) * chunk]

                rays_o, rays_d = get_rays_lean(directions_chunk, cam2rf)
                rays = torch.cat([rays_o, rays_d], -1).view(-1, 6)

                rgb_map_t, depth_map_t = self.tensorfs[rf_id](
                    rays,
                    is_train=is_train,
                    white_bg=white_bg,
                    N_samples=-1,
                    refine=self.is_refining,
                    floater_thresh=floater_thresh,
                )

                rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                    rgbs[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                    rgb_map_t * blending_weight_chunk[..., None]
                )
                depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] = (
                    depth_maps[chunk_idx * chunk : (chunk_idx + 1) * chunk] + 
                    depth_map_t * blending_weight_chunk
                )

        for rf_id, initial_device in zip(active_rf_ids, initial_devices):
            if initial_device != view_ids.device:
                self.tensorfs[rf_id].to(initial_device)
                torch.cuda.empty_cache()

        if self.lr_exposure_init > 0:
            # TODO: cleanup
            if test_id:
                view_ids_m = torch.maximum(view_ids - 1, torch.tensor(0, device=view_ids.device))
                view_ids_m[view_ids_m==view_ids] = 1
                
                view_ids_p = torch.minimum(view_ids + 1, torch.tensor(len(self.exposure) - 1, device=view_ids.device))
                view_ids_p[view_ids_m==view_ids] = len(self.exposure) - 2
                
                exposure_stacked = torch.stack(list(self.exposure), dim=0).clone().detach()
                #exposure_stacked = torch.stack(self.exposure, dim=0).clone().detach()
                exposure = (exposure_stacked[view_ids_m] + exposure_stacked[view_ids_p]) / 2  
            else:
                exposure = torch.stack(list(self.exposure), dim=0)[view_ids]
                #exposure = torch.stack(self.exposure, dim=0)[view_ids]
                #exposure = torch.from_numpy(exposure)
            exposure = exposure.repeat_interleave(ray_ids.shape[0] // view_ids.shape[0], dim=0)
            rgbs = torch.bmm(exposure, rgbs[..., None])[..., 0]
        rgbs = rgbs.clamp(0, 1)

        return rgbs, depth_maps, directions, ij
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
import warnings

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
import yaml
#import tensorflow as tf


warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import sys
import time
import configargparse #——wcd 20230720

from dataLoader.configloading import *

from utils.utils import mtx_to_sixD, sixD_to_mtx
#from utils.inverse_warp import inverse_warp, inverse_warp2

from torch.utils.tensorboard import SummaryWriter

sys.path.append("localTensoRF")
#from dataLoader.localrf_dataset import LocalRFDataset
from dataLoader.localrf_dataset import *
from local_tensorfs import LocalTensorfs
from opt import config_parser
from renderer import render
from utils.utils import (get_fwd_bwd_cam2cams, smooth_poses_spline)
from utils.utils import (N_to_reso, TVLoss, draw_poses, get_pred_flow,
                         compute_depth_loss)
import math
from dataLoader.aa import * #for the obtaining and calculation of fov——wcd 20230720（实际上没用到，用在了localrf_dataset里）
from dataLoader.configloading import *
from comp_ate import compute_ATE, compute_rpe
from align_traj import align_ate_c2b_use_a2b #ATE、RPE


#和reconstruction有关的一些函数，包括了估计pose、渲染图片
def save_transforms(poses_mtx, transform_path, local_tensorfs, train_dataset=None):
    if train_dataset is not None:
        fnames = train_dataset.all_image_paths
    else:
        fnames = [f"{i:06d}.jpg" for i in range(len(poses_mtx))]

    fl = local_tensorfs.focal(local_tensorfs.W).item()
    transforms = {
        "fl_x": fl,
        "fl_y": fl,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": local_tensorfs.W/2,
        "cy": local_tensorfs.H/2,
        "w": local_tensorfs.W,
        "h": local_tensorfs.H,
        "frames": [],
    }
    for pose_mtx, fname in zip(poses_mtx, fnames):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :] = pose_mtx
        frame_data = {
            "file_path": f"images/{fname}",
            "sharpness": 75.0,
            "transform_matrix": pose.tolist(),
        }
        transforms["frames"].append(frame_data)

    with open(transform_path, "w") as outfile:
        json.dump(transforms, outfile, indent=2)


@torch.no_grad()
def render_frames( #render_test & render_path 正常reconstruction的一部分
    args, poses_mtx, local_tensorfs, logfolder, test_dataset, train_dataset
):
    save_transforms(poses_mtx.cpu(), f"{logfolder}/transforms.json", local_tensorfs, train_dataset)
    t_w2rf = torch.stack(list(local_tensorfs.world2rf), dim=0).detach().cpu()
    RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), t_w2rf.clone()[..., None]], axis=-1)
    save_transforms(RF_mtx_inv.cpu(), f"{logfolder}/transforms_rf.json", local_tensorfs)
    
    W, H = train_dataset.img_wh

    if args.render_test: #default:1
        render(
            test_dataset,
            poses_mtx,
            local_tensorfs,
            args,
            W=W, H=H,
            savePath=f"{logfolder}/test",
            save_frames=True,
            test=True,
            train_dataset=train_dataset,
            img_format="png",
            start=0
        )

    if args.render_path: #default:1
        c2ws = smooth_poses_spline(poses_mtx, median_prefilter=True)
        os.makedirs(f"{logfolder}/smooth_spline", exist_ok=True)
        save_transforms(c2ws.cpu(), f"{logfolder}/smooth_spline/transforms.json", local_tensorfs)
        render(
            test_dataset,
            c2ws,
            local_tensorfs,
            args,
            W=int(W / 1.5), H=int(H / 1.5),
            savePath=f"{logfolder}/smooth_spline",
            train_dataset=train_dataset,
            img_format="jpg",
            save_frames=True,
            save_video=True,
            floater_thresh=0.5,
        )

@torch.no_grad()
def render_test(args): #render_only & render_test or render_path 即opt中只render时作这个操作
    # init dataset
    train_dataset = LocalRFDataset(
        f"{args.datadir}",
        pose_cfg=load_config(args.pose_config, "/data/localrf-main-latest/localTensoRF/pose_configs/default.yaml"),
        split="train",
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        n_init_frames=args.n_init_frames,
        with_GT_poses=args.with_GT_poses,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )
    test_dataset = LocalRFDataset(
        f"{args.datadir}",
        pose_cfg=load_config(args.pose_config, "/data/localrf-main-latest/localTensoRF/pose_configs/default.yaml"),
        split="test",
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        with_GT_poses=args.with_GT_poses,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )

    if args.ckpt is None:
        logfolder = f"{args.logdir}"
        ckpt_path = f"{logfolder}/checkpoints.th"
    else:
        ckpt_path = args.ckpt

    if not os.path.isfile(ckpt_path):
        print("Backing up to intermediate checkpoints")
        ckpt_path = f"{logfolder}/checkpoints_tmp.th"
        if not os.path.isfile(ckpt_path):
            print("the ckpt path does not exists!!")
            return  

    with open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location=args.device)
    kwargs = ckpt["kwargs"]
    if args.with_GT_poses:
        kwargs["camera_prior"] = {
            "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(args.device),
            "transforms": train_dataset.transforms
        }
    else:
        kwargs["camera_prior"] = None
    kwargs["device"] = args.device
    local_tensorfs = LocalTensorfs(**kwargs)
    local_tensorfs.load(ckpt["state_dict"])
    local_tensorfs = local_tensorfs.to(args.device)

    logfolder = os.path.dirname(ckpt_path)
    render_frames(
        args,
        local_tensorfs.get_cam2world(),
        local_tensorfs,
        logfolder,
        test_dataset=test_dataset,
        train_dataset=train_dataset
    )

def load_gt_poses(args, dataset, path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for i in range(len(dataset.all_image_paths)):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        poses.append(c2w)
    poses = torch.tensor(poses, dtype=torch.float32)
    poses = poses.to(args.device)
    return poses

def reconstruction(args):
    # Apply speedup factors 针对github里训练速度慢的issue而定义的speedup factors
    args.n_iters_per_frame = int(args.n_iters_per_frame / args.refinement_speedup_factor)
    args.n_iters_reg = int(args.n_iters_reg / args.refinement_speedup_factor)
    args.upsamp_list = [int(upsamp / args.refinement_speedup_factor) for upsamp in args.upsamp_list]
    args.update_AlphaMask_list = [int(update_AlphaMask / args.refinement_speedup_factor) 
                                  for update_AlphaMask in args.update_AlphaMask_list]
    
    args.add_frames_every = int(args.add_frames_every / args.prog_speedup_factor)
    args.lr_R_init = args.lr_R_init * args.prog_speedup_factor
    args.lr_t_init = args.lr_t_init * args.prog_speedup_factor
    args.loss_flow_weight_inital = args.loss_flow_weight_inital * args.prog_speedup_factor
    args.L1_weight = args.L1_weight * args.prog_speedup_factor
    args.TV_weight_density = args.TV_weight_density * args.prog_speedup_factor
    args.TV_weight_app = args.TV_weight_app * args.prog_speedup_factor
    
    # init dataset
    train_dataset = LocalRFDataset(
        datadir=f"{args.datadir}",
        pose_cfg=load_config(path=args.pose_config, default_path="/data/localrf-main-latest/localTensoRF/pose_configs/default.yaml"),
        split="train",
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        with_GT_poses=args.with_GT_poses,
        n_init_frames=args.n_init_frames,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )
    print("pose_dataloader: {}".format(train_dataset.pose_dataloader))
    
    #gt_poses = train_dataset.pose_dataloader[1]['img'].c2ws #从train_dataset读所有的c2w
    #torch.set_printoptions(threshold=sys.maxsize)
    #print('GT poses: {}'.format(gt_poses))
    gt_poses = load_gt_poses(args=args, dataset=train_dataset, path=os.path.join(args.datadir, 'traj.txt'))
    
    test_dataset = LocalRFDataset(
        datadir=f"{args.datadir}",
        pose_cfg=load_config(args.pose_config, "/data/localrf-main-latest/localTensoRF/pose_configs/default.yaml"),
        split="test",
        load_depth=args.loss_depth_weight_inital > 0,
        load_flow=args.loss_flow_weight_inital > 0,
        downsampling=args.downsampling,
        test_frame_every=args.test_frame_every,
        with_GT_poses=args.with_GT_poses,
        subsequence=args.subsequence,
        frame_step=args.frame_step,
    )
    near_far = train_dataset.near_far

    # init resolution
    upsamp_list = args.upsamp_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    logfolder = f"{args.logdir}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    writer = SummaryWriter(log_dir=logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(args.device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)

    # TODO: Add midpoint loading
    # if args.ckpt is not None:
    #     ckpt = torch.load(args.ckpt, map_location=args.device)
    #     kwargs = ckpt["kwargs"]
    #     kwargs.update({"device": args.device})
    #     tensorf = eval(args.model_name)(**kwargs)
    #     tensorf.load(ckpt)
    # else:
    
    print("lr decay", args.lr_decay_target_ratio)

    # linear in logrithmic space
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]
    N_voxel_list = {
        usamp_idx: round(N_voxel**(1/3))**3 for usamp_idx, N_voxel in zip(upsamp_list, N_voxel_list)
    }
    n_views = len(train_dataset.pose_dataloader[0].dataset)
    #n_views = len(train_dataset.pose_dataloader[0])
    #learned_pose = np.empty(n_views)
    learned_pose = []
    for _ in range(n_views):
        learned_pose.append(torch.randn(4, 4).cpu())
    if args.with_GT_poses:
        camera_prior = {
            "rel_poses": torch.from_numpy(train_dataset.rel_poses).to(args.device),
            "transforms": train_dataset.transforms
        }
    
    else:
        camera_prior = None
    
    #print("fov: {}".format(train_dataset.fov_real))
    #localrf网络的初始化
    local_tensorfs = LocalTensorfs(
        camera_prior=camera_prior,
        datadir=train_dataset.root_dir,
        fov = train_dataset.fov_real, #real fov for specific datasets——wcd 20230720 从dataset的焦距、H、W计算出的真实fov（前提：必须标准colmap数据集，带有sparse）
        #fov = 69,
        n_init_frames=min(args.n_init_frames, train_dataset.num_images),
        n_overlap=args.n_overlap,
        WH=train_dataset.img_wh,
        n_iters_per_frame=args.n_iters_per_frame, #默认600，改小或直接取消
        n_iters_reg=args.n_iters_reg,
        lr_R_init=args.lr_R_init,
        lr_t_init=args.lr_t_init,
        lr_i_init=args.lr_i_init,
        lr_exposure_init=args.lr_exposure_init,
        rf_lr_init=args.lr_init,
        rf_lr_basis=args.lr_basis,
        lr_decay_target_ratio=args.lr_decay_target_ratio,
        N_voxel_list=N_voxel_list,
        update_AlphaMask_list=args.update_AlphaMask_list,
        lr_upsample_reset=args.lr_upsample_reset,
        
        #LearnPose部分参数
        #image[]
        num_cams = None,
        learn_R = True,
        learn_t = True,
        cfg = load_config(path=args.pose_config, default_path="/data/localrf-main-latest/localTensoRF/pose_configs/default.yaml"),
        init_c2w = None, # Posenet_args #——wcd 20230725
        pose_dataloader=train_dataset.pose_dataloader,
        #tensorBase参数
        device=args.device,
        alphaMask_thres=args.alpha_mask_thre,
        shadingMode=args.shadingMode,
        aabb=aabb,
        gridSize=reso_cur,
        density_n_comp=n_lamb_sigma,
        appearance_n_comp=n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        rayMarch_weight_thres=args.rm_weight_mask_thre,
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=args.fea_pe,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
        #init_c2w = None, #tensorf_args
    
    )
    
    
        
    
    local_tensorfs = local_tensorfs.to(args.device)

    torch.cuda.empty_cache()
    
    tvreg = TVLoss()
    W, H = train_dataset.img_wh
    
    training = True
    n_added_frames = 0
    last_add_iter = 0
    iteration = 0
    metrics = {}
    start_time = time.time()
    while training: #接下来直到iteration+=1的部分都是一次迭代（iteraion）中做的操作
        
        optimize_poses = args.lr_R_init > 0 or args.lr_t_init > 0
        data_blob = train_dataset.sample(args.batch_size, local_tensorfs.is_refining, optimize_poses)
        view_ids = torch.from_numpy(data_blob["view_ids"]).to(args.device)
        rgb_train = torch.from_numpy(data_blob["rgbs"]).to(args.device)
        loss_weights = torch.from_numpy(data_blob["loss_weights"]).to(args.device)
        train_test_poses = data_blob["train_test_poses"] #按随机取得的16个frame的平均test retio来决定此次iteration是否train test poses
        ray_idx = torch.from_numpy(data_blob["idx"]).to(args.device)
        reg_loss_weight = local_tensorfs.lr_factor ** (local_tensorfs.rf_iter[-1])

        rgb_map, depth_map, directions, ij = local_tensorfs( 
            ray_idx,
            view_ids,
            W,
            H,
            is_train=True,
            test_id=train_test_poses,
           
        ) #调用local_tensorfs的forward，用data_blob随机采样出的16个frame的相关信息，重建出乱序的rgb、depth、directions、ij
        
        # loss
        #rgb loss
        loss = 0.25 * ((torch.abs(rgb_map - rgb_train)) * loss_weights) / loss_weights.mean() #adjust coefficient
               
        loss = loss.mean()
        total_loss = loss
        writer.add_scalar("train/rgb_loss", loss, global_step=iteration)
        

        

 
        ## Regularization loss
        # Get rendered rays schedule
        if local_tensorfs.regularize and args.loss_flow_weight_inital > 0 or args.loss_depth_weight_inital > 0:
            depth_map = depth_map.view(view_ids.shape[0], -1)
            loss_weights = loss_weights.view(view_ids.shape[0], -1)
            depth_map = depth_map.view(view_ids.shape[0], -1)

            writer.add_scalar("train/reg_loss_weights", reg_loss_weight, global_step=iteration)

        # Optical flow loss
        if local_tensorfs.regularize and args.loss_flow_weight_inital > 0:
            if local_tensorfs.fov == 360:  #——wcd 20230720
                raise NotImplementedError
            starting_frame_id = max(train_dataset.active_frames_bounds[0] - 1, 0)
            cam2world = local_tensorfs.get_cam2world(starting_id=starting_frame_id)
            directions = directions.view(view_ids.shape[0], -1, 3)
            ij = ij.view(view_ids.shape[0], -1, 2)
            fwd_flow = torch.from_numpy(data_blob["fwd_flow"]).to(args.device).view(view_ids.shape[0], -1, 2)
            fwd_mask = torch.from_numpy(data_blob["fwd_mask"]).to(args.device).view(view_ids.shape[0], -1)
            fwd_mask[view_ids == len(cam2world) - 1] = 0
            bwd_flow = torch.from_numpy(data_blob["bwd_flow"]).to(args.device).view(view_ids.shape[0], -1, 2)
            bwd_mask = torch.from_numpy(data_blob["bwd_mask"]).to(args.device).view(view_ids.shape[0], -1)
            fwd_cam2cams, bwd_cam2cams = get_fwd_bwd_cam2cams(cam2world, view_ids - starting_frame_id)
            fwd_cam2cams = fwd_cam2cams.to(args.device)
            bwd_cam2cams = bwd_cam2cams.to(args.device)
                       
            pts = directions * depth_map[..., None]
            pred_fwd_flow = get_pred_flow(
                pts, ij, fwd_cam2cams, local_tensorfs.focal(W), local_tensorfs.center(W, H))
            pred_bwd_flow = get_pred_flow(
                pts, ij, bwd_cam2cams, local_tensorfs.focal(W), local_tensorfs.center(W, H))
            flow_loss_arr =  torch.sum(torch.abs(pred_bwd_flow - bwd_flow), dim=-1) * bwd_mask
            flow_loss_arr += torch.sum(torch.abs(pred_fwd_flow - fwd_flow), dim=-1) * fwd_mask
            flow_loss_arr[flow_loss_arr > torch.quantile(flow_loss_arr, 0.9, dim=1)[..., None]] = 0

            flow_loss = (flow_loss_arr).mean() * args.loss_flow_weight_inital * reg_loss_weight / ((W + H) / 2)
            total_loss = total_loss + flow_loss
            writer.add_scalar("train/flow_loss", flow_loss, global_step=iteration)

        # Monocular Depth loss
        if local_tensorfs.regularize and args.loss_depth_weight_inital > 0:
            if local_tensorfs.fov == 360:  #——wcd 20230720
                raise NotImplementedError
            invdepths = torch.from_numpy(data_blob["invdepths"]).to(args.device)
            invdepths = invdepths.view(view_ids.shape[0], -1)
            _, _, depth_loss_arr = compute_depth_loss(1 / depth_map.clamp(1e-6), invdepths)
            depth_loss_arr[depth_loss_arr > torch.quantile(depth_loss_arr, 0.8, dim=1)[..., None]] = 0

            depth_loss = (depth_loss_arr).mean() * args.loss_depth_weight_inital * reg_loss_weight
            total_loss = total_loss + depth_loss 
            writer.add_scalar("train/depth_loss", depth_loss, global_step=iteration)
        
    

        if  local_tensorfs.regularize:
            loss_tv, l1_loss = local_tensorfs.get_reg_loss(tvreg, args.TV_weight_density, args.TV_weight_app, args.L1_weight)
            total_loss = total_loss + loss_tv + l1_loss
            writer.add_scalar("train/loss_tv", loss_tv, global_step=iteration)
            writer.add_scalar("train/l1_loss", l1_loss, global_step=iteration)
        #total_loss = total_loss.to(args.device)
        #total_loss_detach = total_loss.clone().detach()
        #total_loss.retain_grad()
        # Optimizes
        if train_test_poses:
            can_add_rf = False
            #if optimize_poses:
            #    local_tensorfs.optimizer_step_poses_only(total_loss)
                #if (iteration % 100 == 99):
                    #print('optimizing pose')
        else:
            can_add_rf = local_tensorfs.optimizer_step(total_loss, optimize_poses)
            training |= train_dataset.active_frames_bounds[1] != train_dataset.num_images
            #if (iteration % 100 == 99):
            #    print('optimizing pose and rf')
        

        ## Progressive optimization 在不append frame的情况下进行rf和pose的联合optimization
        if not local_tensorfs.is_refining:
            should_refine = (not train_dataset.has_left_frames() or (
                n_added_frames > args.n_overlap and (
                    local_tensorfs.get_dist_to_last_rf().cpu().item() > args.max_drift
                    or (train_dataset.active_frames_bounds[1] - train_dataset.active_frames_bounds[0]) >= args.n_max_frames
                )))
            if should_refine and (iteration - last_add_iter) >= args.add_frames_every:
                local_tensorfs.is_refining = True

            should_add_frame = train_dataset.has_left_frames()
            should_add_frame &= (iteration - last_add_iter + 1) % args.add_frames_every == 0

            should_add_frame &= not should_refine
            should_add_frame &= not local_tensorfs.is_refining
            # Add supervising frames
            if should_add_frame:
                local_tensorfs.append_frame() #loss是否可以定义在这（先这样改）
                #print('current pose loss: {}'.format(local_tensorfs.pose_loss))
                local_tensorfs.optimizer_Posenets_step(local_tensorfs.pose_loss, optimize_poses)
                train_dataset.activate_frames()
                print('appending frame: {}'.format(train_dataset.active_frames_bounds[1]))
                n_added_frames += 1
                #learned_pose[n_added_frames + local_tensorfs.n_init_frames - 1] = torch.inverse(local_tensorfs.w2c_list[n_added_frames + local_tensorfs.n_init_frames - 1]).cpu()#放cpu里

                last_add_iter = iteration
        C2W = local_tensorfs.get_cam2world().detach()
        for i in range(train_dataset.active_frames_bounds[1]):
            learned_pose[i] = torch.eye(4, 4)
            learned_pose[i][:3, :] = C2W[i].cpu()
        
        #（方法1）if iteration % 10 = 0: 单独对posenet所有active的frame全部做一次forward，算一个loss出来，优化一次（optimize，backward）
        '''
        if iteration > 0 and iteration % 100 == 0 and not should_add_frame:
            #pose_temp_pred_list = []
            #pose_temp_loss_list = []
            for i in range(train_dataset.active_frames_bounds[0] + 1, train_dataset.active_frames_bounds[1]):
                poseData_temp = local_tensorfs.pose_dataloader[0].dataset[i]
                pose_temp_pred = local_tensorfs.Posenets(poseData_temp)
                pose_temp_loss = local_tensorfs.Posenets.loss(pose_temp_pred, poseData_temp)['total'].to(args.device)
                T = local_tensorfs.Posenets.get_T()
                T = T.detach().to(args.device)
                local_tensorfs.w2c_list[i] = torch.matmul(T, local_tensorfs.w2c_list[i-1])
                local_tensorfs.c2w_list[i] = torch.inverse(local_tensorfs.w2c_list[i].to(args.device))[:3, :]
                local_tensorfs.optimizer_Posenets_step(pose_temp_loss, optimize_poses=True)
                #pose_temp_loss_list.append(local_tensorfs.Posenets.loss(pose_temp_pred, poseData_temp)['total'].to(args.device))
            #pose_temp_tensor = torch.stack(pose_temp_loss_list)
            #pose_temp_loss = torch.mean(pose_temp_tensor)
            #local_tensorfs.optimizer_Posenets_step(pose_temp_loss, optimize_poses=True)
        '''
        # Add new RF 添加新rf
        if can_add_rf:
            if train_dataset.has_left_frames():
                local_tensorfs.append_rf(n_added_frames)
                n_added_frames = 0
                last_add_rf_iter = iteration

                # Remove supervising frames
                training_frames = (local_tensorfs.blending_weights[:, -1] > 0)
                train_dataset.deactivate_frames(
                    np.argmax(training_frames.cpu().numpy(), axis=0))
            else:
                training = False
        ## Log
        loss = loss.detach().item()

        writer.add_scalar(
            "train/density_app_plane_lr",
            local_tensorfs.rf_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/basis_mat_lr",
            local_tensorfs.rf_optimizers[-1].param_groups[4]["lr"],
            global_step=iteration,
        )
        
        writer.add_scalar(
            "train/lr_r",
            local_tensorfs.r_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )
        writer.add_scalar(
            "train/lr_t",
            local_tensorfs.t_optimizers[-1].param_groups[0]["lr"],
            global_step=iteration,
        )
        '''
        writer.add_scalar(
            "train/lr_pose_0",
            local_tensorfs.pose_optimizers.param_groups[0]["lr"],
            global_step=iteration
        )
        writer.add_scalar(
            "train/lr_pose_1",
            local_tensorfs.pose_optimizers.param_groups[-1]["lr"],
            global_step=iteration
        )
        '''
        writer.add_scalar(
            "train/focal", local_tensorfs.focal(W), global_step=iteration
        )
        writer.add_scalar(
            "train/center0", local_tensorfs.center(W, H)[0].item(), global_step=iteration
        )
        writer.add_scalar(
            "train/center1", local_tensorfs.center(W, H)[1].item(), global_step=iteration
        )

        writer.add_scalar(
            "active_frames_bounds/0", train_dataset.active_frames_bounds[0], global_step=iteration
        )
        writer.add_scalar(
            "active_frames_bounds/1", train_dataset.active_frames_bounds[1], global_step=iteration
        )

        for index, blending_weights in enumerate(
            torch.permute(local_tensorfs.blending_weights, [1, 0])
        ):
            active_cam_indices = torch.nonzero(blending_weights)
            writer.add_scalar(
                f"tensorf_bounds/rf{index}_b0", active_cam_indices[0], global_step=iteration
            )
            writer.add_scalar(
                f"tensorf_bounds/rf{index}_b1", active_cam_indices[-1], global_step=iteration
            )

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            # All poses visualization
            poses_mtx = local_tensorfs.get_cam2world().detach().cpu()
            t_w2rf = torch.stack(list(local_tensorfs.world2rf), dim=0).detach().cpu()
            RF_mtx_inv = torch.cat([torch.stack(len(t_w2rf) * [torch.eye(3)]), -t_w2rf.clone()[..., None]], axis=-1)

            all_poses = torch.cat([poses_mtx,  RF_mtx_inv], dim=0)
            colours = ["C1"] * poses_mtx.shape[0] + ["C2"] * RF_mtx_inv.shape[0]
            img = draw_poses(all_poses, colours)
            writer.add_image("poses/all", (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32), iteration)

            # Get runtime 
            ips = min(args.progress_refresh_rate, iteration + 1) / (time.time() - start_time)
            writer.add_scalar(f"train/iter_per_sec", ips, global_step=iteration)
            print(f"Iteration {iteration:06d}: {ips:.2f} it/s")
            start_time = time.time()

        if (iteration % args.vis_every == args.vis_every - 1):
            poses_mtx = local_tensorfs.get_cam2world().detach()
            rgb_maps_tb, depth_maps_tb, gt_rgbs_tb, fwd_flow_cmp_tb, bwd_flow_cmp_tb, depth_err_tb, loc_metrics = render(
                test_dataset,
                poses_mtx,
                local_tensorfs,
                args,
                W=W // 2, H=H // 2,
                savePath=logfolder,
                save_frames=True,
                img_format="jpg",
                test=True,
                train_dataset=train_dataset,
                start=train_dataset.active_frames_bounds[0],
            )
            #加入ate、rte指标（nopenerf train line 335~337）
            
            with torch.no_grad():
                learned_poses = torch.stack(learned_pose)
                c2ws_est_aligned = align_ate_c2b_use_a2b(learned_poses, gt_poses)
            ate = compute_ATE(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
            rpe_trans, rpe_rot = compute_rpe(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
            #tqdm.write('{0:6d} ep: Train: ATE: {1:.3f} RPE_r: {2:.3f}'.format(epoch_it, ate, rpe_rot* 180 / np.pi))
            eval_dict = {
                'ate_trans': ate,
                'rpe_trans': rpe_trans*100,
                'rpe_rot': rpe_rot* 180 / np.pi
            }
            for l, num in eval_dict.items():
                writer.add_scalar('test/'+l, num, global_step=iteration)
            rpe_trans_reg = rpe_trans*100
            rpe_rot_reg = rpe_rot * 180 / np.pi
            print('current ATE: {}'.format(ate))
            print('current RPE_trans: {}'.format(rpe_trans_reg))
            print('current RPE_rot: {}'.format(rpe_rot_reg))
            

            if len(loc_metrics.values()):
                metrics.update(loc_metrics)
                mses = [metric["mse"] for metric in metrics.values()]
                writer.add_scalar(
                    f"test/PSNR", -10.0 * np.log(np.array(mses).mean()) / np.log(10.0), 
                    global_step=iteration
                )
                psnr = -10.0 * np.log(np.array(mses).mean()) / np.log(10.0)
                print('current PSNR: {}'.format(psnr))
                loc_mses = [metric["mse"] for metric in loc_metrics.values()]
                writer.add_scalar(
                    f"test/local_PSNR", -10.0 * np.log(np.array(loc_mses).mean()) / np.log(10.0), 
                    global_step=iteration
                )
                ssim = [metric["ssim"] for metric in metrics.values()]
                writer.add_scalar(
                    f"test/ssim", np.array(ssim).mean(), 
                    global_step=iteration
                )
                ssim_mean = np.array(ssim).mean()
                print('current SSIM: {}'.format(ssim_mean))
                loc_ssim = [metric["ssim"] for metric in loc_metrics.values()]
                writer.add_scalar(
                    f"test/local_ssim", np.array(loc_ssim).mean(), 
                    global_step=iteration
                )
                lpips = [metric["lpips"] for metric in metrics.values()]
                writer.add_scalar(
                    f"test/lpips", np.array(lpips).mean(),
                    global_step=iteration
                )
                lpips_mean = np.array(lpips).mean()
                print('current LPIPS: {}'.format(lpips_mean))
                writer.add_images(
                    "test/rgb_maps",
                    torch.stack(rgb_maps_tb, 0),
                    global_step=iteration,
                    dataformats="NHWC",
                )
                writer.add_images(
                    "test/depth_map",
                    torch.stack(depth_maps_tb, 0),
                    global_step=iteration,
                    dataformats="NHWC",
                )
                writer.add_images(
                    "test/gt_maps",
                    torch.stack(gt_rgbs_tb, 0),
                    global_step=iteration,
                    dataformats="NHWC",
                )
                
                if len(fwd_flow_cmp_tb) > 0:
                    writer.add_images(
                        "test/fwd_flow_cmp",
                        torch.stack(fwd_flow_cmp_tb, 0)[..., None],
                        global_step=iteration,
                        dataformats="NHWC",
                    )
                    
                    writer.add_images(
                        "test/bwd_flow_cmp",
                        torch.stack(bwd_flow_cmp_tb, 0)[..., None],
                        global_step=iteration,
                        dataformats="NHWC",
                    )
                
                if len(depth_err_tb) > 0:
                    writer.add_images(
                        "test/depth_cmp",
                        torch.stack(depth_err_tb, 0)[..., None],
                        global_step=iteration,
                        dataformats="NHWC",
                    )

            with open(f"{logfolder}/checkpoints_tmp.th", "wb") as f:
                local_tensorfs.save(f)

        iteration += 1
    #save ckpts
    with open(f"{logfolder}/checkpoints.th", "wb") as f:
        local_tensorfs.save(f)
    #渲染图片，输出结果
    poses_mtx = local_tensorfs.get_cam2world().detach()
    render_frames(args, poses_mtx, local_tensorfs, logfolder, test_dataset=test_dataset, train_dataset=train_dataset)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    
    print(args)
    
    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

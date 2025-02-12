import torch
import numpy as np
import torch.nn as nn
import omegaconf
import logging
from torch.nn import functional as nnF
from copy import deepcopy
from models.geometry.losses import scaled_barron
from models.utils import masked_mean
from models.common import make_c2w, convert3x4_4x4
from models.utils import get_pose_model
from models.geometry.wrappers import Pose
from models.geometry.wrappers import Camera, CAMERAS

logger = logging.getLogger(__name__)

class LearnPose(nn.Module):
    def __init__(self, device, num_cams, learn_R, learn_t, cfg, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.device = device
        self.num_cams = num_cams
        #self.init_c2w = None
        #if init_c2w is not None:
            #self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)
        #self.r = nn.Parameter(torch.zeros(size=(100, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        #self.t = nn.Parameter(torch.zeros(size=(100, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

        ##Add
        extractorConf = {'name': 'unet', 'encoder': 'vgg16', 'decoder': [64, 64, 64, 32], 'output_scales': [0, 2, 4],\
                        'output_dim': [32, 128, 128], 'freeze_batch_normalization': False, 'do_average_pooling': False,\
                        'compute_uncertainty': True, 'checkpointed': True}
        optimizerConf = {'name': 'learned_optimizer', 'num_iters': 15, 'pad': 3, 'lambda_': 0.01, 'verbose': False, \
                         'loss_fn': 'scaled_barron(0, 0.1)', 'no_conditions': True, 'jacobi_scaling': False,\
                         'sqrt_diag_damping': False, 'bound_confidence': True, 'learned_damping': True,\
                         'damping': {'type': 'constant'}}
        extractorConf = omegaconf.OmegaConf.create(extractorConf)
        optimizerConf = omegaconf.OmegaConf.create(optimizerConf)

        self.extractor = get_pose_model('unet')(extractorConf)
        #Opt = get_pose_model('learned_optimizer')(optimizerConf)
        oconfs = [deepcopy(optimizerConf) for _ in self.extractor.scales]
        feature_dim = self.extractor.conf.output_dim
        if not isinstance(feature_dim, int):
            for d, oconf in zip(feature_dim, oconfs):
                with omegaconf.read_write(oconf):
                    with omegaconf.open_dict(oconf):
                        oconf.feature_dim = d
        self.optimizer = torch.nn.ModuleList(get_pose_model('learned_optimizer')(oconf) for oconf in oconfs)
    def forward(self, data):##输入cam_id->data
        cam_id = data['img.idx']
        cam_id = int(cam_id)
        #data['img.tgt_img'] = data['img.tgt_img'][np.newaxis, :]
        #data['img.tgt_img'] = torch.from_numpy(data['img.tgt_img'])
        data['img.tgt_img'] = torch.unsqueeze(data['img.tgt_img'], 0)
        #data['img.ref_imgs'] = data['img.ref_imgs'][np.newaxis, :]
        #data['img.ref_imgs'] = torch.from_numpy(data['img.ref_imgs'])
        data['img.ref_imgs'] = torch.unsqueeze(data['img.ref_imgs'], 0)
        #r = self.r[cam_id]  # (3, ) axis-angle
        #t = self.t[cam_id]  # (3, )
        #c2w = make_c2w(r, t)  # (4, 4)
         #learn a delta pose between init pose and target pose, if a init pose is provided
        #if self.init_c2w is not None:
            #c2w = c2w @ self.init_c2w[cam_id]

        ##Add  ##需要提供的数据：rgb,'camera',ref3d点坐标,r2q_init_T可以用c2w替代
        pred_tgt = self.extractor(data['img.tgt_img'])
        pred_ref = self.extractor(data['img.ref_imgs'])
        ##
        C = CAMERAS.split()
        name, camera_model, width, height = C[:4]
        params = np.array(C[4:], float)
        camera = Camera.from_colmap(dict(
            model=camera_model, params=params,
            width=int(width), height=int(height)))
        data['img.camera'] = camera #localrf中没用
        pred_camera_pyr = [data['img.camera'].scale(1/s) for s in self.extractor.scales]
        '''
        for i in range(len(self.extractor.scales)):
            pred_camera = {}
            pred_camera['size'] = data['img.camera']['size'] / self.extractor.scales[i]
            pred_camera['f'] = data['img.camera']['f'] / self.extractor.scales[i]
            pred_camera['c'] = data['img.camera']['c'] + 0.5 / self.extractor.scales[i] - 0.5
            pred_camera_pyr.append(pred_camera)
        p3d_ref = data['img.points3D']
        '''
        ##要设置为nopenerf原网络得到的rt吗
        #T_init = Pose.from_aa(r,t)
        T_init = Pose.from_4x4mat(np.eye(4, dtype=np.float32))
        self.T_init = T_init

        p3d_ref = data['img.points3D']

        pred_T_r2q_init = []
        pred_T_r2q_opt = []
        pred_out = []
        for i in reversed(range(len(self.extractor.scales))):
            F_ref = pred_ref['feature_maps'][i]
            F_tgt = pred_tgt['feature_maps'][i]
            cam_ref = pred_camera_pyr[i]
            cam_tgt = pred_camera_pyr[i]
            opt = self.optimizer[i]

            p2d_ref, visible = cam_ref.world2image(p3d_ref)
            p2d_ref = p2d_ref.cuda()
            F_ref, mask, _ = opt.interpolator(F_ref, p2d_ref)
            mask &= visible.cuda() ##按位与操作

            W_ref = pred_ref['confidences'][i]
            W_tgt = pred_tgt['confidences'][i]
            W_ref, _, _ = opt.interpolator(W_ref, p2d_ref)
            W_ref_tgt = (W_ref,W_tgt)

            ##Normalize
            F_ref = nnF.normalize(F_ref, dim=2)
            F_tgt = nnF.normalize(F_tgt, dim=1)

            T_opt, _ = opt(dict(p3D=p3d_ref, F_ref=F_ref, F_q=F_tgt, T_init=T_init,\
                                cam_q=cam_tgt, mask=mask, W_ref_q=W_ref_tgt))
            pred_T_r2q_init.append(T_init)
            pred_T_r2q_opt.append(T_opt)
            T_init = T_opt.detach()

        pred_out = dict(ref = pred_ref, tgt = pred_tgt, T_r2q_init = pred_T_r2q_init, T_r2q_opt = pred_T_r2q_opt)
        self.T_opt = pred_T_r2q_opt[2]

        return pred_out
    def get_t(self):
       return self.t

    def get_T(self):
        t = self.T_opt.t
        t44 = torch.eye(4)
        t44[:3, 3] = t
        R = self.T_opt.R
        R44 = torch.eye(4)
        R44[:3, :3] = R
        T = torch.matmul(t44, R44)
        return T

    def loss(self, pred, data):
        cam_q = data['img.camera']

        def project(T_r2q):
            return cam_q.world2image(T_r2q * data['img.points3D'])

        p2D_q_gt, mask = project(Pose.from_4x4mat(data['img.T_r2q_gt']))
        p2D_q_i, mask_i = project(self.T_init)
        mask = (mask & mask_i).float()

        too_few = torch.sum(mask, -1) < 10
        '''
        if torch.any(too_few):
            logger.warning(
                'Few points in batch '+str([
                    (data['scene'][i], data['ref']['index'][i].item(),
                     data['query']['index'][i].item())
                    for i in torch.where(too_few)[0]]))
        '''
        def reprojection_error(T_r2q):
            p2D_q, _ = project(T_r2q)
            err = torch.sum((p2D_q_gt - p2D_q)**2, dim=-1)
            err = scaled_barron(1., 2.)(err)[0]/4
            err = masked_mean(err, mask, -1)
            return err

        num_scales = len(self.extractor.scales)
        success = None
        losses = {'total': 0.}
        for i, T_opt in enumerate(pred['T_r2q_opt']):
            err = reprojection_error(T_opt).clamp(max=50)#self.conf.clamp_error = 50
            loss = err / num_scales
            if i > 0:
                loss = loss * success.float()
            thresh = 3 * self.extractor.scales[-1-i]#self.conf.success_thresh = 3
            success = err < thresh
            losses[f'reprojection_error/{i}'] = err
            losses['total'] += loss
        losses['reprojection_error'] = err
        losses['total'] *= (~too_few).float()

        err_init = reprojection_error(pred['T_r2q_init'][0])
        losses['reprojection_error/init'] = err_init

        return losses

    
    


import os, sys
import cv2
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter
from kornia import create_meshgrid

from render_utils import *
from run_nerf_helpers import *
from load_llff import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
np.random.seed(1)
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--fixed_view", action='store_true',
                        help='time interpolation and fixed view')
    parser.add_argument("--no_fixed", action='store_true',
                        help='time interpolation and view interpolation')
    parser.add_argument("--fixed_time", action='store_true',
                        help='fixed time and view interpolation')
    parser.add_argument("--image_size", type=int, default=272,
                        help='rescaled resolution for training')
    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*128,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*128,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--weight_net_width", type=int, default=128,
                        help='channels in weight network')
    parser.add_argument("--dist_encoder_width", type=int, default=128,
                        help='channels in distribution encoder')
    parser.add_argument("--dist_dim", type=int, default=128,
                        help='dimension of target latent distribution')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_test", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--target_idx", type=int, default=10,
                        help='target_idx')
    parser.add_argument("--num_extra_sample", type=int, default=512, 
                        help='num_extra_sample')
    parser.add_argument("--use_motion_mask", action='store_true', 
                        help='use motion segmentation mask for hard-mining data-driven initialization')

    parser.add_argument("--lambda_depth", type=float, default=0.05,
                        help='weight of depth loss')
    parser.add_argument("--lambda_target_flow", type=float, default=0.02,
                        help='weight of warming up flow loss')
    parser.add_argument("--lambda_reg_flow", type=float, default=0.1,
                        help='weight of  flow regularization loss')
    parser.add_argument("--lambda_cons", type=float, default=0.1, 
                        help='weight of flow consistence loss')
    parser.add_argument("--lambda_dist", type=float, default=0.01,
                        help='weight of distribution loss')
    parser.add_argument("--lambda_w", type=float, default=0.5,
                        help='weight of occlusion weight loss')

    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 1000')
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)
    parser.add_argument("--save_epoch", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, depths, masks, poses, bds, \
        render_poses, ref_c2w, motion_coords = load_llff_data(args.datadir, 
                                                            args.start_frame, args.end_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.image_size)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        #print('Loaded llff', images.shape, render_poses.shape, hwf)
        i_train = np.array([i for i in np.arange(int(images.shape[0]))])
        #i_train = np.array([i * 2 for i in np.arange(int(images.shape[0]) // 2)])

        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.
        #print('NEAR FAR', near, far)
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    basedir = args.basedir
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.fixed_time:
        print('fixed time and view interpolation')
        render_poses = torch.Tensor(render_poses).to(device)
        num_img = float(poses.shape[0])
        img_idx_embed = target_idx/float(num_img) * 2. - 1.0
        testsavedir = os.path.join(basedir, expname, f"fixed_time_frame{target_idx}_iter{global_step}")
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_fixed_time(render_poses, img_idx_embed, num_img, hwf,
                               args.chunk, render_kwargs_test,
                               gt_imgs=images, savedir=testsavedir, 
                               render_factor=args.render_factor)
        return
    '''
    if args.fixed_view:
        print('time interpolation and fixed view')
        num_img = float(poses.shape[0])
        ref_c2w = torch.Tensor(ref_c2w).to(device)
        testsavedir = os.path.join(basedir, expname, f"fixed_view_frame{target_idx}_iter{global_step}")
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_fixed_view(ref_c2w, num_img, hwf,
                            args.chunk, render_kwargs_test, 
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor,
                            target_idx=target_idx)
            return
    '''

    if args.no_fixed:
        print('time interpolation and view interpolation')
        render_poses = poses #torch.Tensor(poses).to(device)
        bt_poses = create_bt_poses(hwf) 
        bt_poses = bt_poses * 10
        images = torch.Tensor(images)  # .to(device)
        testsavedir = os.path.join(basedir, expname, f"no_fixed_frame{target_idx}_iter{global_step}")
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            render_no_fixed(depths, render_poses, bt_poses,
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=target_idx)
        return

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
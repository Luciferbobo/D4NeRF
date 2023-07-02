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
    parser.add_argument("--render_lockcam_slowmo", action='store_true', 
                        help='render fixed view + slowmo')
    parser.add_argument("--render_slowmo_bt", action='store_true', 
                        help='render space-time interpolation')
    parser.add_argument("--render_bt", action='store_true',
                        help='render bullet time')
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

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    images = torch.Tensor(images)#.to(device)
    depths = torch.Tensor(depths)#.to(device)

    poses = torch.Tensor(poses).to(device)

    N_iters = 60 * 10000
    uv_grid = create_meshgrid(H, W, normalized_coordinates=False)[0].cuda() # (H, W, 2)

    num_img = float(images.shape[0])
    decay_iteration = max(args.decay_iteration, 
                          args.end_frame - args.start_frame)
    decay_iteration = min(decay_iteration, 250)

    print('Training Iters:', N_iters)
    print('Training Views:', i_train)

    start_time = time.time()
    for i in range(start, N_iters):

        flow_line = (i+1) % 2

        img_i = np.random.choice(i_train)

        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()

        target = images[img_i].cuda()
        pose = poses[img_i, :3,:4]
        depth_gt = depths[img_i].cuda()
        hard_coords = torch.Tensor(motion_coords[img_i]).cuda()

        if img_i == 0:
            flow_fwd, fwd_mask = read_optical_flow(args.datadir, img_i, 
                                                args.start_frame, fwd=True)
            flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
        elif img_i == num_img - 1:
            flow_bwd, bwd_mask = read_optical_flow(args.datadir, img_i, 
                                                args.start_frame, fwd=False)
            flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
        else:
            flow_fwd, fwd_mask = read_optical_flow(args.datadir, 
                                                img_i, args.start_frame, 
                                                fwd=True)
            flow_bwd, bwd_mask = read_optical_flow(args.datadir, 
                                                img_i, args.start_frame, 
                                                fwd=False)

        flow_fwd = torch.Tensor(flow_fwd).cuda()
        fwd_mask = torch.Tensor(fwd_mask).cuda()
        flow_bwd = torch.Tensor(flow_bwd).cuda()
        bwd_mask = torch.Tensor(bwd_mask).cuda()
        flow_fwd = flow_fwd + uv_grid
        flow_bwd = flow_bwd + uv_grid

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

            if args.use_motion_mask and i < decay_iteration * 1000:

                num_extra_sample = args.num_extra_sample
                select_inds_hard = np.random.choice(hard_coords.shape[0], 
                                                    size=[min(hard_coords.shape[0], 
                                                        num_extra_sample)], 
                                                    replace=False)  # (N_rand,)
                select_inds_all = np.random.choice(coords.shape[0], 
                                                size=[N_rand], 
                                                replace=False)  # (N_rand,)

                select_coords_hard = hard_coords[select_inds_hard].long()
                select_coords_all = coords[select_inds_all].long()

                select_coords = torch.cat([select_coords_all, select_coords_hard], 0)

            else:
                select_inds = np.random.choice(coords.shape[0], 
                                            size=[N_rand], 
                                            replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
            
            rays_o = rays_o[select_coords[:, 0], 
                            select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], 
                            select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_rgb = target[select_coords[:, 0], 
                                select_coords[:, 1]]  # (N_rand, 3)

            target_depth = depth_gt[select_coords[:, 0], 
                                select_coords[:, 1]]

            target_of_fwd = flow_fwd[select_coords[:, 0], 
                                     select_coords[:, 1]]
            target_fwd_mask = fwd_mask[select_coords[:, 0], 
                                     select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)

            target_of_bwd = flow_bwd[select_coords[:, 0], 
                                     select_coords[:, 1]]
            target_bwd_mask = bwd_mask[select_coords[:, 0], 
                                     select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)

        img_idx_embed = img_i/num_img * 2. - 1.0


        ret = render(img_idx_embed,
                     flow_line,
                     num_img, H, W, focal, 
                     chunk=args.chunk, 
                     rays=batch_rays,
                     verbose=i < 10, retraw=True,
                     **render_kwargs_train)

        pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
        pose_prev = poses[max(img_i - 1, 0), :3,:4]

        render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, 
                                                            pose, pose_prev, 
                                                            H, W, focal, 
                                                            ret)
        optimizer.zero_grad()
        weight_map_post = ret['prob_map_post']
        weight_map_prev = ret['prob_map_prev']

        weight_post = 1. - ret['raw_prob_ref2post']
        weight_prev = 1. - ret['raw_prob_ref2prev']
        reg_flow_loss = args.lambda_reg_flow * (torch.mean(torch.abs(ret['raw_prob_ref2prev'])) \
                                + torch.mean(torch.abs(ret['raw_prob_ref2post'])))

        if i <= decay_iteration * 1000:
            # dynamic rendering loss
            rec_loss = img2mse(ret['rgb_map_ref_dy'], target_rgb)
            rec_loss += compute_mse(ret['rgb_map_post_dy'], 
                                       target_rgb, 
                                       weight_map_post.unsqueeze(-1))
            rec_loss += compute_mse(ret['rgb_map_prev_dy'], 
                                       target_rgb, 
                                       weight_map_prev.unsqueeze(-1))
        else:
            weights_map_dd = ret['weights_map_dd'].unsqueeze(-1).detach()

            rec_loss = compute_mse(ret['rgb_map_ref_dy'], 
                                      target_rgb, 
                                      weights_map_dd)
            rec_loss += compute_mse(ret['rgb_map_post_dy'], 
                                       target_rgb, 
                                       weight_map_post.unsqueeze(-1) * weights_map_dd)
            rec_loss += compute_mse(ret['rgb_map_prev_dy'], 
                                       target_rgb, 
                                       weight_map_prev.unsqueeze(-1) * weights_map_dd)

        rec_loss += img2mse(ret['rgb_map_ref'][:N_rand, ...], 
                               target_rgb[:N_rand, ...])

        reg_flow_loss += compute_mae(ret['raw_sf_ref2post'],
                                    -ret['raw_sf_post2ref'],
                                    weight_post.unsqueeze(-1), dim=3)
        reg_flow_loss += compute_mae(ret['raw_sf_ref2prev'],
                                    -ret['raw_sf_prev2ref'],
                                    weight_prev.unsqueeze(-1), dim=3)

        render_sf_ref2prev = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2prev'], -1)
        render_sf_ref2post = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2post'], -1)

        reg_flow_loss += args.lambda_reg_flow * (torch.mean(torch.abs(render_sf_ref2prev)) \
                                    + torch.mean(torch.abs(render_sf_ref2post))) 

        divsor = i // (decay_iteration * 1000)

        decay_rate = 10

        lambda_depth = args.lambda_depth/(decay_rate ** divsor)
        lambda_target_flow = args.lambda_target_flow/(decay_rate ** divsor)


        depth_loss_ = lambda_depth * depth_loss(ret['depth_map_ref_dy'], -target_depth)


        if img_i == 0:
            target_flow_loss = lambda_target_flow * compute_mae(render_of_fwd,
                                        target_of_fwd, 
                                        target_fwd_mask)
        elif img_i == num_img - 1:
            target_flow_loss = lambda_target_flow * compute_mae(render_of_bwd,
                                        target_of_bwd, 
                                        target_bwd_mask)
        else:
            target_flow_loss = lambda_target_flow * compute_mae(render_of_fwd,
                                        target_of_fwd, 
                                        target_fwd_mask)
            target_flow_loss += lambda_target_flow * compute_mae(render_of_bwd,
                                        target_of_bwd, 
                                        target_bwd_mask)

        cons_loss_ = args.lambda_cons * (cons_loss(ret['raw_pts_ref'],
                                            ret['raw_pts_post'],
                                            H, W, focal) \
                                + cons_loss(ret['raw_pts_ref'],
                                            ret['raw_pts_prev'],
                                            H, W, focal))
        cons_loss_ += args.lambda_cons * cons_loss_bi(ret['raw_pts_ref'],
                                                ret['raw_pts_post'],
                                                ret['raw_pts_prev'],
                                                H, W, focal)
        cons_loss_ += args.lambda_cons * cons_loss_bi(ret['raw_pts_ref'],
                                                ret['raw_pts_post'],
                                                ret['raw_pts_prev'],
                                                H, W, focal)

        if flow_line:
            cons_loss_ += args.lambda_cons * cons_loss_bi(ret['raw_pts_prev'],
                                                    ret['raw_pts_ref'],
                                                    ret['raw_pts_pp'],
                                                    H, W, focal)

        else:
            cons_loss_ += args.lambda_cons * cons_loss_bi(ret['raw_pts_post'],
                                                    ret['raw_pts_pp'],
                                                    ret['raw_pts_ref'],
                                                    H, W, focal)

        weight_loss_ = args.lambda_w * (weight_loss(ret['w_s1']) + weight_loss(ret['w_s1']) + \
                      2e-3 * torch.mean(-ret['raw_blend_w'] * torch.log(ret['raw_blend_w'] + 1e-8)))
        dist_loss_ = args.lambda_dist * vae_loss(ret['vae_mu'], ret['vae_logvar'])

        loss = rec_loss + target_flow_loss + \
               cons_loss_ + reg_flow_loss + \
               depth_loss_ + weight_loss_ + \
               dist_loss_

        if i % 100 == 0:
            print('rec_loss ', rec_loss.item(),
                  'depth_loss ', depth_loss_.item(),
                  'target_flow_loss ', target_flow_loss.item())
            print('cons_loss ', cons_loss_.item(), 'reg_loss ', reg_flow_loss.item())
            print('weight_loss ', weight_loss_.item(), 'dist_loss ', dist_loss_.item())

        loss.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        #new_lrate = args.lrate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        if i % 100 == 0:
            end_time = time.time()
            time_cost = end_time - start_time
            start_time = time.time()
            print("lr: %.8f" % new_lrate)
            print(f"Iter: {global_step}, Loss: {loss}, Time: {time_cost}")


        if i%args.save_epoch==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

            if args.N_importance > 0:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'network_w': render_kwargs_train['network_w'].state_dict(),
                    'network_vae': render_kwargs_train['network_vae'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                    'network_w': render_kwargs_train['network_w'].state_dict(),
                    'network_vae': render_kwargs_train['network_vae'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)

            print('Saved checkpoints at', path)


        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
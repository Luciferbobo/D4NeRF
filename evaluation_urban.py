
import cv2
import os, sys
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import models


import math
from render_utils import *
from run_nerf_helpers import *

from load_llff import load_nvidia_data
import skimage
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


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


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2)**2 * mask) / num_valid
    
    if mse == 0:
        return 0 #float('inf')

    return 10 * math.log10(1./mse)


def calculate_ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    _, ssim_map = structural_similarity(img1, img2, multichannel=True, full=True)
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid



def evaluation(epoch=0):

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        images, poses, bds, render_poses = load_nvidia_data(args.datadir, 
                                                            args.start_frame, args.end_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.image_size)


        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]

        views_all = np.array([i for i in np.arange(images.shape[0])])
        views_train = np.array([i * 2 for i in np.arange(images.shape[0] // 2)])
        views_eval = np.array([i for i in range(len(views_all)) if i not in views_train])

        #if images.shape[0] % 2 == 0:
        #    views_eval=views_eval[:-1]
        #    images = images[:-1]

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.9 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
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
    render_kwargs_train, render_kwargs_test, \
        start, grad_vars, optimizer = create_nerf(args)

    epoch = start - 1
    print('Test result from epoch: ', epoch)

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    num_img = float(images.shape[0])
    poses = torch.Tensor(poses).to(device)

    with torch.no_grad():

        model = models.PerceptualLoss(model='net-lin',net='alex',
                                      use_gpu=True,version=0.1)
        count = 0
        total_psnr = 0.
        total_ssim = 0.
        total_lpips = 0.
        t = time.time()

        # for each time step

        print('views_eval', views_eval)

        for img_i in views_eval:

            img_idx_embed = img_i/num_img * 2. - 1.0
            print(img_i, img_idx_embed)

            # for each target viewpoint

            for camera_i in range(1):

                print(time.time() - t,'s')
                t = time.time()

                c2w = poses[img_i]
                ret = render_test(img_idx_embed, 0,
                             num_img, 
                             H, W, focal, 
                             chunk=1024*16, c2w=c2w[:3,:4], 
                             **render_kwargs_test)


                rgb = ret['rgb_map_ref'].cpu().numpy()#.append(ret['rgb_map_ref'].cpu().numpy())

                gt_img_path = os.path.join(args.datadir, 
                                        'images_eval',
                                        '%05d'%img_i, 
                                        'eval%02d.png'%(camera_i + 1))

                print('gt_img_path ', gt_img_path)

                gt_img = cv2.imread(gt_img_path)[:, :, ::-1]
                gt_img = cv2.resize(gt_img, 
                                    dsize=(rgb.shape[1], rgb.shape[0]), 
                                    interpolation=cv2.INTER_AREA)
                gt_img = np.float32(gt_img) / 255

                psnr = peak_signal_noise_ratio(gt_img, rgb)
                ssim = structural_similarity(gt_img, rgb,
                                                    multichannel=True)

                gt_img_0 = im2tensor(gt_img).cuda()
                rgb_0 = im2tensor(rgb).cuda()

                lpips = model.forward(gt_img_0, rgb_0)
                lpips = lpips.item()
                print(psnr, ssim, lpips)

                total_psnr += psnr
                total_ssim += ssim
                total_lpips += lpips
                count += 1


        mean_psnr = total_psnr / count
        mean_ssim = total_ssim / count
        mean_lpips = total_lpips / count

        print('mean_psnr ', mean_psnr)
        print('mean_ssim ', mean_ssim)
        print('mean_lpips ', mean_lpips)

        f_i = os.path.join(basedir, expname, 'eval_result.txt')
        with open(f_i, 'a') as file:
            file.write('####### Epoch='+str(epoch)+'\n')
            file.write('mean_psnr:'+str(mean_psnr)+' mean_ssim:'+str(mean_ssim)+' mean_lpips:'+str(mean_lpips)+'\n')


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    evaluation()

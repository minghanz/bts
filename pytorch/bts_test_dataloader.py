import torch
from bts_dataloader import *
from bts_main_argparse import parse_args_main

from bts import * # if not loading from checkpoint

import sys, os
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, "../../"))
from c3d.utils.cam_proj import CamProj
from c3d.c3d_loss import C3DLoss
from c3d.utils.cam import lidar_to_depth
from c3d.utils.vis import vis_depth_np, overlay_dep_on_rgb_np, vis_depth, overlay_dep_on_rgb, dep_img_bw, vis_depth_err, uint8_np_from_img_tensor, vis_pts_dist, comment_on_img
from c3d.utils.io import save_tensor_to_img

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
def main_worker(args, args_rest):
    ## set manual seed to make it reproducible
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # mode = 'train'
    mode = 'online_eval'
    ## input
    cam_proj_model = CamProj(args.data_path, batch_size=args.batch_size)
    if mode == 'train':
        dataloader = BtsDataLoader(args, 'train', cam_proj=cam_proj_model)
    else:
        dataloader = BtsDataLoader(args, 'online_eval', data_source='kitti_depth', cam_proj=cam_proj_model)

    ## model
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    # C3D module
    c3d_model = C3DLoss(seq_frame_n=args.seq_frame_n_c3d)
    c3d_model.parse_opts(args_rest)
    c3d_model.cuda()
    
    for step, sample_batched in enumerate(dataloader.data):
        # batch_vis_input(step, sample_batched)
        # batch_vis_input_eval(step, sample_batched)
        batch_vis_output(step, sample_batched, model, mode=mode, c3d_model=c3d_model)
        print('step', step)

def batch_vis_output(step, sample_batched, model, mode, c3d_model=None):
    ## input
    image = sample_batched['image'].cuda()
    focal = sample_batched['focal'].cuda()
    ## Predict
    lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, _ = model(image, focal)

    ## scale the image and depth estimation to have the same size as depth_gt
    depth_gt = sample_batched['depth'].cuda()
    if mode == 'online_eval':
        has_valid_depth = sample_batched['has_valid_depth']
        if not has_valid_depth:
            # print('Invalid depth. continue.')
            return
        depth_est = scale_image(depth_est, depth_gt.shape[3], depth_gt.shape[2], torch_mode=True, nearest=False, raw_float=True, align_corner=False)
        image = scale_image(image, depth_gt.shape[3], depth_gt.shape[2], torch_mode=True, nearest=False, raw_float=True, align_corner=False)

    batch_size = image.shape[0]
    batch_vis_depth_diff(batch_size, step, image, depth_est, depth_gt)
    # batch_vis_pts_dist(batch_size, step, image, depth_est, depth_gt, sample_batched, c3d_model)

def batch_vis_depth_diff(batch_size, step, image, depth_est, depth_gt):
    '''visualize the difference in depth estimation ang GT'''
    for ib in range(batch_size):
        dep_diff = vis_depth_err(depth_est[ib], depth_gt[ib])
        # dep_diff = np.ones_like(dep_diff)*255
        filename="{}_{}.jpg".format(step, ib)
        unnormalized_img_i = inv_normalize(image[ib])
        img_vis_i = uint8_np_from_img_tensor(unnormalized_img_i)
        dep_rgb_vis_i_np = overlay_dep_on_rgb_np(dep_diff, img_vis_i, path='vis_intr_dep_c3d', name=filename )

def batch_vis_pts_dist(batch_size, step, image, depth_est, depth_gt, sample_batched, c3d_model):
    
    cam_info = sample_batched['cam_info'].cuda()
    depth_mask = sample_batched['mask'].cuda(args.gpu, non_blocking=True)
    depth_gt_mask = sample_batched['mask_gt'].cuda(args.gpu, non_blocking=True)

    inp = c3d_model(image, depth_est, depth_gt, depth_mask, depth_gt_mask, cam_info ) 
    dist_stat = c3d_model.pc3ds["gt"].grid.feature['dist_stat']

    for ib in range(batch_size):
        unnormalized_img_i = inv_normalize(image[ib])
        img_vis_i = uint8_np_from_img_tensor(unnormalized_img_i)

        img_min, img_max, img_mean, scale_min, scale_max, scale_mean = vis_pts_dist(dist_stat[ib])
        dep_rgb_vis_i_np = overlay_dep_on_rgb_np(img_min, comment_on_img(img_vis_i, item_name='min', num=scale_min), path='vis_intr', name="{}_{}_min.jpg".format(step, ib) )
        dep_rgb_vis_i_np = overlay_dep_on_rgb_np(img_max, comment_on_img(img_vis_i, item_name='max', num=scale_max), path='vis_intr', name="{}_{}_max.jpg".format(step, ib) )
        dep_rgb_vis_i_np = overlay_dep_on_rgb_np(img_mean, comment_on_img(img_vis_i, item_name='mean', num=scale_mean), path='vis_intr', name="{}_{}_mean.jpg".format(step, ib) )
        

def batch_vis_input_eval(step, sample_batched):
    depth_gt = sample_batched['depth']
    image = sample_batched['image']
    has_valid_depth = sample_batched['has_valid_depth']
    batch_size = image.shape[0]
    for ib in range(batch_size):
        if has_valid_depth[ib] == False:
            continue
        unnormalized_img_i = inv_normalize(image[ib])
        depth_vis_i = vis_depth(depth_gt[ib])
        filename="{}_{}.jpg".format(step, ib)
        dep_rgb_vis_i = overlay_dep_on_rgb(depth_vis_i, unnormalized_img_i, path='vis_intr', name=filename, overlay=False)
        # filename_bw="{}_{}_bw.png".format(step, ib)
        # dep_img_bw(depth_vis_i, path='vis_intr', name=filename_bw)

def batch_vis_input(step, sample_batched):
    ## take out image, velo, cam_info
    ## project velo to image to test that the intrinsics are correct
    velo = sample_batched['velo']
    image = sample_batched['image_ori']
    cam_info = sample_batched['cam_info']
    depth_gt = sample_batched['depth']
    batch_size = image.shape[0]
    for ib in range(batch_size):
        depth_i_np = lidar_to_depth(velo[ib], cam_info.P_cam_li, K_unit=None, im_shape=(cam_info.height, cam_info.width), K_ready=cam_info.K[ib], torch_mode=True)
        depth_vis_i_np = vis_depth_np(depth_i_np)
        filename="{}_{}.jpg".format(step, ib)
        dep_rgb_vis_i_np = overlay_dep_on_rgb_np(depth_vis_i_np, image[ib].cpu().numpy().transpose(1,2,0), path='vis_intr', name=filename )
        depth_vis_i = vis_depth(depth_gt[ib])
        filename="{}_{}_dataset.jpg".format(step, ib)
        dep_rgb_vis_i = overlay_dep_on_rgb(depth_vis_i, image[ib], path='vis_intr', name=filename )
        
    image_scaled = sample_batched['image_ori_scaled']
    cam_info_scaled = sample_batched['cam_info_scaled']
    for ib in range(batch_size):
        depth_i_np = lidar_to_depth(velo[ib], cam_info_scaled.P_cam_li, K_unit=None, im_shape=(cam_info_scaled.height, cam_info_scaled.width), K_ready=cam_info_scaled.K[ib], torch_mode=True)
        depth_vis_i_np = vis_depth_np(depth_i_np)
        filename="{}_scaled_{}.jpg".format(step, ib)
        dep_rgb_vis_i_np = overlay_dep_on_rgb_np(depth_vis_i_np, image_scaled[ib].cpu().numpy().transpose(1,2,0), path='vis_intr', name=filename )


    ## visualize all inputs
    for iside in range(sample_batched['image_side_scaled'].shape[1]):
        save_tensor_to_img(sample_batched['image_side_scaled'][:, iside], filename=os.path.join('vis_intr', '{}_{}_side_scaled'.format(step, iside)), mode='rgb')
        save_tensor_to_img(sample_batched['image_side'][:, iside], filename=os.path.join('vis_intr', '{}_{}_side'.format(step, iside)), mode='rgb')
        

    ## check that scale-crop is the same as crop-scale
    xy_crop = sample_batched['xy_crop']
    x_st, y_st, x_si, y_si = [elem.numpy().astype(int) for elem in xy_crop]
    xy_crop_scaled = sample_batched['xy_crop_scaled']
    x_start, y_start, x_size, y_size = [elem.numpy().astype(int) for elem in xy_crop_scaled]
    side_scale_crops = []
    side_crop_scales = []
    batch_size = sample_batched['image_side_scaled'].shape[0]

    compare_255 = False
    calc_int = False     # use pil.image.resize has smaller error then F.interpolate. In F.interpolate, align_corner=False has smaller error. 
    ### Result shows that the maximum difference between scale-crop image and crop-scale image is about 10-20, avg is about 0.02 in 255 scale using pil.image.resize. 
    ### This should mean that we can treat the two interchangably. 
    if not os.path.exists('vis_intr'):
        os.mkdir('vis_intr')
    for ib in range(batch_size):
        ## scaled -> cropped
        side_scale_crop = sample_batched['image_side_scaled'][ib, 0, :, y_start[ib]:y_start[ib]+y_size[ib], x_start[ib]:x_start[ib]+x_size[ib] ]
        if compare_255:
            side_scale_crop = torch.round(side_scale_crop*255)

        ## cropped -> scaled
        side_crop = sample_batched['image_side'][ib, 0, :, y_st[ib]:y_st[ib]+y_si[ib], x_st[ib]:x_st[ib]+x_si[ib] ]
        if not calc_int:
            ## use torch.nn.functional.interpolate for resizing with float numbers
            side_crop_scale = scale_image(side_crop, x_size[ib], y_size[ib], torch_mode=True, nearest=False, raw_float=True, align_corner=False )
            if compare_255:
                side_crop_scale = torch.round(side_crop_scale * 255)
        else:
            ## use pil.image.resize for resizing with integer numbers
            side_crop_np = side_crop.numpy()
            side_crop_scale_pil = scale_image(side_crop_np, x_size[ib], y_size[ib], torch_mode=False, nearest=False, raw_float=False, align_corner=False )
            if compare_255:
                side_crop_scale_np = np.array(side_crop_scale_pil).transpose(2, 0, 1).astype(np.float32)
                side_crop_scale = torch.from_numpy(side_crop_scale_np)
            else:
                side_crop_scale_np = np.array(side_crop_scale_pil).transpose(2, 0, 1).astype(np.float32) / 255
                side_crop_scale = torch.from_numpy(side_crop_scale_np)

        print('ib:', ib)
        print(torch.abs(side_scale_crop-side_crop_scale).max() )
        print(torch.abs(side_scale_crop-side_crop_scale).mean() )
        print(side_scale_crop.max(), side_crop_scale.max() )
        # print(torch.all(torch.eq(side_scale_crop, side_crop_scale)))
        side_scale_crops.append(side_scale_crop)
        side_crop_scales.append(side_crop_scale)
    side_scale_crops = torch.stack(side_scale_crops, dim=0)
    side_crop_scales = torch.stack(side_crop_scales, dim=0)
    save_tensor_to_img(side_scale_crops, filename=os.path.join('vis_intr', '{}_side_scaled_crop'.format(step)), mode='rgb')
    save_tensor_to_img(side_crop_scales, filename=os.path.join('vis_intr', '{}_side_crop_scaled'.format(step)), mode='rgb')

        



if __name__ == '__main__':
    if sys.argv.__len__() == 2:
        args, args_rest = parse_args_main()
    else:
        args = parse_args_main()
    
    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    main_worker(args, args_rest)
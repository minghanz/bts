import torch
from bts_dataloader import *
from bts_main_argparse import parse_args_main

from bts import * # if not loading from checkpoint

import sys, os
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, "../../"))
from c3d.utils.cam_proj import CamProj
from c3d.utils.cam import lidar_to_depth
from c3d.utils.vis import vis_depth_np, overlay_dep_on_rgb_np, vis_depth, overlay_dep_on_rgb
from c3d.utils.io import save_tensor_to_img

def main_worker(args):
    ## set manual seed to make it reproducible
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    cam_proj_model = CamProj(args.data_path, batch_size=args.batch_size)
    dataloader = BtsDataLoader(args, 'train', cam_proj=cam_proj_model)
    
    for step, sample_batched in enumerate(dataloader.data):
        process_batch(step, sample_batched)
        print('step', step)

def process_batch(step, sample_batched):
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

    main_worker(args)
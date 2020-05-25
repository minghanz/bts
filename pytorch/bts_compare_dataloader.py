import torch
from bts_dataloader import BtsDataLoader
from bts_dataloader_old import BtsDataLoaderOld
from bts_main_argparse import parse_args_main

from bts import * # if not loading from checkpoint

import sys, os
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, "../../"))
from c3d.utils.cam_proj_old import CamProjOld
from c3d.utils.cam_proj import CamProj

from c3d.c3d_loss import C3DLoss
from c3d.utils_general.calib import lidar_to_depth
from c3d.utils_general.vis import vis_depth_np, overlay_dep_on_rgb_np, vis_depth, overlay_dep_on_rgb, dep_img_bw, vis_depth_err, uint8_np_from_img_tensor, vis_pts_dist, comment_on_img, save_np_to_img

from c3d.utils_general.dataset_read import DataReaderKITTI

from torchvision import transforms
import numpy as np

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

    mode = 'train'
    # mode = 'online_eval'

    ## input
    # DataReader
    data_meta_reader = DataReaderKITTI(data_root=args.data_path)        ### TODO: args.data_path_eval is not used

    # CamProj Module
    cam_proj_model = CamProj(data_meta_reader, batch_size=args.batch_size)
    if mode == 'train':
        dataloader = BtsDataLoader(args, 'train', data_meta_reader, cam_proj=cam_proj_model)
    else:
        dataloader = BtsDataLoader(args, 'online_eval', data_meta_reader, data_source='kitti_depth', cam_proj=cam_proj_model)

    cam_proj_model_old = CamProjOld(args.data_path, batch_size=args.batch_size)
    if mode == 'train':
        dataloader_old = BtsDataLoaderOld(args, 'train', cam_proj=cam_proj_model_old)
    else:
        dataloader_old = BtsDataLoaderOld(args, 'online_eval', data_source='kitti_depth', cam_proj=cam_proj_model_old)

    for trial in range(10):
        if mode == 'train':
            keys = list(dataloader_old.training_samples.lines_group.keys())
            idx_to_sample = dataloader_old.training_samples.lines_group[keys[0]][trial]
            sample0 = dataloader.training_samples[idx_to_sample]
            sample0_old = dataloader_old.training_samples[idx_to_sample]
        else:
            sample0 = dataloader.testing_samples[trial]
            sample0_old = dataloader_old.testing_samples[trial]

        ##  {'image': image_aug, 'image_ori':image, 'depth': depth_gt, 'focal': focal, 'mask': mask, 'mask_gt': mask_gt, 'xy_crop': xy_crop, 'T': T, 'ntp': sample_ntp, 'cam_ops': cam_ops}
        print(sample0['image'].max(), sample0_old['image'].max())
        print(sample0['depth'].max(), sample0_old['depth'].max())
        print(sample0['T'][:3].max(), sample0_old['T'][:3].max())
        print(sample0['ntp'].fid, sample0_old['frame'])


if __name__ == '__main__':
    if sys.argv.__len__() == 2:
        args, args_rest = parse_args_main()
    else:
        args = parse_args_main()
    
    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    main_worker(args, args_rest)
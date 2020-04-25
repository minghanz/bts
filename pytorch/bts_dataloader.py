# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random

from distributed_sampler_no_evenly_divisible import *

import cv2
from skimage.morphology import binary_dilation, binary_closing
from sampler_kitti import SamplerKITTI, Collate_Cfg, samp_from_seq, gen_samp_list, frame_line_mapping, gen_rand_crop_tensor_param, crop_for_perfect_scaling
# from bts_pre_intr import preload_K, load_velodyne_points, lidar_to_depth
import sys
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, "../../"))
from c3d.utils.dataset_kitti import preload_K, load_velodyne_points
from c3d.utils.cam import lidar_to_depth, scale_K, scale_from_size, crop_and_scale_K, scale_image, CamScale, CamCrop, CamRotate, scale_depth_from_lidar, crop_and_scale_depth_from_lidar

import re

from check_neighbor import process_line

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class BtsDataLoader(object):
    def __init__(self, args, mode, data_source=None, cam_proj=None):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode), data_source=data_source)
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
                
                self.data = DataLoader(self.training_samples, args.batch_size,
                                    shuffle=(self.train_sampler is None),
                                    num_workers=args.num_threads,
                                    pin_memory=True,
                                    sampler=self.train_sampler)
            else:
                # self.train_sampler = None
                self.train_sampler = SamplerKITTI(self.training_samples, args.batch_size, args.seq_frame_n_c3d, args.seq_frame_n_pho)   # to make sure all samples in a mini-batch have the same intrinsics
                self.collate_class = Collate_Cfg(args.input_width, args.input_height, args.seq_frame_n_c3d, args.other_scale, args.side_full_img, cam_proj=cam_proj)
                                    
                self.data = DataLoader(self.training_samples, 
                                    num_workers=args.num_threads,
                                    pin_memory=True,
                                    batch_sampler=self.train_sampler, 
                                    collate_fn=self.collate_class.collate_common_crop)

        ## for batch size 1, there is no need to use SamplerKITTI
        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode), data_source=data_source)
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=2,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode), data_source=data_source)
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, data_source=None):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        # self.is_for_online_eval = is_for_online_eval

        self.dilate_struct = np.ones((35, 35))

        # ## separate the lines to different dates
        # self.lines_in_date = {}
        # dates = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]
        # for date in dates:
        #     self.lines_in_date[date] = []
        
        # for i, line in enumerate(self.filenames):
        #     date = line.split('/')[0]
        #     assert date in dates
        #     self.lines_in_date[date].append(i)

        if mode =='train':
            ## process lines and retrive date, seq, side and frame
            self.line_idx = {}
            self.frame_idx = {}
            for i, line in enumerate(self.filenames):
                date, seq, side, frame = process_line(line)
                if date not in self.line_idx:
                    self.line_idx[date] = {}
                    self.frame_idx[date] = {}
                if (seq, side) not in self.line_idx[date]:
                    self.line_idx[date][(seq, side)] = []
                    self.frame_idx[date][(seq, side)] = []
                self.line_idx[date][(seq, side)].append(i)
                self.frame_idx[date][(seq, side)].append(frame)

            ### generate both-direction mapping
            self.frame2line, self.line2frame = frame_line_mapping(self.frame_idx, self.line_idx)

            ### for each (date, seq, side), get sample points, which should be sampled from
            frame_idxs_to_sample, line_idx_to_sample = samp_from_seq(self.frame_idx, self.line_idx, args.seq_frame_n_c3d, args.seq_frame_n_pho)
            self.lines_group = gen_samp_list(line_idx_to_sample, use_date_key=args.batch_same_intr) #every mini-batch should be sampled from the same group

            self.rand_crop_done_indv = self.args.seq_frame_n_c3d == 1
            self.side_img_needed = self.args.seq_frame_n_pho > 1
            self.scale_img_needed = self.rand_crop_done_indv and self.args.seq_frame_n_pho > 1 and self.args.other_scale > 0
            self.velo_needed = self.args.keep_velo
            self.dont_crop_side = self.args.side_full_img


        self.K_dict = preload_K(args.data_path)

        self.regex_dep2lid = re.compile('.+?sync')

        if data_source is None:
            self.data_source = self.args.data_source
        else:
            self.data_source = data_source
    
    def image_process(self, image, mode, need_mask=False, need_crop=True, op_params=None, cam_ops_list=None, lidar_combo=None):
        '''
        image: PIL.Image object
        op_params.keys: 'random_angle'
        '''
        assert mode == 'img' or mode == 'dep'
        if op_params is None:
            new_op_params = {}

        if isinstance(image, Image.Image):
            data_height = image.height
            data_width = image.width
        else:
            data_height = image.shape[0]
            data_width = image.shape[1]
            assert data_height > 3 and data_width > 3 ## make sure these two dims are not channel dim

        xy_crop = (0, 0, data_width, data_height)
        ## kb_crop the center 1216*352
        ## not dependent on need_crop
        ############## !!!!!!!!!!! Minghan: we disable do_kb_crop in training because later there are random crops
        if self.args.do_kb_crop is True:
            top_margin = int(data_height - 352)
            left_margin = int((data_width - 1216) / 2)
            if isinstance(image, Image.Image):
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            else:
                image = image[top_margin:top_margin+352, left_margin:left_margin+1216]
            
            xy_crop_ini = (left_margin, top_margin, 1216, 352)
            
            if cam_ops_list is not None:
                cam_ops_list.append( CamCrop(left_margin, top_margin, x_size=1216, y_size=352) )
    
        # # To avoid blank boundaries due to pixel registration
        # if self.args.dataset == 'nyu':
        #     image = image.crop((43, 45, 608, 472))

        ## init_scaling after kb_crop: This is to downscale the input for networks
        if self.args.init_width > 0:
            if mode == 'img':
                image = scale_image(image, new_width=self.args.init_width, new_height=self.args.init_height, torch_mode=True, raw_float=False, nearest=False, align_corner=False)
            elif mode == 'dep':
                if self.data_source == 'kitti_depth':
                    image = scale_image(image, self.args.init_width, self.args.init_height, torch_mode=True, nearest=True, raw_float=True, align_corner=False)
                # if using lidar points, we need to do kb_crop and scale together and get the target K and image size, before doing the projection. The previous calculated depth_gt is disgarded. 
                elif self.data_source == 'kitti_raw':
                    velo, extr_cam_li, K_unit = lidar_combo
                    original_K = scale_K(K_unit, new_width=data_width, new_height=data_height, torch_mode=False)
                    image = crop_and_scale_depth_from_lidar(velo, extr_cam_li, original_K, xy_crop_ini, self.args.init_width, self.args.init_height, align_corner=False)
                else:
                    raise ValueError("self.data_source not recognized")
            if cam_ops_list is not None:
                cam_ops_list.append(CamScale(scale=0, new_width=self.args.init_width, new_height=self.args.init_height, align_corner=False))

        ## up to here, the xy_crop_ini is useless

        # ## rotate image
        # if self.args.do_random_rotate is True:
        #     if op_params is not None:
        #         random_angle = op_params['random_angle']
        #     else:
        #         random_angle = (random.random() - 0.5) * 2 * self.args.degree
        #         new_op_params['random_angle'] = random_angle

        #     if mode == 'img':
        #         image = self.rotate_image(image, random_angle)
        #         if cam_ops_list is not None:
        #             cam_ops_list.append(CamRotate(angle_deg=random_angle, nearest=False))
        #     elif mode == 'dep':
        #         image = self.rotate_image(image, random_angle, flag=Image.NEAREST)
        #         if cam_ops_list is not None:
        #             cam_ops_list.append(CamRotate(angle_deg=random_angle, nearest=True))
        
        ## to np array
        if mode == 'img':
            image = np.asarray(image, dtype=np.float32) / 255.0
        elif mode == 'dep':
            image = np.asarray(image, dtype=np.float32)

        ## generate masks
        if mode == 'dep':
            if need_mask:
                mask_gt = image > 0
                mask = binary_closing(mask_gt, self.dilate_struct)

            ## create channel dimension at the end
            image = np.expand_dims(image, axis=2)
            if need_mask:
                mask_gt = np.expand_dims(mask_gt, axis=2)
                mask = np.expand_dims(mask, axis=2)

            ## get the depth map of true scale
            if self.args.dataset == 'nyu':
                image = image / 1000.0
            elif self.data_source == 'kitti_depth':
                image = image / 256.0

        if need_crop and self.rand_crop_done_indv:
            ## random crop
            if op_params is not None:
                x_start = op_params['random_crop_x']
                y_start = op_params['random_crop_y']
            else:
                x_start, y_start = gen_rand_crop_tensor_param(image, target_width=self.args.input_width, target_height=self.args.input_height, h_dim=0, w_dim=1, scale=self.args.other_scale, ori_crop=None)
                # x_start = random.randint(0, image.shape[1] - self.args.input_width)
                # y_start = random.randint(0, image.shape[0] - self.args.input_height)
                # ## Make sure the crop offset can be divided by scaling factor if any, 
                # ## to make sure that the cropped-then-scaled image will be exactly a subset of the original-scaled image. 
                # if self.args.other_scale > 0:
                #     if xy_crop is not None:
                #         x_res = (x_start+xy_crop[0]) % self.args.other_scale
                #         y_res = (y_start+xy_crop[1]) % self.args.other_scale
                #     else:
                #         x_res = x_start % self.args.other_scale
                #         y_res = y_start % self.args.other_scale
                #     x_start -= x_res
                #     y_start -= y_res
                #     if x_start < 0:
                #         x_start += self.args.other_scale
                #         if x_start + self.args.input_width > image.shape[1]:
                #             raise ValueError('cropping failed, the previous cropping is too tight to adjust for proper scaling consistency')
                #     if y_start < 0:
                #         y_start += self.args.other_scale
                #         if y_start + self.args.input_height > image.shape[0]:
                #             raise ValueError('cropping failed, the previous cropping is too tight to adjust for proper scaling consistency')

                new_op_params['random_crop_x'] = x_start
                new_op_params['random_crop_y'] = y_start
            
            assert image.shape[0] >= self.args.input_height and image.shape[1] >= self.args.input_width
            image = image[y_start:y_start + self.args.input_height, x_start:x_start + self.args.input_width, :]
            # image, depth_gt, mask, mask_gt, x_start, y_start = self.random_crop(image, depth_gt, mask, mask_gt, self.args.input_height, self.args.input_width)
            if mode == 'dep' and need_mask:
                mask = mask[y_start:y_start + self.args.input_height, x_start:x_start + self.args.input_width, :]
                mask_gt = mask_gt[y_start:y_start + self.args.input_height, x_start:x_start + self.args.input_width, :]

            if cam_ops_list is not None:
                cam_ops_list.append(CamCrop(x_start, y_start, x_size=self.args.input_width, y_size=self.args.input_height))

            if op_params is None:
                x_size = self.args.input_width
                y_size = self.args.input_height
                x_start = x_start
                y_start = y_start
                xy_crop = (x_start, y_start, x_size, y_size)

        if op_params is None:
            new_op_params['xy_crop'] = xy_crop

        if mode == 'img':
            if op_params is None:
                return image, new_op_params
            else:
                return image
        elif mode == 'dep':
            if need_mask:
                if op_params is None:
                    return image, mask_gt, mask, new_op_params
                else:
                    return image, mask_gt, mask
            else:
                if op_param is None:
                    return image, new_op_params
                else:
                    return image

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])

        path_strs = sample_path.split()[0].split('/')
        date_str = path_strs[0]
        seq_str = path_strs[1]
        seq_n = int(seq_str.split('_drive_')[1].split('_')[0])  # integer of the sequence number
        side = int(path_strs[2].split('_')[1])
        frame = int(path_strs[-1].split('.')[0])

        if self.mode == 'train':
            if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                image_path = os.path.join(self.args.data_path, "./" + sample_path.split()[3])
                depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[4])
            else:
                image_path = os.path.join(self.args.data_path, "./" + sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[1])
    
            ## load image and depth
            image = Image.open(image_path)
            if self.data_source == 'kitti_depth':
                depth_gt = Image.open(depth_path)
            elif self.data_source == 'kitti_raw':
                seq_path = self.regex_dep2lid.search(depth_path)
                seq_path = seq_path.group(0)
                lidar_path = os.path.join(seq_path, 'velodyne_points', 'data', '{:010d}.bin'.format(frame))
                # 2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin
                velo = load_velodyne_points(lidar_path)
                velo = velo[ velo[:, 0] >= 0, : ]
                K_unit = self.K_dict[(date_str, side)].K_unit
                extr_cam_li = self.K_dict[(date_str, side)].P_cam_li
                im_shape = (image.height, image.width)
                assert image.height == self.K_dict[(date_str, side)].height
                assert image.width == self.K_dict[(date_str, side)].width
                depth_gt = lidar_to_depth(velo, extr_cam_li, K_unit, im_shape, torch_mode=False)
                depth_gt = Image.fromarray(depth_gt.astype(np.float32), 'F') #mode 'F' means float32
            else:
                raise ValueError("self.data_source not recognized")
            
            ## log the image operations for later intrinsics and CamInfo processing
            cam_ops = []
            if self.scale_img_needed:
                scale_cam_ops = []

            # ## initial scaling before any cropping. This is to downscale the input for networks
            # if self.args.init_scaling != 1:
            #     init_width = int(image.width / self.args.init_scaling)
            #     init_height = int(image.height / self.args.init_scaling)
            #     image = scale_image(image, new_width=init_width, new_height=init_height, torch_mode=True, raw_float=False, nearest=False, align_corner=False)
            #     if self.data_source == 'kitti_depth':
            #         depth_gt = scale_image(depth_gt, init_width, init_height, torch_mode=True, nearest=True, raw_float=True, align_corner=False)
            #     elif self.data_source == 'kitti_raw':
            #         depth_gt = scale_depth_from_lidar(velo, extr_cam_li, K_unit, init_width, init_height, align_corner=False)
            #     else:
            #         raise ValueError("self.data_source not recognized")
            #     cam_ops.append(CamScale(scale=1/self.args.init_scaling, new_width=init_width, new_height=init_height, align_corner=False))

            ## cropping, rotation augmentations
            image, op_params = self.image_process(image, mode='img', need_mask=False, op_params=None, cam_ops_list=cam_ops)
            depth_gt, mask_gt, mask = self.image_process(depth_gt, mode='dep', need_mask=True, op_params=op_params, lidar_combo=(velo, extr_cam_li, K_unit))
            xy_crop = op_params['xy_crop']
            # xy_crop = (x_start, y_start, x_size, y_size)

            ## if random crop is done in __getitem__, we can continue to scale the cropped image if wanted
            ## if random crop is done in collate_fn, then the scaling should be done there too. 
            if self.scale_img_needed:
                scale = 1/self.args.other_scale
                scaled_height = int(scale * xy_crop[3])
                scaled_width = int(scale * xy_crop[2])
                image_ori_scaled = scale_image(image, new_width=scaled_width, new_height=scaled_height, torch_mode=True, raw_float=False, nearest=False, align_corner=False)
                ## cam_ops and scale_cam_ops are separate
                scale_cam_ops = cam_ops.copy()
                scale_cam_ops.append(CamScale(scale=scale, new_width=scaled_width, new_height=scaled_height, align_corner=False))
                # image_ori_scaled = np.asarray(image_ori_scaled, dtype=np.float32) / 255.0
                # if self.data_source == 'kitti_depth':
                #     depth_gt_scaled = scale_image(depth_gt, new_width=scaled_width, new_height=scaled_height, torch_mode=False, nearest=True, raw_float=True )
                #     depth_gt_scaled = np.asarray(depth_gt_scaled, dtype=np.float32)
                # elif self.data_source == 'kitti_raw':
                #     scale_w, scale_h = scale_from_size(new_width=image.width, new_height=image.height)
                #     original_K = scale_K(K_unit, scale_w, scale_h, torch_mode=False)

                #     scaled_cropped_K = crop_and_scale_K(original_K, xy_crop, scale, torch_mode=False)
                #     depth_gt_scaled = lidar_to_depth(velo, extr_cam_li, im_shape=(scaled_height, scaled_width), K_ready=scaled_cropped_K, K_unit=None)
                #     depth_gt_scaled = np.expand_dims(depth_gt_scaled, axis=2)
                # else:
                #     raise ValueError("self.data_source not recognized")


            ## random flip, gamma, brightness, color augmentation
            ## Minghan: random flip disabled
            image_aug, depth_gt, mask, mask_gt = self.train_preprocess(image, depth_gt, mask, mask_gt)

            ## get global pose of the camera
            pose_file = os.path.join(self.args.data_path, date_str, seq_str, 'poses', 'cam_{:02d}.txt'.format(side))
            with open(pose_file) as f:
                T_lines = f.readlines()
            cur_line = T_lines[frame]
            T = np.array( list(map(float, cur_line.split())) ).reshape(3,4)
            T = np.vstack( (T, np.array([[0,0,0,1]]))).astype(np.float32)

            ## load neighbor images in seq_aside mode
            if self.side_img_needed:
                ## use the same randomized quantity as the curent frame
                T_side = []
                image_side = []
                off_side = []
                if self.scale_img_needed:
                    image_side_scaled = []
                follow = (self.args.seq_frame_n_pho - 1) // 2
                front = self.args.seq_frame_n_pho - 1 - follow
                for offset in range(-front, follow+1):
                    if offset == 0:
                        continue
                    ## load image
                    line_cur = self.frame2line[(date_str, (seq_n, side), frame+offset)]
                    image_path_cur = os.path.join(self.args.data_path, "./" + self.filenames[line_cur].split()[0])
                    image_cur = Image.open(image_path_cur)
                    # if self.args.init_scaling != 1:
                    #     init_width = int(image_cur.width / self.args.init_scaling)
                    #     init_height = int(image_cur.height / self.args.init_scaling)
                    #     image_cur = scale_image(image_cur, new_width=init_width, new_height=init_height, torch_mode=True, raw_float=False, nearest=False, align_corner=False)
                    image_cur = self.image_process(image_cur, mode='img', need_mask=False, need_crop=(not self.dont_crop_side), op_params=op_params)
                    ## load pose
                    T_cur_line = T_lines[frame+offset]
                    T_cur = np.array( list(map(float, T_cur_line.split())) ).reshape(3,4)
                    T_cur = np.vstack( (T_cur, np.array([[0,0,0,1]]))).astype(np.float32)

                    T_side.append(T_cur)
                    image_side.append(image_cur)
                    off_side.append(offset)

                    if self.scale_img_needed:
                        scale = 1/self.args.other_scale
                        if self.dont_crop_side:
                            x_crop_size, y_crop_size = crop_for_perfect_scaling(image_cur, self.args.other_scale, h_dim=0, w_dim=1)
                            image_cur = image_cur[:y_crop_size, :x_crop_size, :]
                            scaled_height_side = int(scale * y_crop_size)
                            scaled_width_side = int(scale * x_crop_size)
                        else:
                            scaled_height_side = int(scale * xy_crop[3])
                            scaled_width_side = int(scale * xy_crop[2])
                        image_side_scaled_cur = scale_image(image_cur, new_width=scaled_width_side, new_height=scaled_height_side, torch_mode=True, raw_float=False, nearest=False, align_corner=False)
                        # image_side_scaled_cur = np.asarray(image_side_scaled_cur, dtype=np.float32) / 255.0
                        image_side_scaled.append(image_side_scaled_cur)

            xy_crop = tuple(np.float32(elem) for elem in xy_crop)
            sample = {'image': image_aug, 'image_ori':image, 'depth': depth_gt, 'focal': focal, 'mask': mask, 'mask_gt': mask_gt, 
                            'date_str': date_str, 'side': side, 'xy_crop': xy_crop, 'seq': seq_n, 'frame': frame, 'T': T, 
                            'cam_ops': cam_ops}

            if self.side_img_needed:
                sample.update({'T_side': T_side, 'image_side': image_side, 'off_side': off_side})
            if self.scale_img_needed:
                sample.update({'image_ori_scaled': image_ori_scaled, 'scale_cam_ops': scale_cam_ops})
            if self.side_img_needed and self.scale_img_needed:
                sample.update({'image_side_scaled': image_side_scaled})
            if self.velo_needed:
                sample.update({'velo': velo})
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))
                    mask_gt = False
                    mask = False

                if has_valid_depth:
                    if self.data_source == 'kitti_depth':
                        depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    elif self.data_source == 'kitti_raw':
                        seq_path = self.regex_dep2lid.search(depth_path)
                        seq_path = seq_path.group(0)
                        lidar_path = os.path.join(seq_path, 'velodyne_points', 'data', '{:010d}.bin'.format(frame))
                        # 2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin
                        velo = load_velodyne_points(lidar_path)
                        velo = velo[ velo[:, 0] >= 0, : ]
                        K_unit = self.K_dict[(date_str, side)].K_unit
                        extr_cam_li = self.K_dict[(date_str, side)].P_cam_li
                        im_shape = (image.shape[0], image.shape[1])
                        assert image.shape[0] == self.K_dict[(date_str, side)].height
                        assert image.shape[1] == self.K_dict[(date_str, side)].width
                        depth_gt = lidar_to_depth(velo, extr_cam_li, K_unit, im_shape, torch_mode=False)
                        depth_gt = depth_gt.astype(np.float32)
                        # depth_gt = Image.fromarray(depth_gt.astype(np.float32), 'F') #mode 'F' means float32
                        # depth_gt_ary = np.array(depth_gt)
                        # print(depth_gt_ary.min(), depth_gt_ary.max(), depth_gt_ary.dtype)
                    else:
                        raise ValueError("self.data_source not recognized")

                    mask_gt = depth_gt > 0
                    mask = binary_closing(mask_gt, self.dilate_struct)

                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    elif self.data_source == 'kitti_depth':
                        depth_gt = depth_gt / 256.0

                    mask_gt = np.expand_dims(mask_gt, axis=2)
                    mask = np.expand_dims(mask, axis=2)

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    mask_gt = mask_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    mask = mask[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            else:
                image = cv2.resize(image, (1216, 352))      ## TODO: resize is not handled in c3d_loss
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = cv2.resize(depth_gt, (1216, 352))      ## TODO: resize mode is not taken care of for depth and masks
                    mask = cv2.resize(mask, (1216, 352))
                    mask_gt = cv2.resize(mask_gt, (1216, 352))      

                
            x_start = 0
            y_start = 0
            if self.args.do_kb_crop is True:
                x_start = x_start + left_margin
                y_start = y_start + top_margin
            x_size = 1216
            y_size = 352
            xy_crop = (x_start, y_start, x_size, y_size)
            xy_crop = tuple(float(elem) for elem in xy_crop)

            ## if init_width is not 0, we also do the scaling here, since this is the default setting for the network. 
            # We scale the image here, but do not scale true depth. Then after network inference we scale back the depth prediction to compare with gt_depth in original scale.
            if self.args.init_width > 0:
                image = scale_image(image, new_width=self.args.init_width, new_height=self.args.init_height, torch_mode=True, raw_float=False, nearest=False, align_corner=False)

            if self.mode == 'online_eval':
                ## get global pose of the camera
                pose_file = os.path.join(self.args.data_path, date_str, seq_str, 'poses', 'cam_{:02d}.txt'.format(side))
                with open(pose_file) as f:
                    cur_line = f.readlines()[frame]
                T = np.array( list(map(float, cur_line.split())) ).reshape(3,4)
                T = np.vstack( (T, np.array([[0,0,0,1]]))).astype(np.float32)
            
            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 
                            'has_valid_depth': has_valid_depth, 'mask': mask, 'mask_gt': mask_gt, 'date_str': date_str, 'side': side, 'xy_crop': xy_crop, 'seq': seq_n, 'frame': frame, 'T': T}
            else:   ## i.e. self.mode == 'test'. Only input image is needed. 
                sample = {'image': image, 'focal': focal}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    # def random_crop(self, img, depth, mask, mask_gt, height, width):
    #     assert img.shape[0] >= height
    #     assert img.shape[1] >= width
    #     assert img.shape[0] == depth.shape[0]
    #     assert img.shape[1] == depth.shape[1]
    #     x = random.randint(0, img.shape[1] - width)
    #     y = random.randint(0, img.shape[0] - height)
    #     img = img[y:y + height, x:x + width, :]
    #     depth = depth[y:y + height, x:x + width, :]
    #     mask = mask[y:y + height, x:x + width, :]
    #     mask_gt = mask_gt[y:y + height, x:x + width, :]
    #     return img, depth, mask, mask_gt, x, y

    def train_preprocess(self, image, depth_gt, mask, mask_gt):
        # # Random flipping
        # do_flip = random.random()
        # if do_flip > 0.5:
        #     image = (image[:, ::-1, :]).copy()
        #     depth_gt = (depth_gt[:, ::-1, :]).copy()
        #     mask = (mask[:, ::-1, :]).copy()
        #     mask_gt = (mask_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt, mask, mask_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.filenames)

def need_totensor(x):
    if isinstance(x, list):
        return need_totensor(x[0])
    # return _is_pil_image(x) or _is_numpy_image(x)
    return isinstance(x, np.ndarray) or isinstance(x, Image.Image)

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        ## load all miscellaneous items that does not need any processing here (avoid manually adding them here)
        new_batch = {key: sample[key] for key in sample if not need_totensor(sample[key]) }

        # process pose
        T = sample['T']
        T = torch.from_numpy(T)

        new_batch.update({'image': image, 'T': T })

        depth = sample['depth']
        mask = sample['mask']
        mask_gt = sample['mask_gt']

        if self.mode == 'train':
            depth = self.to_tensor(depth)
            mask = self.to_tensor(mask)
            mask_gt = self.to_tensor(mask_gt)

            image_ori = self.to_tensor(sample['image_ori'])

            new_batch.update({'depth': depth, 'mask': mask, 'mask_gt': mask_gt, 'image_ori': image_ori })

            if 'image_side' in sample:
                ## there are neighboring images to process
                new_image_side = torch.stack([self.to_tensor(image_cur) for image_cur in sample['image_side'] ], dim=0) 
                new_T_side = torch.stack([torch.from_numpy(T_cur) for T_cur in sample['T_side'] ], dim=0)
                # new_image_side = self.to_tensor(sample['image_side']) 
                # new_T_side = torch.from_numpy(sample['T_side'])
                new_batch.update({'image_side': new_image_side, 'T_side': new_T_side})
                
            if 'image_ori_scaled' in sample:
                image_ori_scaled = self.to_tensor(sample['image_ori_scaled'])
                new_batch.update({'image_ori_scaled': image_ori_scaled})

            if 'image_side_scaled' in sample:
                image_side_scaled = torch.stack([self.to_tensor(image_cur) for image_cur in sample['image_side_scaled'] ], dim=0) 
                new_batch.update({'image_side_scaled': image_side_scaled})

            if 'velo' in sample:
                new_batch.update({'velo': torch.from_numpy(sample['velo']) })

            return new_batch
        else:
            new_batch.update({'depth': depth, 'mask': mask, 'mask_gt': mask_gt})

            return new_batch
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))#.to(dtype=torch.float32)
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

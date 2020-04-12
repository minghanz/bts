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
from sampler_kitti import SamplerKITTI, Collate_Cfg, samp_from_seq, gen_samp_list, frame_line_mapping
from bts_pre_intr import preload_K, load_velodyne_points, lidar_to_depth
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
    def __init__(self, args, mode, data_source=None):
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
                self.train_sampler = SamplerKITTI(self.training_samples, args.batch_size, args.seq_frame_n)   # to make sure all samples in a mini-batch have the same intrinsics
                self.collate_class = Collate_Cfg(args.input_width, args.input_height)
                                    
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
                                   num_workers=1,
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
        frame_idxs_to_sample, line_idx_to_sample = samp_from_seq(self.frame_idx, self.line_idx, args.seq_frame_n)
        self.lines_group = gen_samp_list(line_idx_to_sample, use_date_key=args.batch_same_intr) #every mini-batch should be sampled from the same group


        self.K_dict = preload_K(args.data_path)

        self.regex_dep2lid = re.compile('.+?sync')

        if data_source is None:
            self.data_source = self.args.data_source
        else:
            self.data_source = data_source

    
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
                depth_gt = lidar_to_depth(velo, extr_cam_li, K_unit, im_shape)
                depth_gt = Image.fromarray(depth_gt.astype(np.float32), 'F') #mode 'F' means float32
            else:
                raise ValueError("self.data_source not recognized")
            
            ## kb_crop the center 1216*352
            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

                xy_crop = (left_margin, top_margin, 1216, 352)
            
            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))
    
            ## rotate image
            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            ## to np array
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)

            ## generate masks
            mask_gt = depth_gt > 0
            mask = binary_closing(mask_gt, self.dilate_struct)

            ## create channel dimension at the end
            depth_gt = np.expand_dims(depth_gt, axis=2)
            mask_gt = np.expand_dims(mask_gt, axis=2)
            mask = np.expand_dims(mask, axis=2)

            ## get the depth map of true scale
            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            elif self.data_source == 'kitti_depth':
                depth_gt = depth_gt / 256.0

            # ## random crop
            # image, depth_gt, mask, mask_gt, x_start, y_start = self.random_crop(image, depth_gt, mask, mask_gt, self.args.input_height, self.args.input_width)

            # if self.args.do_kb_crop is True:
            #     x_start = x_start + left_margin
            #     y_start = y_start + top_margin
            # x_size = self.args.input_width
            # y_size = self.args.input_height
            # xy_crop = (x_start, y_start, x_size, y_size)

            ## random flip, gamma, brightness, color augmentation
            image, depth_gt, mask, mask_gt = self.train_preprocess(image, depth_gt, mask, mask_gt)

            ## get global pose of the camera
            pose_file = os.path.join(self.args.data_path, date_str, seq_str, 'poses', 'cam_{:02d}.txt'.format(side))
            with open(pose_file) as f:
                cur_line = f.readlines()[frame]
            T = np.array( list(map(float, cur_line.split())) ).reshape(3,4)
            T = np.vstack( (T, np.array([[0,0,0,1]]))).astype(np.float32)

            # sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'mask': mask, 'mask_gt': mask_gt, 'date_str': date_str, 'side': side, 'xy_crop': xy_crop}
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'mask': mask, 'mask_gt': mask_gt, 'date_str': date_str, 'side': side, 'xy_crop': xy_crop, 'seq': seq_n, 'frame': frame, 'T': T}
        
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
                        depth_gt = lidar_to_depth(velo, extr_cam_li, K_unit, im_shape)
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

    def random_crop(self, img, depth, mask, mask_gt, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        mask = mask[y:y + height, x:x + width, :]
        mask_gt = mask_gt[y:y + height, x:x + width, :]
        return img, depth, mask, mask_gt, x, y

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

        depth = sample['depth']
        mask = sample['mask']
        mask_gt = sample['mask_gt']

        # process pose
        T = sample['T']
        T = torch.from_numpy(T)

        if self.mode == 'train':
            depth = self.to_tensor(depth)
            mask = self.to_tensor(mask)
            mask_gt = self.to_tensor(mask_gt)
            
            return {'image': image, 'depth': depth, 'focal': focal, 
                    'mask': mask, 'mask_gt': mask_gt, 'date_str': sample['date_str'], 'side': sample['side'], 'xy_crop': sample['xy_crop'], 'seq': sample['seq'], 'frame': sample['frame'], 'T': T}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, 
                    'mask': mask, 'mask_gt': mask_gt, 'date_str': sample['date_str'], 'side': sample['side'], 'xy_crop': sample['xy_crop'], 'seq': sample['seq'], 'frame': sample['frame'], 'T': T}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
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

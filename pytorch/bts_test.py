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

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts_dataloader import *

import sys
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, '../../'))
from c3d.utils.geometry import NormalFromDepthDense
from c3d.utils_general.argparse_f import init_argparser_f
from c3d.utils_general.vis import uint8_np_from_img_np, save_np_to_img, uint8_np_from_img_tensor, vis_depth_np
from c3d.utils_general.dataset_read import DataReaderKITTI

parser = init_argparser_f(description='BTS Pytorch test.')

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--save_np', help='if set, save outputs in npy file', action='store_true')
parser.add_argument('--save_normal', help='if set, save disp and normal outputs in png file', action='store_true')
parser.add_argument('--init_width',                type=float, help='rescale the width to what at the beginning after kb cropping', default=0)
parser.add_argument('--init_height',               type=float, help='rescale the height to what at the beginning after kb cropping', default=0)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def test(params):
    """Test function."""
    args.mode = 'test'
    data_meta_reader = DataReaderKITTI(data_root=args.data_path)
    dataloader = BtsDataLoader(args, 'test', data_meta_reader)
    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)

    if args.save_normal:
        normal_model = NormalFromDepthDense()
        normal_model = torch.nn.DataParallel(normal_model)
        normal_model.eval()
        normal_model.cuda()
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    ########### general dataset
    ntps = data_meta_reader.ffinder.ntps_from_split_file(args.filenames_file)
    num_test_samples = len(ntps)

    ###########
    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    pred_8x8s = []
    pred_4x4s = []
    pred_2x2s = []
    pred_1x1s = []

    if args.save_normal:
        normals = []
        K = np.array([[725.0087, 0, 620.5, 0], 
                        [0, 725.0087, 187, 0], 
                        [0, 0, 1, 0], 
                        [0, 0, 0, 1]])
        K[0, :] = K[0, :] / 1242 * 1216
        K[1, :] = K[1, :] / 375 * 352
        K = torch.from_numpy(np.array([K]).astype(np.float32)).cuda()

    start_time = time.time()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            focal = Variable(sample['focal'].cuda())
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, _ = model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())
            pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

            if args.save_normal:
                # print("depth_est.shape", depth_est.shape)
                normal = normal_model(depth_est, K) *0.5 + 0.5
                normals.append(uint8_np_from_img_tensor(normal))


    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    # save_name = 'result_' + args.model_name + args.dataset
    if args.checkpoint_path[0] == '/':
        save_name = os.path.join('result_' + args.dataset, args.checkpoint_path[1:])
    else:
        save_name = os.path.join('result_' + args.dataset, args.checkpoint_path)

    if args.save_np:
        depth_stack = np.stack(pred_depths)
        output_path = os.path.join(save_name, "depth_preds.npy")
        np.save(output_path, depth_stack)
        return
    
    print('Saving result pngs to {}'.format(save_name))
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.makedirs(save_name)
            os.makedirs(save_name + '/raw')
            os.makedirs(save_name + '/cmap')
            os.makedirs(save_name + '/rgb')
            os.makedirs(save_name + '/gt')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    for s in tqdm(range(num_test_samples)):
        rgb_path = data_meta_reader.ffinder.fname_from_ntp(ntps[s], 'rgb')
        rgb_path = os.path.relpath(rgb_path, data_meta_reader.ffinder.data_root)
        if args.dataset == 'kitti':
            date_drive = rgb_path.split('/')[1]
            filename_pred_png = save_name + '/raw/' + date_drive + '_' + rgb_path.split('/')[-1].replace(
                '.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + date_drive + '_' + rgb_path.split('/')[
                -1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + date_drive + '_' + rgb_path.split('/')[-1]
        elif args.dataset == 'kitti_benchmark':
            filename_pred_png = save_name + '/raw/' + rgb_path.split('/')[-1].replace('.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + rgb_path.split('/')[-1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + rgb_path.split('/')[-1]
        else:
            scene_name = rgb_path.split('/')[0]
            filename_pred_png = save_name + '/raw/' + scene_name + '_' + rgb_path.split('/')[-1].replace(
                '.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + rgb_path.split('/')[-1].replace(
                '.jpg', '.png')
            filename_gt_png = save_name + '/gt/' + scene_name + '_' + rgb_path.split('/')[-1].replace(
                '.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + scene_name + '_' + rgb_path.split('/')[-1]
        
        rgb_path = os.path.join(args.data_path, './' + rgb_path)
        image = cv2.imread(rgb_path)
        ### TODO: haven't adapt to nyu yet
        # if args.dataset == 'nyu':
        #     gt_path = os.path.join(args.data_path, './' + lines[s].split()[1])
        #     gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
        #     gt[gt == 0] = np.amax(gt)
        
        pred_depth = pred_depths[s]
        pred_8x8 = pred_8x8s[s]
        pred_4x4 = pred_4x4s[s]
        pred_2x2 = pred_2x2s[s]
        pred_1x1 = pred_1x1s[s]
        
        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark'or args.dataset == 'vkitti':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth * 1000.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if args.save_normal:
            ######### here we want to process each numpy array of depth (H*W) and normal (C*H*W) and save them
            pred_depth_inv = vis_depth_np(pred_depth)
            normal = normals[s]

            pred_depth_inv = uint8_np_from_img_np(pred_depth_inv)
            save_np_to_img(pred_depth_inv, filename_pred_png.replace('.png', '')+'_depinv')
            save_np_to_img(normal, filename_pred_png.replace('.png', '')+'nml')


        
        if args.save_lpg:
            cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            if args.dataset == 'nyu':
                plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
                pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
                pred_8x8_cropped = pred_8x8[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='Greys')
                pred_4x4_cropped = pred_4x4[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='Greys')
                pred_2x2_cropped = pred_2x2[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='Greys')
                pred_1x1_cropped = pred_1x1[10:-1 - 9, 10:-1 - 9]
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1_cropped), cmap='Greys')
            else:
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')
                filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1), cmap='Greys')
    
    return


if __name__ == '__main__':
    test(args)

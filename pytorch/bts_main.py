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

import time
import datetime
import sys
import os

import torch
import torch.nn as nn
import torch.nn.utils as utils

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
import threading
from tqdm import tqdm

from bts import BtsModel
from bts_dataloader import *

from bts_main_argparse import parse_args_main

script_path = os.path.dirname(__file__)
# from c3d_loss import C3DLoss
# sys.path.append(os.path.join(script_path, "../../monodepth2"))
# from cvo_utils import save_tensor_to_img

sys.path.append(os.path.join(script_path, "../../c3d"))
from c3d_loss import C3DLoss
from utils.io import save_tensor_to_img

from bts_utils import vis_depth, overlay_dep_on_rgb

if sys.argv.__len__() == 2:
    args, args_rest = parse_args_main()
else:
    args = parse_args_main()

if args.mode == 'train' and not args.checkpoint_path:
    from bts import *

elif args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    # model_name = os.path.basename(model_dir)
    model_name = 'bts'
    
    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


def set_misc(model):
    if args.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if args.fix_first_conv_blocks:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif args.fix_first_conv_block:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            # print(name, name2)
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False


def online_eval(model, dataloader_eval, gpu, ngpus):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            focal = torch.autograd.Variable(eval_sample_batched['focal'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            _, _, _, _, pred_depth, _ = model(image, focal)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

        # break

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.3f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None


def main_worker(gpu, ngpus_per_node, args, args_rest):
    ## set manual seed to make it reproducible
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    model = BtsModel(args)
    model.train()
    model.decoder.apply(weights_init_xavier)
    set_misc(model)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0
    # best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    # best_eval_measures_higher_better = torch.zeros(3).cpu()
    # best_eval_steps = np.zeros(9, dtype=np.int32)
    best_eval_measures_lower_better_raw = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better_raw = torch.zeros(3).cpu()
    best_eval_steps_raw = np.zeros(9, dtype=np.int32)
    best_eval_measures_lower_better_dep = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better_dep = torch.zeros(3).cpu()
    best_eval_steps_dep = np.zeros(9, dtype=np.int32)

    # C3D module
    c3d_model = C3DLoss(args.data_path, batch_size=args.batch_size, seq_frame_n=args.seq_frame_n)
    c3d_model.parse_opts(args_rest)
    c3d_model.cuda()

    # Training parameters
    optimizer = torch.optim.AdamW([{'params': model.module.encoder.parameters(), 'weight_decay': args.weight_decay},
                                   {'params': model.module.decoder.parameters(), 'weight_decay': 0}],
                                  lr=args.learning_rate, eps=args.adam_eps)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                # best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                # best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                # best_eval_steps = checkpoint['best_eval_steps']
                best_eval_measures_higher_better_raw = checkpoint['best_eval_measures_higher_better_raw'].cpu()
                best_eval_measures_lower_better_raw = checkpoint['best_eval_measures_lower_better_raw'].cpu()
                best_eval_steps_raw = checkpoint['best_eval_steps_raw']
                best_eval_measures_higher_better_dep = checkpoint['best_eval_measures_higher_better_dep'].cpu()
                best_eval_measures_lower_better_dep = checkpoint['best_eval_measures_lower_better_dep'].cpu()
                best_eval_steps_dep = checkpoint['best_eval_steps_dep']
            except KeyError:
                print("Could not load values for online evaluation")

            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True

    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    dataloader = BtsDataLoader(args, 'train')
    dataloader_eval_raw = BtsDataLoader(args, 'online_eval', data_source='kitti_raw')
    dataloader_eval_dep = BtsDataLoader(args, 'online_eval', data_source='kitti_depth')

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer_path = os.path.join(args.log_directory, 'tbsummary', args.model_name)
        writer = SummaryWriter(writer_path+"_train", flush_secs=30)
        # writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            # if args.eval_summary_directory != '':
            #     eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            # else:
            #     eval_summary_path = os.path.join(args.log_directory, 'eval')
            # eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)
            
            eval_summary_writer_raw = SummaryWriter(writer_path+"_eval_raw", flush_secs=30)
            eval_summary_writer_dep = SummaryWriter(writer_path+"_eval_dep", flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, iconv1 = model(image, focal)

            Ts = sample_batched['T'].cuda(args.gpu, non_blocking=True)
            depth_mask = sample_batched['mask'].cuda(args.gpu, non_blocking=True)
            depth_gt_mask = sample_batched['mask_gt'].cuda(args.gpu, non_blocking=True)
            date_side = (sample_batched['date_str'], sample_batched['side'])
            inp = c3d_model(image, depth_est, depth_gt, depth_mask, depth_gt_mask, date_side, xy_crop=sample_batched['xy_crop'], Ts=Ts ) 

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1
            else:
                mask = depth_gt > 1.0

            #### can see that the mask_gt is (at least visually) the same as mask generated above. 
            # save_tensor_to_img(depth_mask, os.path.join(args.log_directory, args.model_name, '{}_pred'.format(global_step) ), 'mask')
            # save_tensor_to_img(depth_gt_mask, os.path.join(args.log_directory, args.model_name, '{}_gt'.format(global_step) ), 'mask')
            # save_tensor_to_img(mask, os.path.join(args.log_directory, args.model_name, '{}_ori'.format(global_step) ), 'mask')

            loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            # loss = silog_criterion.forward(depth_est, depth_gt, depth_gt_mask)
            # loss.backward()
            loss_total = loss * args.silog_weight - inp * args.c3d_weight
            loss_total.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if global_step % 100 == 0 :
                    print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    print('depth_est[mask].min(), max(): {}, {}'.format(depth_est[mask].min(), depth_est[mask].max() ))
                    print('depth_gt[mask].min(), max(): {}, {}'.format(depth_gt[mask].min(), depth_gt[mask].max() ))
                    print('iconv1[mask].min(), max(): {}, {}'.format(iconv1[mask].min(), iconv1[mask].max() ))
                    print('log(depth_est[mask].min(), max()): {}, {}'.format(torch.log(depth_est[mask]).min(), torch.log(depth_est[mask]).max() ))
                    print('log(depth_gt[mask].min(), max()): {}, {}'.format(torch.log(depth_gt[mask]).min(), torch.log(depth_gt[mask]).max() ))
                    print('depth_est.min(), max(): {}, {}'.format(depth_est.min(), depth_est.max() ))
                    print('depth_gt.min(), max(): {}, {}'.format(depth_gt.min(), depth_gt.max() ))
                    print('iconv1.min(), max(): {}, {}'.format(iconv1.min(), iconv1.max() ))
                    return -1

            duration += time.time() - before_op_time
            # if True:
            if global_step and ( (global_step % 100 == 0 and global_step < 1000) or global_step % args.log_freq == 0 ) and not model_just_loaded:
                var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                # training_time_left = (num_total_steps / (global_step+1) - 1.0) * time_sofar
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('silog_loss', loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                    writer.add_scalar('c3d inp', inp, global_step) # cvo logging
                    writer.add_scalar('loss_total', loss_total, global_step) # cvo logging
                    # depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                    batch_size_cur = depth_gt.shape[0]  # this fixes a bug which may not always occur because it happenss only when the logging iteration (once every args.log_freq) happens to be the last mini-batch in a epoch.
                    for i in range(batch_size_cur):
                        writer.add_image('depth_gt/image/{}'.format(i), vis_depth(depth_gt[i, :, :, :]), global_step)
                        writer.add_image('depth_est/image/{}'.format(i), vis_depth(depth_est[i, :, :, :]), global_step)
                        writer.add_image('reduc1x1/image/{}'.format(i), normalize_result(1/reduc1x1[i, :, :, :].data), global_step)
                        writer.add_image('lpg2x2/image/{}'.format(i), normalize_result(1/lpg2x2[i, :, :, :].data), global_step)
                        writer.add_image('lpg4x4/image/{}'.format(i), normalize_result(1/lpg4x4[i, :, :, :].data), global_step)
                        writer.add_image('lpg8x8/image/{}'.format(i), normalize_result(1/lpg8x8[i, :, :, :].data), global_step)
                        writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
                        writer.add_image('mask/image/{}'.format(i), depth_mask[i, :, :, :].data, global_step)
                        writer.add_image('mask_gt/image/{}'.format(i), depth_gt_mask[i, :, :, :].data, global_step)
                        writer.add_image('mask_ori/image/{}'.format(i), mask[i, :, :, :].data, global_step)
                        # name_global_step = '{}_{}.jpg'.format(global_step, i)
                        # name_abs_file = '{}_{}_{}.jpg'.format(sample_batched['date_str'][i], sample_batched['seq'][i], sample_batched['frame'][i])
                        # img_dep = overlay_dep_on_rgb(vis_depth(depth_gt[i, :, :, :]), inv_normalize(image[i, :, :, :]), 
                        #             path=os.path.join(args.log_directory, args.model_name, 'img_dep'), name=name_abs_file )
                    writer.flush()

            model_just_loaded = False
            global_step += 1

        
        ## Minghan: If not do_online_eval, models are saved per save_freq
        # if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
        if (epoch+1) % args.save_freq == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                checkpoint = {'global_step': global_step,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

        ## Minghan: If do_online_eval, models are evaled per eval_freq, and saved if is the best in terms of some metrics
        # if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
        if args.do_online_eval:
            time.sleep(0.1)
            model.eval()
            eval_measures_raw = online_eval(model, dataloader_eval_raw, gpu, ngpus_per_node)
            best_eval_measures_lower_better_raw, best_eval_measures_higher_better_raw, best_eval_steps_raw = log_eval(model, optimizer, 
                                                                        eval_measures_raw, eval_summary_writer_raw, global_step, 'raw', best_eval_measures_lower_better_raw, best_eval_measures_higher_better_raw, best_eval_steps_raw)
            eval_measures_dep = online_eval(model, dataloader_eval_dep, gpu, ngpus_per_node)
            best_eval_measures_lower_better_dep, best_eval_measures_higher_better_dep, best_eval_steps_dep = log_eval(model, optimizer, 
                                                                        eval_measures_dep, eval_summary_writer_dep, global_step, 'dep', best_eval_measures_lower_better_dep, best_eval_measures_higher_better_dep, best_eval_steps_dep)
            
            model.train()
            block_print()
            set_misc(model)
            enable_print()

        epoch += 1

def log_eval(model, optimizer, eval_measures, eval_summary_writer, global_step, mode, best_eval_measures_lower_better, best_eval_measures_higher_better, best_eval_steps):
    assert mode == 'raw' or mode == 'dep'

    if eval_measures is not None:
        for i in range(9):
            eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
            measure = eval_measures[i]
            is_best = False
            if i < 6 and measure < best_eval_measures_lower_better[i]:
                old_best = best_eval_measures_lower_better[i].item()
                best_eval_measures_lower_better[i] = measure.item()
                is_best = True
            elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                old_best = best_eval_measures_higher_better[i-6].item()
                best_eval_measures_higher_better[i-6] = measure.item()
                is_best = True
            if is_best:
                old_best_step = best_eval_steps[i]
                old_best_name = '/model-{}-best_{}_{:.5f}_{}'.format(old_best_step, eval_metrics[i], old_best, mode)
                model_path = args.log_directory + '/' + args.model_name + old_best_name
                if os.path.exists(model_path):
                    command = 'rm {}'.format(model_path)
                    os.system(command)
                best_eval_steps[i] = global_step
                model_save_name = '/model-{}-best_{}_{:.5f}_{}'.format(global_step, eval_metrics[i], measure, mode)
                print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                checkpoint = {'global_step': global_step,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_eval_measures_higher_better_{}'.format(mode): best_eval_measures_higher_better,
                                'best_eval_measures_lower_better_{}'.format(mode): best_eval_measures_lower_better,
                                'best_eval_steps_{}'.format(mode): best_eval_steps
                                }
                torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
        eval_summary_writer.flush()
    
    return best_eval_measures_lower_better, best_eval_measures_higher_better, best_eval_steps

def main():
    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1

    ctime = time.ctime()
    ctime = ctime.replace(' ', '_')
    args.model_name = args.model_name + '/' + ctime

    # model_filename = args.model_name + '.py'
    model_filename = 'bts.py'

    command = 'mkdir -p ' + args.log_directory + '/' + args.model_name
    os.system(command)

    args_out_path = args.log_directory + '/' + args.model_name + '/' + sys.argv[1]
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    if args.checkpoint_path == '':
        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        command = 'cp bts.py ' + model_out_path
        os.system(command)
        aux_out_path = args.log_directory + '/' + args.model_name + '/.'
        command = 'cp bts_main.py ' + aux_out_path
        os.system(command)
        command = 'cp bts_dataloader.py ' + aux_out_path
        os.system(command)
    else:
        loaded_model_dir = os.path.dirname(args.checkpoint_path)
        loaded_model_filename = model_filename
        # loaded_model_name = os.path.basename(loaded_model_dir)
        # loaded_model_filename = loaded_model_name + '.py'

        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        command = 'cp ' + loaded_model_dir + '/' + loaded_model_filename + ' ' + model_out_path
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every epoch.")
        # print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
        #       .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, args_rest))
    else:
        main_worker(args.gpu, ngpus_per_node, args, args_rest)


if __name__ == '__main__':
    main()

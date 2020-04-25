import torch
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
# from torch._six import int_classes as int_classes
import random
from torch._six import container_abcs, string_classes, int_classes
import numpy as np

import sys, os
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, "../../"))
from c3d.utils.cam import lidar_to_depth, scale_K, scale_from_size, crop_and_scale_K, scale_image, CamCrop, CamScale
from c3d.utils.cam_proj import seq_ops_on_cam_info

def scale_xy_crop(xy_crop, scale):
    apply_scale = 1 / scale
    assert apply_scale != 0
    new_xy_crop = tuple([elem * apply_scale for elem in xy_crop])
    # print('ori:', xy_crop)
    # print('scl:', new_xy_crop)
    return new_xy_crop

def crop_for_perfect_scaling(img, scale, h_dim, w_dim):
    x_res = img.shape[w_dim] % scale
    x_crop_size = img.shape[w_dim] - x_res
    y_res = img.shape[h_dim] % scale
    y_crop_size = img.shape[h_dim] - y_res
    return x_crop_size, y_crop_size

def gen_rand_crop_tensor_param(img, target_width, target_height, h_dim, w_dim, scale=-1, ori_crop=None):
    # if batched:
    #     h, w = 2, 3
    # else:
    #     h, w = 1, 2
    assert img.shape[h_dim] >= target_height
    assert img.shape[w_dim] >= target_width
    x = random.randint(0, img.shape[w_dim] - target_width)
    y = random.randint(0, img.shape[h_dim] - target_height)
    if scale > 0:
        if ori_crop is not None:
            ori_x_start = ori_crop[0]
            ori_y_start = ori_crop[1]
        else:
            ori_x_start = 0
            ori_y_start = 0
        x_res = (x + ori_x_start) % scale
        y_res = (y + ori_y_start) % scale
        x -= x_res
        y -= y_res

        if x < 0:
            x += scale
            if x + target_width > img.shape[w_dim]:
                raise ValueError('cropping failed, the previous cropping is too tight to adjust for proper scaling consistency. img.shape: {}, w_dim: {}, h_dim: {}, tgt_w: {}, tgt_h: {}, x: {}, ori_x: {}'.format(\
                                    img.shape, w_dim, h_dim, target_width, target_height, x, ori_x_start))
        if y < 0:
            y += scale
            if y + target_height > img.shape[h_dim]:
                raise ValueError('cropping failed, the previous cropping is too tight to adjust for proper scaling consistency. img.shape: {}, w_dim: {}, h_dim: {}, tgt_w: {}, tgt_h: {}, y: {}, ori_y: {}'.format(\
                                    img.shape, w_dim, h_dim, target_width, target_height, y, ori_y_start))

    return x, y

# def random_crop_tensor(img, depth, mask, mask_gt, image_ori, height, width, batched):
#     if batched:
#         h, w = 2, 3
#     else:
#         h, w = 1, 2
#     assert img.shape[h] >= height
#     assert img.shape[w] >= width
#     assert img.shape[h] == depth.shape[h]
#     assert img.shape[w] == depth.shape[w]
#     x = random.randint(0, img.shape[w] - width)
#     y = random.randint(0, img.shape[h] - height)

#     img = img[..., y:y + height, x:x + width]
#     depth = depth[..., y:y + height, x:x + width]
#     mask = mask[..., y:y + height, x:x + width]
#     mask_gt = mask_gt[..., y:y + height, x:x + width]
#     image_ori = image_ori[..., y:y + height, x:x + width]
#     return img, depth, mask, mask_gt, image_ori, x, y


class Collate_Cfg:
    def __init__(self, width, height, seq_frame_n_c3d, other_scale, side_full_img, cam_proj=None):
        self.net_input_width = width
        self.net_input_height = height
        self.seq_frame_n_c3d = seq_frame_n_c3d
        self.other_scale = other_scale
        
        self.rand_crop_done_batch = seq_frame_n_c3d > 1
        self.scale_cam_info_needed = self.other_scale > 0
        self.scale_img_needed = self.rand_crop_done_batch and other_scale > 0
        self.dont_crop_side = side_full_img
        self.scale_crop_needed = self.scale_cam_info_needed and self.dont_crop_side

        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")

        self.cam_proj = cam_proj

    ### This is modified from original default_collate_fn in pytorch source code
    ### https://github.com/pytorch/pytorch/blob/dc1f9eee531a95cb8f89b734c05f52c4bcdc59ab/torch/utils/data/_utils/collate.py#L42
    def collate_common_crop(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(self.default_collate_err_msg_format.format(elem.dtype))

                return self.collate_common_crop([torch.as_tensor(b, dtype=torch.float32) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch, dtype=torch.float32)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float32)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch, dtype=torch.float32)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            if self.rand_crop_done_batch:
                ### random crop happens here if self.seq_frame_n_c3d > 1
                new_batch = {}

                ##################### random cropping
                image = torch.stack([batchi['image'] for batchi in batch], 0)
                depth = torch.stack([batchi['depth'] for batchi in batch], 0)
                mask = torch.stack([batchi['mask'] for batchi in batch], 0)
                mask_gt = torch.stack([batchi['mask_gt'] for batchi in batch], 0)
                image_ori = torch.stack([batchi['image_ori'] for batchi in batch], 0)

                w_start, h_start = gen_rand_crop_tensor_param(image, self.net_input_width, self.net_input_height, h_dim=-2, w_dim=-1, scale=self.other_scale, ori_crop=batchi['xy_crop'])
                image = image[..., h_start: h_start+self.net_input_height, w_start: w_start+self.net_input_width]
                depth = depth[..., h_start: h_start+self.net_input_height, w_start: w_start+self.net_input_width]
                mask  =  mask[..., h_start: h_start+self.net_input_height, w_start: w_start+self.net_input_width]
                mask_gt = mask_gt[..., h_start: h_start+self.net_input_height, w_start: w_start+self.net_input_width]
                image_ori = image_ori[..., h_start: h_start+self.net_input_height, w_start: w_start+self.net_input_width]
                # image, depth, mask, mask_gt, image_ori, w_start, h_start = random_crop_tensor(image, depth, mask, mask_gt, image_ori, self.net_input_height, self.net_input_width, batched=True)

                new_batch['image'] = image
                new_batch['depth'] = depth
                new_batch['mask'] = mask
                new_batch['mask_gt'] = mask_gt
                new_batch['image_ori'] = image_ori

                if 'image_side' in elem: ## 'off_side' does not need separate processing
                    # new_batch['image_side'] = [ [ imagej[...,h_start:h_start + height, w_start:w_start + width] for imagej in batchi['image_side']] for batchi in batch]
                    new_batch['T_side'] = torch.stack([batchi['T_side'] for batchi in batch], 0)
                    new_batch['image_side'] = torch.stack([batchi['image_side'] for batchi in batch], 0)
                    if self.dont_crop_side:
                        x_side_size, y_side_size = crop_for_perfect_scaling(new_batch['image_side'], self.other_scale, h_dim=-2, w_dim=-1)
                        new_batch['image_side'] = new_batch['image_side'][..., :y_side_size, :x_side_size]
                    else:
                        new_batch['image_side'] = new_batch['image_side'][..., h_start:h_start + self.net_input_height, w_start:w_start + self.net_input_width]

                for batchi in batch:
                    (x_start, y_start, x_size, y_size) = batchi['xy_crop']
                    batchi['xy_crop'] = (x_start + w_start, y_start + h_start, self.net_input_width, self.net_input_height)
                    batchi['xy_crop'] = tuple(np.float32(elem) for elem in batchi['xy_crop'])
                
                ##################### scaling
                if self.scale_img_needed:
                    scale = 1 / self.other_scale
                    new_width = self.net_input_width * scale
                    new_height = self.net_input_height * scale
                    image_ori_scaled = scale_image(new_batch['image_ori'], new_width, new_height, torch_mode=True, nearest=False, raw_float=False)
                    new_batch['image_ori_scaled'] = image_ori_scaled

                    if 'image_side' in elem:
                        if self.dont_crop_side:
                            new_width_side = x_side_size * scale
                            new_height_side = y_side_size * scale
                        else:
                            new_width_side = new_width
                            new_height_side = new_height

                        img_side_scaled =[]
                        for i_side in new_batch['image_side'].shape[1]:
                            img_side_i = scale_image(new_batch['image_side'][:, i_side], new_width_side, new_height_side, torch_mode=True, nearest=False, raw_float=False)
                            img_side_scaled.append(img_side_i)
                        img_side_scaled = torch.stack(img_side_scaled, dim=1)
                        new_batch['image_side_scaled'] = img_side_scaled

                ##################### velo
                if 'velo' in elem:
                    new_batch['velo'] = [batchi['velo'] for batchi in batch]

                ##################### all other items
                for key in elem:
                    if key not in ['image', 'depth', 'mask', 'mask_gt', 'image_ori', 'image_side', 'T_side', 'image_ori_scaled', 'image_side_scaled', 'velo']:
                        new_batch[key] = self.collate_common_crop([d[key] for d in batch])

                ##################### cam_ops_list
                cam_ops = new_batch['cam_ops']
                cam_ops.append(CamCrop(w_start, h_start, self.net_input_width, self.net_input_height))
                new_batch['cam_ops'] = cam_ops
                if self.scale_img_needed:
                    scale_cam_ops = cam_ops.copy()
                    scale_cam_ops.append(CamScale(scale=scale, new_width=new_width, new_height=new_height, align_corner=False))
                    new_batch['scale_cam_ops'] = scale_cam_ops

                ##################### cam_info
                date_side = (new_batch['date_str'][0], int(new_batch['side'][0]) )
                # cam_info = self.cam_proj.prepare_cam_info(date_side=date_side, xy_crop=new_batch['xy_crop'])
                cam_info_ori = self.cam_proj.prepare_cam_info(date_side=date_side)
                cam_info = seq_ops_on_cam_info(cam_info_ori, new_batch['cam_ops'])
                new_batch['cam_info'] = cam_info

                if self.scale_cam_info_needed:
                    # cam_info_scaled = cam_info.scale(new_width, new_height)
                    cam_info_scaled = seq_ops_on_cam_info(cam_info_ori, new_batch['scale_cam_ops'])
                    new_batch['cam_info_scaled'] = cam_info_scaled

                if self.scale_crop_needed:
                    new_batch['xy_crop_scaled'] = scale_xy_crop(new_batch['xy_crop'], self.other_scale)
                
                # ## if multiscale is used, need to create the scaled cropped image and image_side
                # if self.other_scale > 0:
                #     new_w = self.net_input_width / self.other_scale
                #     new_h = self.net_input_height / self.other_scale
                    
                #     new_batch['image'] = scale_image(new_batch['image'], new_w, new_h, torch_mode=True, nearest=False, raw_float=False)
                #     if 'velo' not in new_batch:
                #         new_batch['depth'] = scale_image(new_batch['depth'], new_w, new_h, torch_mode=True, nearest=True, raw_float=True)
                #     else:
                #         cropped_K = new_batch['cam_info'].K_cur
                #         scale_w_crop, scale_h_crop = scale_from_size(old_width=self.net_input_width, old_height=self.net_input_height, new_width=new_w, new_height=new_h)
                #         scaled_cropped_K = scale_K(cropped_K, scale_w_crop, scale_h_crop, torch_mode=True)

                #         depth_gt_scaled = lidar_to_depth(velo, extr_cam_li, im_shape=(scaled_height, scaled_width), K_ready=scaled_cropped_K, K_unit=None)

                return new_batch
            else:
                new_batch = {}
                ##################### velo
                if 'velo' in elem:
                    new_batch['velo'] = [batchi['velo'] for batchi in batch]

                ##################### all other items
                for key in elem:
                    if key not in ['velo']:
                        new_batch[key] = self.collate_common_crop([d[key] for d in batch])

                # new_batch = {key: self.collate_common_crop([d[key] for d in batch]) for key in elem}

                ##################### cam_info
                date_side = (new_batch['date_str'][0], int(new_batch['side'][0]) )
                # cam_info = self.cam_proj.prepare_cam_info(date_side=date_side, xy_crop=new_batch['xy_crop'])
                cam_info = self.cam_proj.prepare_cam_info(date_side=date_side)
                cam_ops_list = new_batch['cam_ops']
                cam_info_main = seq_ops_on_cam_info(cam_info, cam_ops_list)
                new_batch['cam_info'] = cam_info_main
                
                if self.scale_cam_info_needed:
                    cam_ops_list_scale = new_batch['scale_cam_ops']
                    cam_info_scaled = seq_ops_on_cam_info(cam_info, cam_ops_list_scale)
                    new_batch['cam_info_scaled'] = cam_info_scaled
                    
                    # scale = 1 / self.other_scale
                    # new_width = self.net_input_width * scale
                    # new_height = self.net_input_height * scale
                    # cam_info_scaled = cam_info.scale(new_width, new_height)
                    # new_batch['cam_info_scaled'] = cam_info_scaled
                
                ##################### scale xy_crop
                if self.scale_crop_needed:
                    new_batch['xy_crop_scaled'] = scale_xy_crop(new_batch['xy_crop'], self.other_scale)

                return new_batch
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.collate_common_crop(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return [self.collate_common_crop(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

def check_neighbor_exist(frame, frame_idxs, seq_frame_n):
    follow = (seq_frame_n - 1) // 2
    front = seq_frame_n - 1 - follow
    need = True
    need = need and frame in frame_idxs
    for i in range(1, follow+1):
        need = need and frame+i in frame_idxs
    for i in range(1, front+1):
        need = need and frame-i in frame_idxs
    return need

def check_need_to_sample(frame, frame_idxs, frame_idxs_to_sample, seq_frame_n):
    need = True
    for i in range(seq_frame_n):
        need = need and frame+i in frame_idxs
        need = need and frame-i not in frame_idxs_to_sample
        need = need and frame+i not in frame_idxs_to_sample
        
    return need

def samp_from_seq(frame_idxs, line_idxs, seq_frame_n_c3d, seq_frame_n_pho):
    '''Samp once every seq_frame_n frames in a date-seq-side sequence
    '''
    line_idx_to_sample = {}
    frame_idxs_to_sample = {}

    for date in frame_idxs:
        assert date in line_idxs
        line_idx_to_sample[date] = {}
        frame_idxs_to_sample[date] = {}

        for seq_side in frame_idxs[date]:
            assert seq_side in line_idxs[date]
            line_idx_to_sample[date][seq_side] = []
            frame_idxs_to_sample[date][seq_side] = []

            for i, frame in enumerate(frame_idxs[date][seq_side]):
                condition = check_neighbor_exist(frame, frame_idxs[date][seq_side], seq_frame_n_pho) and \
                            check_need_to_sample(frame, frame_idxs[date][seq_side], frame_idxs_to_sample[date][seq_side], seq_frame_n_c3d)
                if condition:
                    frame_idxs_to_sample[date][seq_side].append(frame)
                    line_idx_to_sample[date][seq_side].append(line_idxs[date][seq_side][i])

    return frame_idxs_to_sample, line_idx_to_sample

def gen_samp_list(line_idx_to_sample, use_date_key):
    '''Construct line groups from line_idx_to_sample dict
    '''
    lines = {}
    
    for date in line_idx_to_sample:
        key = date if use_date_key else 'all'
        if key not in lines:
            lines[key] = []
        for seq_side in line_idx_to_sample[date]:
            lines[key].extend(line_idx_to_sample[date][seq_side])

    return lines

def frame_line_mapping(frame_idxs, line_idxs):
    frame2line = {}
    line2frame = {}
    for date in frame_idxs:
        assert date in line_idxs
        for seq_side in frame_idxs[date]:
            assert seq_side in line_idxs[date]
            for frame, line in zip(frame_idxs[date][seq_side], line_idxs[date][seq_side]):
                frame2line[(date, seq_side, frame)] = line
                line2frame[line] = (date, seq_side, frame)

    return frame2line, line2frame

class SamplerKITTI(Sampler):
    """Every sampled mini-batch are from the same date so that they can share the same uvb_flat and xy1_flat in C3DLoss
    """
    def __init__(self, dataset, batch_size, seq_frame_n_c3d, seq_frame_n_pho, drop_last=False ): 
        # drop_last default to False according to https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seq_frame_n_c3d = seq_frame_n_c3d
        self.seq_frame_n_pho = seq_frame_n_pho

        if not isinstance(batch_size, int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
            
        ### we want to first sample date, then sample data
        self.sub_sizes = {}
        self.sub_idxs = []
        for key in self.dataset.lines_group:
            self.sub_sizes[key] = len(self.dataset.lines_group[key])
            idx = [key] * self.sub_sizes[key]
            self.sub_idxs.extend(idx)
        assert len(self.sub_idxs) == sum([self.sub_sizes[key] for key in self.dataset.lines_group]), "The number of samples are the the same as the sum of each subset"
        # self.num_samples = len(self.sub_idxs)
        self.hlvl_sampler = SubsetRandomSampler(self.sub_idxs)

        self.sub_samplers = {}
        for key in self.dataset.lines_group:
            self.sub_samplers[key] = SubsetRandomSampler(self.dataset.lines_group[key])

    def __iter__(self): ## The dataloader creates the iterator object at the beginning of for loop        
        sub_iters = {}
        end_reached = {}
        for key in self.sub_samplers:
            sub_iters[key] = iter(self.sub_samplers[key])
            end_reached[key] = False

        for key in self.hlvl_sampler:
            if all(end_reached.values()):
                break
            batch = []
            while True:
                try:
                    idx = next(sub_iters[key])
                    date, seq_side, frame = self.dataset.line2frame[idx]
                    if self.seq_frame_n_c3d > 1:
                        idx_next = []
                        for i in range(1, self.seq_frame_n_c3d):
                            idx_next.append(self.dataset.frame2line[(date, seq_side, frame+i)])

                except StopIteration:
                    end_reached[key] = True
                    break
                else:
                    batch.append(idx)
                    if self.seq_frame_n_c3d > 1:
                        for idx_new in idx_next:
                            batch.append(idx_new)

                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                        break
            if len(batch) > 0 and not self.drop_last:
                yield batch


    def __len__(self):
        if self.drop_last:
            return sum( sub_size * self.seq_frame_n_c3d // self.batch_size for sub_size in self.sub_sizes.values() )
        else:
            return sum( (sub_size * self.seq_frame_n_c3d + self.batch_size-1) // self.batch_size for sub_size in self.sub_sizes.values() )
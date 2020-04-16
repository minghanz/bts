import torch
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
# from torch._six import int_classes as int_classes
import random
from torch._six import container_abcs, string_classes, int_classes

def random_crop_tensor(img, depth, mask, mask_gt, image_ori, height, width, batched):
    if batched:
        h, w = 2, 3
    else:
        h, w = 1, 2
    assert img.shape[h] >= height
    assert img.shape[w] >= width
    assert img.shape[h] == depth.shape[h]
    assert img.shape[w] == depth.shape[w]
    x = random.randint(0, img.shape[w] - width)
    y = random.randint(0, img.shape[h] - height)

    img = img[..., y:y + height, x:x + width]
    depth = depth[..., y:y + height, x:x + width]
    mask = mask[..., y:y + height, x:x + width]
    mask_gt = mask_gt[..., y:y + height, x:x + width]
    image_ori = image_ori[..., y:y + height, x:x + width]
    return img, depth, mask, mask_gt, image_ori, x, y


class Collate_Cfg:
    def __init__(self, width, height):
        self.net_input_width = width
        self.net_input_height = height
        # self.seq_frame_n = seq_frame_n

        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")

    ### This is modified from original default_collate_fn in pytorch source code
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

                return self.collate_common_crop([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            new_batch = {}

            image = torch.stack([batchi['image'] for batchi in batch], 0)
            depth = torch.stack([batchi['depth'] for batchi in batch], 0)
            mask = torch.stack([batchi['mask'] for batchi in batch], 0)
            mask_gt = torch.stack([batchi['mask_gt'] for batchi in batch], 0)
            image_ori = torch.stack([batchi['image_ori'] for batchi in batch], 0)
            
            image, depth, mask, mask_gt, image_ori, w_start, h_start = random_crop_tensor(image, depth, mask, mask_gt, image_ori, self.net_input_height, self.net_input_width, batched=True)

            new_batch['image'] = image
            new_batch['depth'] = depth
            new_batch['mask'] = mask
            new_batch['mask_gt'] = mask_gt
            new_batch['image_ori'] = image_ori

            for batchi in batch:
                (x_start, y_start, x_size, y_size) = batchi['xy_crop']
                batchi['xy_crop'] = (x_start + w_start, y_start + h_start, self.net_input_width, self.net_input_height)

            for key in elem:
                if key not in ['image', 'depth', 'mask', 'mask_gt', 'image_ori']:
                    new_batch[key] = self.collate_common_crop([d[key] for d in batch])

            return new_batch
            # return {key: self.collate_common_crop([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.collate_common_crop(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return [self.collate_common_crop(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))


def check_need_to_sample(frame, frame_idxs, frame_idxs_to_sample, seq_frame_n):
    need = True
    for i in range(seq_frame_n):
        need = need and frame+i in frame_idxs
        need = need and frame-i not in frame_idxs_to_sample
        need = need and frame+i not in frame_idxs_to_sample
        
    return need

def samp_from_seq(frame_idxs, line_idxs, seq_frame_n):
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
                if check_need_to_sample(frame, frame_idxs[date][seq_side], frame_idxs_to_sample[date][seq_side], seq_frame_n):
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
    def __init__(self, dataset, batch_size, seq_frame_n, drop_last=False ): 
        # drop_last default to False according to https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seq_frame_n = seq_frame_n

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
                    idx_next = []
                    for i in range(1, self.seq_frame_n):
                        idx_next.append(self.dataset.frame2line[(date, seq_side, frame+i)])

                except StopIteration:
                    end_reached[key] = True
                    break
                else:
                    batch.append(idx)
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
            return sum( sub_size * self.seq_frame_n // self.batch_size for sub_size in self.sub_sizes.values() )
        else:
            return sum( (sub_size * self.seq_frame_n + self.batch_size-1) // self.batch_size for sub_size in self.sub_sizes.values() )
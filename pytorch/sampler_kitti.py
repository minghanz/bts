import torch
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch._six import int_classes as _int_classes

class SamplerKITTI(Sampler):
    """Every sampled mini-batch are from the same date so that they can share the same uvb_flat and xy1_flat in C3DLoss
    """
    def __init__(self, dataset, batch_size, drop_last=False ): 
        # drop_last default to False according to https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        self.data_source = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
            
        ### we want to first sample date, then sample data
        self.sub_sizes = {}
        self.sub_idxs = []
        for date in self.data_source.lines_in_date:
            self.sub_sizes[date] = len(self.data_source.lines_in_date[date])
            idx = [date] * self.sub_sizes[date]
            self.sub_idxs.extend(idx)
        assert len(self.sub_idxs) == sum([self.sub_sizes[date] for date in self.data_source.lines_in_date]), "The number of samples are the the same as the sum of each subset"
        self.num_samples = len(self.sub_idxs)
        self.hlvl_sampler = SubsetRandomSampler(self.sub_idxs)

        self.sub_samplers = {}
        for date in self.data_source.lines_in_date:
            self.sub_samplers[date] = SubsetRandomSampler(self.data_source.lines_in_date[date])

    def __iter__(self): ## The dataloader creates the iterator object at the beginning of for loop        
        sub_iters = {}
        end_reached = {}
        for date in self.sub_samplers:
            sub_iters[date] = iter(self.sub_samplers[date])
            end_reached[date] = False

        for date in self.hlvl_sampler:
            if all(end_reached.values()):
                break
            batch = []
            while True:
                try:
                    idx = next(sub_iters[date])
                except StopIteration:
                    end_reached[date] = True
                    break
                else:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                        break
            if len(batch) > 0 and not self.drop_last:
                yield batch


    def __len__(self):
        if self.drop_last:
            return sum( sub_size//self.batch_size for sub_size in self.sub_sizes.values() )
        else:
            return sum( (sub_size+self.batch_size-1) // self.batch_size for sub_size in self.sub_sizes.values() )
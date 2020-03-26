import torch
import numpy as np 

def vis_depth(depth, ref_depth=10):        ## why normalize_result in bts_main.py convert it to numpy?
    dum_zero = torch.zeros_like(depth)
    inv_depth = torch.where(depth>0, ref_depth/(ref_depth+depth), dum_zero)
    return inv_depth
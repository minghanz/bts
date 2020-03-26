import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict
import os
import sys
script_path = os.path.dirname(__file__)
sys.path.append(os.path.join(script_path, '../../pytorch-unet'))
from geometry import rgb_to_hsv
sys.path.append(os.path.join(script_path, '../../monodepth2'))
from cvo_utils import *

import argparse

def gen_uv_grid(width, height, torch_mode):
    """
    return: uv_coords(2*H*W)
    """
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    uv_grid = np.stack(meshgrid, axis=0).astype(np.float32) # 2*H*W
    if torch_mode:
        uv_grid = torch.from_numpy(uv_grid)
    return uv_grid

def xy1_from_uv(uv_grid, inv_K, torch_mode):
    """
    uv_grid: 2*H*W
    inv_K: 3*3
    return: uv1_flat, xy1_flat(3*N)
    """
    if torch_mode:
        uv_flat = uv_grid.reshape(2, -1) # 2*N
        dummy_ones = torch.ones((1, uv_flat.shape[1]), dtype=uv_flat.dtype, device=uv_flat.device) # 1*N
        uv1_flat = torch.cat((uv_flat, dummy_ones), dim=0) # 3*N
        xy1_flat = torch.matmul(inv_K, uv1_flat)
    else:
        uv_flat = uv_grid.reshape(2, -1) # 2*N
        dummy_ones = torch.ones((1, uv_flat.shape[1]), dtype=np.float32)
        uv1_flat = np.concatenate((uv_flat, dummy_ones), axis=0) # 3*N
        xy1_flat = np.matmul(inv_K, uv1_flat)

    return uv1_flat, xy1_flat

def set_from_intr(intr, batch_size, device=None):

    to_torch = True
    uv_grid = gen_uv_grid(intr.width, intr.height, to_torch) # 2*H*W

    K = intr.K_unit.copy()
    K[0,:] = K[0,:] * float(intr.width)
    K[1,:] = K[1,:] * float(intr.height)
    inv_K = np.linalg.inv(K)
    if to_torch:
        inv_K = torch.from_numpy(inv_K)
        K = torch.from_numpy(K)

    uv1_flat, xy1_flat = xy1_from_uv(uv_grid, inv_K, to_torch)  # 3*N

    uv1_grid = uv1_flat.reshape(3, uv_grid.shape[1], uv_grid.shape[2] ) # 3*H*W
    xy1_grid = xy1_flat.reshape(3, uv_grid.shape[1], uv_grid.shape[2] )

    xy1_grid = xy1_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1) # B*3*H*W
    if device is not None:
        xy1_grid = xy1_grid.to(device=device)

    uvb_grid = uv1_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1) # B*3*H*W
    for ib in range(batch_size):
        uvb_grid[ib, 2, :, :] = ib
    if device is not None:
        uvb_grid = uvb_grid.to(device=device)

    K = K.unsqueeze(0).repeat(batch_size, 1, 1) # B*3*3
    if device is not None:
        K = K.to(device=device)

    width_cur = intr.width
    height_cur = intr.height

    return uvb_grid, xy1_grid, width_cur, height_cur, K

def init_pc3d():
    pcl_c3d = EasyDict()
    pcl_c3d.flat = EasyDict()
    pcl_c3d.grid = EasyDict()
    pcl_c3d.flat.uvb = None
    pcl_c3d.flat.feature = EasyDict()
    pcl_c3d.grid.mask = None
    pcl_c3d.grid.feature = EasyDict() 
    return pcl_c3d 

def load_simp_pc3d(pcl_c3d, mask_grid, uvb_flat, feat_grid, feat_flat):
    batch_size = mask_grid.shape[0]

    ## grid features
    pcl_c3d.grid.mask = mask_grid
    for feat in feat_grid:
        pcl_c3d.grid.feature[feat] = feat_grid[feat]   # B*C*H*W

    ## flat features
    pcl_c3d.flat.uvb = []
    for feat in feat_flat:
        pcl_c3d.flat.feature[feat] = []

    mask_flat = mask_grid.reshape(batch_size, 1, -1)
    for ib in range(batch_size):
        mask_vec = mask_flat[ib, 0]
        pcl_c3d.flat.uvb.append(uvb_flat[[ib]][:,:, mask_vec])
        for feat in feat_flat:
            pcl_c3d.flat.feature[feat].append(feat_flat[feat][[ib]][:,:, mask_vec])      # 1*C*N
    
    pcl_c3d.flat.uvb = torch.cat(pcl_c3d.flat.uvb, dim=2)
    for feat in feat_flat:
        pcl_c3d.flat.feature[feat] = torch.cat(pcl_c3d.flat.feature[feat], dim=2)

    return pcl_c3d

def load_pc3d(pcl_c3d, depth_grid, mask_grid, xy1_grid, uvb_flat, K_cur, feat_comm_grid, feat_comm_flat, sparse, use_normal, sparse_nml_opts=None, dense_nml_op=None):
    assert not (sparse_nml_opts is None and dense_nml_op is None)
    """sparse is a bool
    """
    feat_flat = EasyDict()
    feat_grid = EasyDict()

    for feat in feat_comm_flat:
        feat_flat[feat] = feat_comm_flat[feat]
    for feat in feat_comm_grid:
        feat_grid[feat] = feat_comm_grid[feat]

    ## xyz
    xyz_grid = xy1_grid * depth_grid

    batch_size = depth_grid.shape[0]
    xyz_flat = xyz_grid.reshape(batch_size, 3, -1)

    feat_flat['xyz'] = xyz_flat
    feat_grid['xyz'] = xyz_grid

    ## normal for dense
    if use_normal>0 and not sparse:
        normal_grid = dense_nml_op(depth_grid, K_cur)
        nres_grid = res_normal_dense(xyz_grid, normal_grid, K_cur)
        feat_grid['normal'] = normal_grid
        feat_grid['nres'] = nres_grid
        feat_flat['normal'] = normal_grid.reshape(batch_size, 3, -1)
        feat_flat['nres'] = nres_grid.reshape(batch_size, 1, -1)     
    
    ## load into pc3d object
    pcl_c3d = load_simp_pc3d(pcl_c3d, mask_grid, uvb_flat, feat_grid, feat_flat)

    ## normal for sparse
    if use_normal>0 and sparse:
        normal_flat, nres_flat = calc_normal(pcl_c3d.flat.uvb, xyz_grid, mask_grid, sparse_nml_opts.normal_nrange, sparse_nml_opts.ignore_ib, sparse_nml_opts.min_dist_2)
        ## TODO: How to deal with points with no normal?
        uvb_split = pcl_c3d.flat.uvb.to(dtype=torch.long).squeeze(0).transpose(0,1).split(1,dim=1) # a tuple of 3 elements of tensor N*1, only long/byte/bool tensors can be used as indices
        grid_xyz_shape = xyz_grid.shape
        normal_grid = grid_from_concat_flat_func(uvb_split, normal_flat, grid_xyz_shape)
        nres_grid = grid_from_concat_flat_func(uvb_split, nres_flat, grid_xyz_shape)

        pcl_c3d.flat.feature['normal'] = normal_flat
        pcl_c3d.flat.feature['nres'] = nres_flat
        pcl_c3d.grid.feature['normal'] = normal_grid
        pcl_c3d.grid.feature['nres'] = nres_grid

    return pcl_c3d

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def preload_K(data_root):
    "Designed for KITTI dataset. Preload intrinsic params, which is different for each date"
    dates = os.listdir(data_root)
    K_dict = {}
    for date in dates:
        cam_intr_file = os.path.join(data_root, date, 'calib_cam_to_cam.txt')
        intr = read_calib_file(cam_intr_file)
        im_shape = intr["S_rect_02"][::-1].astype(np.int32) ## ZMH: [height, width]
        for side in [2,3]:
            K_dict[(date, side)] = EasyDict()
            P_rect = intr['P_rect_0'+str(side)].reshape(3, 4)
            K = P_rect[:, :3]
            K_unit = np.identity(3).astype(np.float32)
            K_unit[0] = K[0] / float(im_shape[1])
            K_unit[1] = K[1] / float(im_shape[0])

            K_dict[(date, side)].width = im_shape[1]
            K_dict[(date, side)].height = im_shape[0]
            K_dict[(date, side)].K_unit = K_unit
    return K_dict

class C3DLoss(nn.Module):
    def __init__(self, data_root, batch_size=None):
        super(C3DLoss, self).__init__()
        self.width = {}
        self.height = {}
        # self.K = {}
        # self.uvb_flat = {}
        # self.xy1_flat = {}
        
        intr_dict = preload_K(data_root)
        for intr in intr_dict:
            uvb_grid, xy1_grid, self.width[intr], self.height[intr], K = set_from_intr(intr_dict[intr], batch_size)
            self.register_buffer("uvb_grid_{}_{}".format(intr[0], intr[1]), uvb_grid)
            self.register_buffer("xy1_grid_{}_{}".format(intr[0], intr[1]), xy1_grid)
            self.register_buffer("K_{}_{}".format(intr[0], intr[1]), K)

        self.feat_inp_self = ["xyz", "hsv"]
        self.feat_inp_cross = ["xyz", "hsv"]

        self.normal_op_dense = NormalFromDepthDense()
        self.batch_size = batch_size

    def K(self, date_side):
        date = date_side[0]
        side = date_side[1]
        return self.__getattr__("K_{}_{}".format(date, side))
    def uvb_grid(self, date_side):
        date = date_side[0]
        side = date_side[1]
        return self.__getattr__("uvb_grid_{}_{}".format(date, side))
        # return getattr(self, "uvb_grid_{}_{}".format(date, side))
    def xy1_grid(self, date_side):
        date = date_side[0]
        side = date_side[1]
        return self.__getattr__("xy1_grid_{}_{}".format(date, side))

    def parse_opts(self, inputs=None):
        parser = argparse.ArgumentParser(description='Options for continuous 3D loss')

        ## switch for enabling CVO loss
        parser.add_argument("--ell_basedist",          type=float, default=0,
                            help="if not zero, the length scale is proportional to the depth of gt points when the depth is larger than this value. If zero, ell is constant")
        parser.add_argument("--ell_keys",              nargs="+", type=str, default=['xyz', 'hsv'], 
                            help="keys of ells corresponding to ell_values")
        parser.add_argument("--ell_values_min",        nargs="+", type=float, default=[0.05, 0.1], 
                            help="min values of ells corresponding to ell_keys")
        parser.add_argument("--ell_values_rand",       nargs="+", type=float, default=[0.1, 0], 
                            help="parameter of randomness for values of ells corresponding to ell_keys")

        parser.add_argument("--use_normal",            type=int, default=0, 
                            help="if set, normal vectors of sparse pcls are from PtSampleInGridCalcNormal, while those of dense images are from NormalFromDepthDense")
        parser.add_argument("--neg_nkern_to_zero",     action="store_true",
                            help="if set, negative normal kernels are truncated to zero, otherwise use absolute value of normel kernel")
        parser.add_argument("--norm_in_dist",          action="store_true", 
                            help="if set, the normal information will be used in exp kernel besides as a coefficient term. Neet use_normal_v2 to be true to be effective")
        parser.add_argument("--res_mag_min",           type=float, default=0.1,
                            help="the minimum value for the normal kernel (or viewing it as a coefficient of geometric kernel)")
        parser.add_argument("--res_mag_max",           type=float, default=2,
                            help="the maximum value for the normal kernel (or viewing it as a coefficient of geometric kernel)")

        parser.add_argument("--neighbor_range",        type=int, default=2,
                            help="neighbor range when calculating inner product")
        parser.add_argument("--normal_nrange",         type=int, default=5,
                            help="neighbor range when calculating normal direction on sparse point cloud")

        self.opts, rest = parser.parse_known_args(args=inputs) # inputs can be None, in which case _sys.argv[1:] are parsed

        self.opts.ell_min = {}
        self.opts.ell_rand = {}
        for i, ell_item in enumerate(self.opts.ell_keys):
            self.opts.ell_min[ell_item] = self.opts.ell_values_min[i]
            self.opts.ell_rand[ell_item] = self.opts.ell_values_rand[i]

        self.nml_opts = EasyDict() # nml_opts.neighbor_range, nml_opts.ignore_ib, nml_opts.min_dist_2
        self.nml_opts.normal_nrange = int(self.opts.normal_nrange)
        self.nml_opts.ignore_ib = False
        self.nml_opts.min_dist_2 = 0.05

        return rest

    def forward(self, rgb, depth, depth_gt, depth_mask, depth_gt_mask, date_side=None, xy_crop=None, intr=None, nkern_fname=None):
        """
        rgb: B*3*H*W
        depth, depth_gt, depth_mask, depth_gt_mask: B*1*H*W
        """
        assert date_side is not None or intr is not None
        date_side = (date_side[0][0], int(date_side[1][0]) ) # originally it is a list. Take the first since the mini_batch share the same intrinsics. 

        batch_size = rgb.shape[0]       ## if drop_last is False in Sampler/DataLoader, then the batch_size is not constant. 
        if intr is not None:
            uvb_grid_cur, xy1_grid_cur, width_cur, height_cur, K_cur = set_from_intr(intr, batch_size, device=rgb.device)
        else:
            xy1_grid_cur = self.xy1_grid(date_side)
            uvb_grid_cur = self.uvb_grid(date_side)
            width_cur = self.width[date_side]
            height_cur = self.height[date_side]
            K_cur = self.K(date_side)

        ## In case the batch_size is not constant as originally set
        if batch_size != self.batch_size:
            xy1_grid_cur = xy1_grid_cur[:batch_size]
            uvb_grid_cur = uvb_grid_cur[:batch_size]
            K_cur = K_cur[:batch_size]

        if xy_crop is not None:
            x_size = xy_crop[2][0]
            y_size = xy_crop[3][0]
            uvb_grid_crop = uvb_grid_cur[:batch_size,:,:y_size, :x_size]
            xy1_grid_crop = torch.zeros((batch_size, 3, y_size, x_size), device=rgb.device, dtype=torch.float32)
            K_crop = torch.zeros((batch_size, 3, 3), device=rgb.device, dtype=torch.float32)
            for ib in range(batch_size):
                x_start = xy_crop[0][ib]
                y_start = xy_crop[1][ib]
                x_size = xy_crop[2][ib]
                y_size = xy_crop[3][ib]
                xy1_grid_crop[ib] = xy1_grid_cur[ib,:,y_start:y_start+y_size, x_start:x_start+x_size]
                K_crop[ib] = K_cur[ib]
                K_crop[ib, 0, 2] = K_crop[ib, 0, 2] - x_start
                K_crop[ib, 1, 2] = K_crop[ib, 1, 2] - y_start
            K_cur = K_crop
            width_cur = x_size
            height_cur = y_size
            xy1_grid_cur = xy1_grid_crop
            uvb_grid_cur = uvb_grid_crop

        uvb_flat_cur = uvb_grid_cur.reshape(batch_size, 3, -1)

        pc3ds = EasyDict()
        pc3ds["gt"] = init_pc3d()
        pc3ds["pred"] = init_pc3d()

        ## rgb to hsv
        hsv = rgb_to_hsv(rgb, flat=False)           # B*3*H*W
        hsv_flat = hsv.reshape(batch_size, 3, -1)   # B*3*N

        feat_comm_grid = {}
        feat_comm_grid['hsv'] = hsv
        feat_comm_flat = {}
        feat_comm_flat['hsv'] = hsv_flat
        
        pc3ds["gt"] = load_pc3d(pc3ds["gt"], depth_gt, depth_gt_mask, xy1_grid_cur, uvb_flat_cur, K_cur, feat_comm_grid, feat_comm_flat, sparse=True, use_normal=self.opts.use_normal, sparse_nml_opts=self.nml_opts)
        pc3ds["pred"] = load_pc3d(pc3ds["pred"], depth, depth_mask, xy1_grid_cur, uvb_flat_cur, K_cur, feat_comm_grid, feat_comm_flat, sparse=False, use_normal=self.opts.use_normal, dense_nml_op=self.normal_op_dense)
        
        ## random ell
        self.ell = {}
        for key in self.opts.ell_keys:
            self.ell[key] = self.opts.ell_min[key] + np.abs(self.opts.ell_rand[key]* np.random.normal()) 

        inp = self.calc_inn_pc3d(pc3ds["gt"], pc3ds["pred"], nkern_fname)
        
        return inp
    
    def calc_inn_pc3d(self, pc3d_sp, pc3d_dn, nkern_fname=None):
        assert pc3d_sp.flat.feature.keys() == pc3d_dn.flat.feature.keys()
        assert pc3d_sp.grid.feature.keys() == pc3d_dn.grid.feature.keys()

        inp_feat_dict = {}

        for feat in self.feat_inp_self:
            if feat == "hsv":
                inp_feat_dict[feat] = PtSampleInGrid.apply(pc3d_sp.flat.uvb.contiguous(), pc3d_sp.flat.feature[feat].contiguous(), pc3d_dn.grid.feature[feat].contiguous(), pc3d_dn.grid.mask.contiguous(), \
                    self.opts.neighbor_range, self.ell[feat], False, False, self.opts.ell_basedist) # ignore_ib=False, sqr=False
            elif feat == "xyz":
                if self.opts.use_normal > 0:
                    if nkern_fname is None:
                        inp_feat_dict[feat] = PtSampleInGridWithNormal.apply(pc3d_sp.flat.uvb.contiguous(), pc3d_sp.flat.feature[feat].contiguous(), pc3d_dn.grid.feature[feat].contiguous(), \
                            pc3d_dn.grid.mask.contiguous(), pc3d_sp.flat.feature['normal'], pc3d_sp.grid.feature['normal'], pc3d_sp.flat.feature['nres'], pc3d_sp.grid.feature['nres'], \
                                self.opts.neighbor_range, self.ell[feat], self.opts.res_mag_max, self.opts.res_mag_min, False, self.opts.norm_in_dist, self.opts.neg_nkern_to_zero, self.opts.ell_basedist, False, None) 
                                # ignore_ib=False, return_nkern=False, filename=None
                    else:
                        inp_feat_dict[feat] = PtSampleInGridWithNormal.apply(pc3d_sp.flat.uvb.contiguous(), pc3d_sp.flat.feature[feat].contiguous(), pc3d_dn.grid.feature[feat].contiguous(), \
                            pc3d_dn.grid.mask.contiguous(), pc3d_sp.flat.feature['normal'], pc3d_sp.grid.feature['normal'], pc3d_sp.flat.feature['nres'], pc3d_sp.grid.feature['nres'], \
                                self.opts.neighbor_range, self.ell[feat], self.opts.res_mag_max, self.opts.res_mag_min, False, self.opts.norm_in_dist, self.opts.neg_nkern_to_zero, self.opts.ell_basedist, True, nkern_fname)
                else:
                    inp_feat_dict[feat] = PtSampleInGrid.apply(pc3d_sp.flat.uvb.contiguous(), pc3d_sp.flat.feature[feat].contiguous(), pc3d_dn.grid.feature[feat].contiguous(), pc3d_dn.grid.mask.contiguous(), \
                        self.opts.neighbor_range, self.ell[feat], False, False, self.opts.ell_basedist) # ignore_ib=False, sqr=False
            elif feat == "normal":
                pass
            elif feat == "panop":
                pass
            elif feat == "seman": 
                pass
            else:
                raise ValueError("feature {} not recognized".format(feat))
                    
        inp = torch.prod( torch.cat([inp_feat_dict[feat] for feat in inp_feat_dict], dim=0), dim=0).sum()

        return inp

    

if __name__ == "__main__":
    pcl_c3d = init_pcl()
    print(pcl_c3d['flat']["feature"]['xyz'])
    # print(pcl_c3d['flat']["feature"]['abc'])
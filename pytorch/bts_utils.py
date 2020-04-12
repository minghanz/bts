import torch
import numpy as np 
import cv2
import os

def vis_depth(depth, ref_depth=10):        ## why normalize_result in bts_main.py convert it to numpy?
    dum_zero = torch.zeros_like(depth)
    inv_depth = torch.where(depth>0, ref_depth/(ref_depth+depth), dum_zero)
    return inv_depth

def overlay_dep_on_rgb(depth, img, path=None, name=None, overlay=False):
    ''' both 3-dim: C*H*W. not including batch
    dep: output from vis_depth
    both are torch.tensor, between 0~1
    '''
    dep_np = depth.cpu().numpy().transpose(1,2,0)
    img_np= img.permute(1,2,0).cpu().numpy()
    return overlay_dep_on_rgb_np(dep_np, img_np, path, name, overlay)

def overlay_dep_on_rgb_np(dep_np, img_np, path=None, name=None, overlay=False):
    ''' both 3-dim: H*W*C.
    both are np.array, between 0~1
    '''
    dep_255 = dep_np*255

    dep_mask = np.zeros_like(dep_255).astype(np.uint8)
    dep_mask[dep_255 > 0] = 255

    r, g, b = rgbmap(dep_255, mask_zeros=True)              # return int, shape the same as input

    dep_in_color = np.dstack((b, g, r))
    dep_in_color = dep_in_color.astype(np.uint8)

    img_np= img_np * 255 # H*W*C
    img_np = img_np.astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    if not overlay:
        img_dep = cv2.add(img_np, dep_in_color)
    else:
        inv_mask = 255 - dep_mask
        img_masked = cv2.bitwise_and(img_np, img_np, mask=inv_mask)     # the mask should be uint8 or int8, single channel
        img_dep = cv2.add(img_masked, dep_in_color)

    if path is not None and name is not None:
        if not os.path.exists(path):
            os.mkdir(path)
        # full_path = os.path.join(path, 'img'+name)
        # cv2.imwrite(full_path, img_np)
        # full_path = os.path.join(path, 'dep'+name)
        # cv2.imwrite(full_path, dep_in_color)
        full_path = os.path.join(path, name)
        cv2.imwrite(full_path, img_dep)
        # full_path = os.path.join(path, 'mask'+name)
        # cv2.imwrite(full_path, dep_mask)

    return img_np

def rgbmap(gray, mask_zeros=False, max_val=-1, min_val=0):
    """
    Assuming input gray is 1D ndarray between [0,255]
    https://www.particleincell.com/2014/colormap/ 
    """
    # ##
    # r = inten
    # g = np.zeros_like(r)
    # b = 255- inten
    # ##

    if mask_zeros:
        valid_mask = gray>0
        invalid_mask = np.invert(valid_mask)
        min_cur = gray[valid_mask].min()
        max_cur = gray[valid_mask].max()
    else:
        min_cur = gray.min()
        max_cur = gray.max()
    
    if max_val == -1:
        ## mode 1: normalize to fulfill the range, excluding points of zero value (deal with them outside of this function)
        gray = (gray.astype(float) - min_cur )/( max_cur - min_cur )  # normalize to fulfill the range
    else:
        ## mode 2: normalize with fixed ratio
        gray = (gray.astype(float) - min_val )/( max_val - min_val )

    if mask_zeros:
        gray[invalid_mask] = 0.5 # this value does not matter as long as it will not cause out-of-range in calculating idx0. It will be set to 0 at the end

    gray_flat = gray.reshape(-1)

    a = (gray_flat)/0.25
    X = np.floor(a)           # group
    Y = np.floor( 255*(a-X) ) # residual
    cand = np.zeros((5,gray_flat.shape[0], 3))
    cand[0,:,0] = 255
    cand[0,:,1] = Y
    cand[0,:,2] = 0

    cand[1,:,0] = 255-Y
    cand[1,:,1] = 255
    cand[1,:,2] = 0

    cand[2,:,0] = 0
    cand[2,:,1] = 255
    cand[2,:,2] = Y

    cand[3,:,0] = 0
    cand[3,:,1] = 255-Y
    cand[3,:,2] = 255

    cand[4,:,0] = 0
    cand[4,:,1] = 0
    cand[4,:,2] = 255
    
    idx0 = X.astype(int)
    idx1 = np.arange(gray_flat.shape[0])

    rgb = cand[idx0, idx1]

    # gray = 0.3 + gray * 0.7
    # r = (rgb[:,0] * gray).astype(int)
    # g = (rgb[:,1] * gray).astype(int)
    # b = (rgb[:,2] * gray).astype(int)
    r = rgb[:,0].astype(int)
    g = rgb[:,1].astype(int)
    b = rgb[:,2].astype(int)

    r = r.reshape(gray.shape)
    g = g.reshape(gray.shape)
    b = b.reshape(gray.shape)

    if mask_zeros:
        r[invalid_mask] = 0
        g[invalid_mask] = 0
        b[invalid_mask] = 0
    
    return r,g,b
import numpy as np 
import os
from PIL import Image
import cv2
from bts_pre_intr import read_calib_file, load_velodyne_points, sub2ind
from collections import Counter
from bts_utils import overlay_dep_on_rgb_np

seq_folder = '{}_drive_{:04d}_sync'

def vis_depth(depth, ref_depth=10):        ## why normalize_result in bts_main.py convert it to numpy?
    dum_zero = np.zeros_like(depth)
    inv_depth = np.where(depth>0, ref_depth/(ref_depth+depth), dum_zero)
    return inv_depth

def read_img(root, date, seq, idx, new_shape=None):
    img_path = os.path.join(root, date, seq_folder.format(date, seq), 'image_02', 'data', '{:010d}.jpg'.format(idx))

    image = Image.open(img_path)
    if new_shape is not None:
        image = image.resize(new_shape[::-1])
    img_np = np.array(image)
    return img_np

def lidar2img(root, date, seq, idx, new_shape=None):
    calib_cam_path = os.path.join(root, date, 'calib_cam_to_cam.txt')
    calib_lid_path = os.path.join(root, date, 'calib_velo_to_cam.txt')
    img_path = os.path.join(root, date, seq_folder.format(date, seq), 'image_02', 'data', '{:010d}.jpg'.format(idx))
    lid_path = os.path.join(root, date, seq_folder.format(date, seq), 'velodyne_points', 'data', '{:010d}.bin'.format(idx))

    intr = read_calib_file(calib_cam_path)
    extr = read_calib_file(calib_lid_path)

    ## get image shape
    S_02 = intr['S_rect_02'][::-1].astype(np.int32) # [height, width]

    ## get K
    P_rect_02 = intr['P_rect_02'].reshape(3,4)
    K = P_rect_02[:, :3]

    ## get translation from 0 to 2
    Kt = P_rect_02[:, [3]]
    K_inv = np.linalg.inv(K)
    t = np.dot(K_inv, Kt)
    P_rect_t = np.identity(4)
    P_rect_t[:3, [3]] = t

    ## get rotation for rectification
    R_rect_00 = intr['R_rect_00'].reshape(3, 3)
    T_rect_00 = np.identity(4)
    T_rect_00[:3, :3] = R_rect_00

    ## get extrinsic transformation from lidar to camera
    T_cam_lidar = np.hstack( (extr['R'].reshape(3, 3), extr['T'].reshape(3, 1)) )
    T_cam_lidar = np.vstack( (T_cam_lidar, np.array([0,0,0,1])) )

    if new_shape is None:
        ## Overall projection matrix
        P_cam_lidar = np.matmul(P_rect_02, np.matmul(T_rect_00, T_cam_lidar))
        im_shape = S_02
    else:
        K_new = K.copy()
        K_new[0] = K[0] / S_02[1] * new_shape[1]
        K_new[1] = K[1] / S_02[0] * new_shape[0]
        K_new = np.hstack( (K_new, np.array([0,0,0]).reshape(3,1)) )
        P_cam_lidar = np.matmul(np.matmul(K_new, P_rect_t), np.matmul(T_rect_00, T_cam_lidar))
        im_shape = new_shape

    ## load lidar points
    pts_lidar = load_velodyne_points(lid_path)
    pts_lidar = pts_lidar[pts_lidar[:, 0] >= 0, :]

    ## project to image (-1 to be consistent with matlab result)
    lidar_prj = np.matmul(P_cam_lidar, pts_lidar.T)
    lidar_prj[:2] = lidar_prj[:2] / lidar_prj[[2]]
    lidar_prj[:2] = np.round(lidar_prj[:2]) - 1

    ## crop out-of-sight points
    valid_idx = ( lidar_prj[0] >= 0 ) & ( lidar_prj[0] < im_shape[1] ) & ( lidar_prj[1] >= 0 ) & ( lidar_prj[1] < im_shape[0] )
    lidar_prj = lidar_prj[:, valid_idx]

    ## compose depth image
    depth_img = np.zeros(im_shape)
    depth_img[lidar_prj[1].astype(np.int), lidar_prj[0].astype(np.int)] = lidar_prj[2]

    ## find the duplicate points and choose the closest depth
    velo_proj_lin = sub2ind(depth_img.shape, lidar_prj[1], lidar_prj[0])
    dupe_proj_lin = [item for item, count in Counter(velo_proj_lin).items() if count > 1]
    for dd in dupe_proj_lin:
        pts = np.where(velo_proj_lin == dd)[0]
        x_loc = int(lidar_prj[0, pts[0]])
        y_loc = int(lidar_prj[1, pts[0]])
        depth_img[y_loc, x_loc] = lidar_prj[2, pts].min()
    depth_img[depth_img < 0] = 0

    return depth_img

def get_n_files(root, date, seq):
    seq_path = os.path.join(root, date, seq_folder.format(date, seq), 'image_02', 'data')
    n_files = len(os.listdir(seq_path))
    return n_files

def gen_single_file(data_root, date, seq, idx, output_root, new_shape):
    rgb = read_img(data_root, date, seq, idx)
    rgb_01 = rgb.astype(np.float32) / 255

    dep = lidar2img(data_root, date, seq, idx)
    inv_dep = vis_depth(dep)

    name = '{}_{}_{}_ori.jpg'.format(date, seq, idx)
    dep_rgb = overlay_dep_on_rgb_np(inv_dep, rgb_01, output_root, name, overlay=True)


    rgb = read_img(data_root, date, seq, idx, new_shape)
    rgb_01 = rgb.astype(np.float32) / 255

    dep = lidar2img(data_root, date, seq, idx, new_shape)
    inv_dep = vis_depth(dep)

    name = '{}_{}_{}_resized.jpg'.format(date, seq, idx)
    dep_rgb = overlay_dep_on_rgb_np(inv_dep, rgb_01, output_root, name, overlay=True)

def dupli_files(ref_root, data_root, output_root, new_shape):
    list_files = os.listdir(ref_root)
    for f in list_files:
        name = f.split('.')[0]
        date_str = name[:10]
        other_str = name[11:]
        seq = int(other_str.split('_')[0])
        idx = int(other_str.split('_')[1])

        gen_single_file(data_root, date_str, seq, idx, output_root, new_shape)


if __name__ == '__main__':
    # save_root = 'dep_img'
    # root = '/media/sda1/minghanz/datasets/kitti/kitti_data'
    # date = '2011_09_26'
    # seq = 5
    # new_shape = (192, 640)
    
    # n_files = get_n_files(root, date, seq)

    # for idx in range(n_files):
    #     print(idx, '/', n_files)
    #     gen_single_file(root, date, seq, idx, save_root, new_shape)

    ref_root = 'models/bts_eigen_v2_pytorch_test/Fri_Apr_10_15:19:48_2020/img_dep'
    data_root = '/media/sda1/minghanz/datasets/kitti/kitti_data'
    output_root = ref_root
    new_shape = (192, 640)
    dupli_files(ref_root, data_root, output_root, new_shape)
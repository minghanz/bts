import numpy as np
import os
from easydict import EasyDict
from collections import Counter

# def read_calib_file(path):
#     """Read KITTI calibration file
#     (from https://github.com/hunse/kitti)
#     """
#     float_chars = set("0123456789.e+- ")
#     data = {}
#     with open(path, 'r') as f:
#         for line in f.readlines():
#             key, value = line.split(':', 1)
#             value = value.strip()
#             data[key] = value
#             if float_chars.issuperset(value):
#                 # try to cast to float array
#                 try:
#                     data[key] = np.array(list(map(float, value.split(' '))))
#                 except ValueError:
#                     # casting error: data[key] already eq. value, so pass
#                     pass

#     return data

# def preload_K(data_root):
#     "Designed for KITTI dataset. Preload intrinsic params, which is different for each date"
#     dates = os.listdir(data_root)
#     K_dict = {}
#     for date in dates:
#         cam_intr_file = os.path.join(data_root, date, 'calib_cam_to_cam.txt')
#         intr = read_calib_file(cam_intr_file)
#         im_shape = intr["S_rect_02"][::-1].astype(np.int32) ## ZMH: [height, width]

#         cam_lidar_extr_file = os.path.join(data_root, date, 'calib_velo_to_cam.txt')
#         extr_li = read_calib_file( cam_lidar_extr_file )

#         for side in [2,3]:
#             K_dict[(date, side)] = EasyDict()
#             P_rect = intr['P_rect_0'+str(side)].reshape(3, 4)
#             K = P_rect[:, :3]
#             K_unit = np.identity(3).astype(np.float32)
#             K_unit[0] = K[0] / float(im_shape[1])
#             K_unit[1] = K[1] / float(im_shape[0])

#             T_cam_lidar = np.hstack((extr_li['R'].reshape(3, 3), extr_li['T'][..., np.newaxis]))
#             T_cam_lidar = np.vstack((T_cam_lidar, np.array([0, 0, 0, 1.0])))
#             R_rect_cam = np.eye(4)
#             R_rect_cam[:3, :3] = intr['R_rect_00'].reshape(3, 3)

#             K_inv = np.linalg.inv(K)
#             Kt = P_rect[:, 3:4]
#             t = np.dot(K_inv, Kt)
#             P_rect_t = np.identity(4)
#             P_rect_t[:3, 3:4] = t # ZMH: 4*4
            
#             P_rect_li = np.dot(P_rect_t, np.dot(R_rect_cam, T_cam_lidar))

#             K_dict[(date, side)].width = im_shape[1]
#             K_dict[(date, side)].height = im_shape[0]
#             K_dict[(date, side)].K_unit = K_unit
#             K_dict[(date, side)].P_cam_li = P_rect_li 
#     return K_dict

# def sub2ind(matrixSize, rowSub, colSub):
#     """Convert row, col matrix subscripts to linear indices
#     """
#     m, n = matrixSize
#     return rowSub * (n-1) + colSub - 1

# def load_velodyne_points(filename):
#     """Load 3D point cloud from KITTI file format
#     (adapted from https://github.com/hunse/kitti)
#     """
#     points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
#     points[:, 3] = 1.0  # homogeneous
#     return points

# def lidar_to_depth(velo, extr_cam_li, K_unit, im_shape):
#     """extr_cam_li: 4x4, intr_K: 3x3"""
#     ## recover K
#     intr_K = K_unit.copy()
#     intr_K[0, :] = K_unit[0, :] * float(im_shape[1])
#     intr_K[1, :] = K_unit[1, :] * float(im_shape[0])

#     ## transform to camera frame
#     velo_in_cam_frame = np.dot(extr_cam_li, velo.T).T # N*4
#     velo_in_cam_frame = velo_in_cam_frame[:, :3] # N*3, xyz

#     ## project to image
#     velo_proj = np.dot(intr_K, velo_in_cam_frame.T).T
#     velo_proj[:, :2] = velo_proj[:, :2] / velo_proj[:, [2]]
#     velo_proj[:, :2] = np.round(velo_proj[:, :2]) -1    # -1 is for kitti dataset aligning with its matlab script

#     ## crop out-of-view points
#     valid_idx = ( velo_proj[:, 0] >= 0 ) & ( velo_proj[:, 0] < im_shape[1] ) & ( velo_proj[:, 1] >= 0 ) & ( velo_proj[:, 1] < im_shape[0] )
#     velo_proj = velo_proj[valid_idx, :]

#     ## compose depth image
#     depth_img = np.zeros((im_shape[:2]))
#     depth_img[velo_proj[:, 1].astype(np.int), velo_proj[:, 0].astype(np.int)] = velo_proj[:, 2]

#     ## find the duplicate points and choose the closest depth
#     velo_proj_lin = sub2ind(depth_img.shape, velo_proj[:, 1], velo_proj[:, 0])
#     dupe_proj_lin = [item for item, count in Counter(velo_proj_lin).items() if count > 1]
#     for dd in dupe_proj_lin:
#         pts = np.where(velo_proj_lin == dd)[0]
#         x_loc = int(velo_proj[pts[0], 0])
#         y_loc = int(velo_proj[pts[0], 1])
#         depth_img[y_loc, x_loc] = velo_proj[pts, 2].min()
#     depth_img[depth_img < 0] = 0

#     return depth_img
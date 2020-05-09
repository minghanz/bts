##### check whether train and validation files will duplicate when we use next or previous frame in the training/validation set
##### it turns out that the training and validation files do not share the same sequences
##### but the train and val files are mostly sequential. It does not matter since monodepth2 also use partially overlapped sequential tracklets 

##### train file in bts generally corresponds to train file + val file in monodepth2/eigen_zhou
##### test file in bts and monodepth2/eigen_benchmark are the same
##### train files in bts are all sequential, with refined depth (but starting from frame 5, because 11 frames together produce 1 refined depth)
##### train and val files in monodepth2/eigen_zhou does not include static frames, may not have refined depth
##### train and val files in monodepth2/eigen_zhou_w_dense does not include static frames, have refined depth

import os
import numpy as np 

def process_path(imgpath):
    path_strs = imgpath.split('/')
    date_str = path_strs[0]
    seq_str = path_strs[1]
    seq_n = int(seq_str.split('_drive_')[1].split('_')[0])  # integer of the sequence number
    side = int(path_strs[2].split('_')[1])
    frame = int(path_strs[-1].split('.')[0])
    return date_str, seq_n, side, frame

def process_seq_path(seq_path):
    path_strs = seq_path.split('/')
    date_str = path_strs[0]
    seq_str = path_strs[1]
    seq_n = int(seq_str.split('_drive_')[1].split('_')[0])  # integer of the sequence number
    return date_str, seq_n

def process_line(line):
    img, dep, focal = line.split()
    return process_path(img)

def process_line_mn2(line):
    frames = {}
    seq_path, frame_n, lr = line.split()
    date_str, seq_n = process_seq_path(seq_path)
    frame_n = int(frame_n)
    lr_dict = {'l': 2, 'r': 3}
    side = lr_dict[lr]
    return date_str, seq_n, side, frame_n

def lines2frames(lines, mode):
    frames = {}
    line_seq = {}
    for i, line in enumerate(lines):
        if mode == 'bts':
            date, seq, side, frame = process_line(line)
        elif mode == 'monodepth2':
            date, seq, side, frame = process_line_mn2(line)
        else:
            raise ValueError('mode not recognized: {}'.format(mode))

        if (date, seq, side) not in frames:
            frames[(date, seq, side)] = []
            line_seq[(date, seq, side)] = []
        frames[(date, seq, side)].append(frame)
        line_seq[(date, seq, side)].append(i)
    return frames, line_seq

def file2frames(filename, mode):
    with open(filename) as f:
        lines = f.readlines()
    
    return lines2frames(lines, mode)

def check_dup(frames1, frames2):
    dup = []
    for date_seq in frames1:
        for frame in frames1[date_seq]:
            # print(frame)
            # if frame+1 in frames1[date_seq]:
            #     dup.append((date_seq, frame))
            if date_seq in frames2:
                # dup.append(date_seq)
                if frame in frames2[date_seq]:# or frame+1 in frames2[date_seq] or frame-1 in frames2[date_seq] or frame+2 in frames2[date_seq] or frame-2 in frames2[date_seq]:
                    dup.append((date_seq, frame))
    return dup

def check_ctn_and_sort(frames, lineseq):
    for date_seq in frames:
        assert date_seq in lineseq
        zipsort = sorted(zip(frames[date_seq], lineseq[date_seq]))
        frames[date_seq], lineseq[date_seq] = list(zip(*(zipsort)))
        frames[date_seq] = list(frames[date_seq])
        lineseq[date_seq] = list(lineseq[date_seq])
        # print(frames[date_seq], lineseq[date_seq] )
        # print(frames[date_seq])
        # for i, frame in enumerate(frames[date_seq]):
        #     if i < len(frames[date_seq]) - 1:
        #         if frames[date_seq][i+1] != frame + 1:
        #             print(date_seq, frame, frames[date_seq][i+1])

# 原 dataset：
# 有 refined depth
# 新 dataset:
# 有 refined depth
# not static
# monodepth2/eigen_zhou/train+val dataset: 
# not static
# has prev and next frame
# monodepth2/eigen_zhou_w_dense/train+val dataset: 
# not static
# prev and next frame has refined depth
    
def take_intersection(frames_itc, lineseq_itc, frames1, lineseq1, frames2, lineseq2):
    for date_seq in frames1:
        if date_seq in frames2:
            if date_seq not in frames_itc:
                frames_itc[date_seq] = []
                lineseq_itc[date_seq] = []
            for i, frame in enumerate(frames1[date_seq]):
                if frame in frames2[date_seq]:
                    frames_itc[date_seq].append(frame)
                    lineseq_itc[date_seq].append(lineseq1[date_seq][i])

def write_file_from_file(frames, lineseq, ori_file, out_file):
    with open(ori_file) as f:
        ori_lines = f.readlines()

    with open(out_file, 'w') as f:
        for date_seq in lineseq:
            for i in lineseq[date_seq]:
                f.write(ori_lines[i])

if __name__ == "__main__":
    mode = 'monodepth2' #'monodepth2' 'bts'

    train_file = {}
    val_file = {}
    test_file = {}
    train_file['bts'] = '/root/repos/bts/train_test_inputs/eigen_train_files_with_gt_jpg_fullpath.txt'
    test_file['bts'] = '/root/repos/bts/train_test_inputs/eigen_test_files_with_gt_jpg_fullpath.txt'
    train_file['monodepth2'] = '/root/repos/monodepth2/splits/eigen_zhou_w_dense/train_files.txt'
    val_file['monodepth2'] = '/root/repos/monodepth2/splits/eigen_zhou_w_dense/val_files.txt'
    test_file['monodepth2'] = '/root/repos/monodepth2/splits/eigen_benchmark/test_files.txt'


    train_bts_frames, train_bts_lineseq = file2frames(train_file['bts'], 'bts')

    train_mn2_frames, train_mn2_lineseq = file2frames(train_file['monodepth2'], 'monodepth2')

    val_mn2_frames, val_mn2_lineseq = file2frames(val_file['monodepth2'], 'monodepth2')


    # dup = check_dup(train_frames, test_frames )
    # print(len(dup))

    # check_ctn_and_sort(train_frames, train_lineseq)

    frames_itc = {}
    lineseq_itc = {}
    take_intersection(frames_itc, lineseq_itc, train_bts_frames, train_bts_lineseq, train_mn2_frames, train_mn2_lineseq)
    take_intersection(frames_itc, lineseq_itc, train_bts_frames, train_bts_lineseq, val_mn2_frames, val_mn2_lineseq)

    check_ctn_and_sort(frames_itc, lineseq_itc)

    out_file = '/root/repos/bts/train_test_inputs/eigen_train_files_with_gt_nonstatic_jpg_fullpath.txt'
    write_file_from_file(frames_itc, lineseq_itc, train_file['bts'], out_file)
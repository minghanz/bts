### this file is to generate eigen_test_files_with_gt_jpg_fullpath.txt
import os

filename = 'eigen_test_files_with_gt_jpg.txt'
newfile = 'eigen_test_files_with_gt_jpg_fullpath.txt'
with open(filename) as f:
    lines = f.readlines()
    with open(newfile, 'w') as g:
        new_lines = []
        for line in lines:
            strs = line.split()
            if strs[1] != "None":
                date = strs[0].split("/")[0]
                new_gt_path = date + "/" + strs[1]
                new_line = "{} {} {}\n".format(strs[0], new_gt_path, strs[2])
            else:
                new_line = line
            new_lines.append(new_line)
        g.writelines(new_lines)
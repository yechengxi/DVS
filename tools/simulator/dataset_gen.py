#!/usr/bin/python3


import argparse
import numpy as np
import cv2
import os, sys, signal, glob, time
import pydvs

from utils import *

global_shape = (200, 346)
global_scale_pn = 100
global_scale_pp = 100
slice_width = 1


exr_img = OpenEXR.InputFile('./ev_datasets/data/exr/0002.exr')

print exr_img.header()

z = extract_depth(exr_img)
img = extract_grayscale(exr_img)

print np.min(z), np.max(z)

cv2.imwrite('/home/alice/Desktop/depth.png', z * 10)
cv2.imwrite('/home/alice/Desktop/grayscale.png', img * 255)


def read_calib(fname):
    print ("Reading camera calibration params from: ", fname)
    K = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    D = np.array([0.0, 0.0, 0.0, 0.0])

    lines = []
    with open(fname) as calib:
        lines = calib.readlines()

    K_txt = lines[0:3]
    D_txt = lines[4]
    
    for i, line in enumerate(K_txt):
        for j, num_txt in enumerate(line.split(' ')[0:3]):
            K[i][j] = float(num_txt)

    for j, num_txt in enumerate(D_txt.split(' ')[0:4]):
        D[j] = float(num_txt)

    return K, D


def undistort_img(img, K, D):
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted


def dvs_img(cloud, shape, K, D):
    fcloud = cloud.astype(np.float32) # Important!

    cmb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    pydvs.dvs_img(fcloud, cmb)
    cmb[:,:,0] *= global_scale_pp
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= global_scale_pn

    cmb = undistort_img(cmb, K, D)
    return cmb


def get_mask_paths(folder_path, every_nth):
    file_list = os.listdir(folder_path)
    ret = {}
    for f in file_list:
        if '.png' not in f:
            continue
        comp = f.split('_')
        if (comp[0] != 'mask'):
            continue
        id_ int(comp[1])
        num = int (comp[2].split('.')[0])
    
        if (num % every_nth != 0):
            continue

        if (id_ not in ret.keys())
            ret[id_] = []

        ret[id_].append(os.path.join(folder_path, f))
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder',
                        type=str,
                        required=True)
    parser.add_argument('--oname',
                        type=str,
                        required=True)
    parser.add_argument('--fps',
                        nargs='+',
                        type=int,
                        default=[1000, 20],
                        required=False)
    parser.add_argument('--width',
                        type=float,
                        default=0.05,
                        required=False)

    args = parser.parse_args()

    print ("Opening", args.base_folder)
	slice_width = args.width

    K, D = read_calib()
    cloud, idx = get_cloud(os.path.join(args.base_folder, 'events.txt'))
    
    frame_step = args.fps[0] / args.fps[1]
    time_step = float(frame_step) / float()
    mask_paths = get_mask_paths(args.fps[0])

    print ("Using every", frame_step, '\'th frame, step is', time_step, 'seconds')

    sl = cloud[idx[args.bounds[0]]:idx[args.bounds[1]]]


    slice_width = sl[-1][0] - sl[0][0]

    cmb = dvs_img(sl, global_shape)
    cv2.imwrite(args.img_name, cmb)

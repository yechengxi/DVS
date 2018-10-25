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

    cam_file = open(fname)
    cam_data = yaml.safe_load(cam_file)
    K[0][0] = cam_data['cam_fx']
    K[1][1] = cam_data['cam_fy']
    K[0][2] = cam_data['cam_cx']
    K[1][2] = cam_data['cam_cy']
    cam_file.close()

    return K, D


def get_cloud(fname, index_w=0.01):
    cloud = np.loadtxt(fname, dtype=np.float32)
    idx = [0]
    if (cloud.shape[0] < 1):
        return cloud, idx

    last_ts = cloud[0][0]
    for i, e in enumerate(cloud):
        while (e[0] - last_ts > index_w):
            idx.append(i)
            last_ts += index_w
    return cloud, idx


def get_slice(cloud, idx, ts, width, mode=1, idx_step=0.01):
    ts_lo = ts
    ts_hi = ts + width
    if (mode == 1):
        ts_lo = ts - width / 2.0
        ts_hi = ts + width / 2.0
    if (mode == 2):
        ts_lo = ts - width
        ts_hi = ts
    if (mode > 2 or mode < 0):
        print ("Wrong mode! Reverting to default...")
    if (ts_lo < 0): ts_lo = 0

    t0 = cloud[0][0]

    idx_lo = int((ts_lo - t0) / idx_step)
    idx_hi = int((ts_hi - t0) / idx_step)
    if (idx_lo >= len(idx)): idx_lo = -1
    if (idx_hi >= len(idx)): idx_hi = -1

    sl = np.copy(cloud[idx[idx_lo]:idx[idx_hi]].astype(np.float32))
    idx_ = np.copy(idx[idx_lo:idx_hi])

    if (idx_lo == idx_hi):
        return sl, np.array([0])

    idx_0 = idx_[0]
    idx_ -= idx_0

    t0 = sl[0][0]
    sl[:,0] -= t0

    return sl, idx_


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


def get_mask_paths(folder_path, every_nth, dt):
    file_list = os.listdir(folder_path)
    ret = {}
    i = 0
    for f in file_list:
        if '.png' not in f:
            continue
        comp = f.split('_')
        if (comp[0] != 'mask'):
            continue
        id_ int(comp[1])
        num = int(comp[2].split('.')[0])
    
        if (num % every_nth != 0):
            continue

        if (id_ not in ret.keys())
            ret[id_] = {}

        time = float(num) * float(dt)
        ret[id_][i] = [os.path.join(folder_path, f), time]
        i += 1
    return ret


def get_exr_paths(folder_path, every_nth, dt):
    file_list = os.listdir(folder_path)
    ret = {}
    i = 0
    for f in file_list:
        if '.exr' not in f:
            continue
        comp = f.split('.')
        num = int(comp[0])

        if (num % every_nth != 0):
            continue

        time = float(num) * float(dt)
        ret[i] = [os.path.join(folder_path, f), time]
        i += 1
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

    K, D = read_calib(os.path.join(args.base_folder, 'rendered', 'camera.yaml'))
    cloud, idx = get_cloud(os.path.join(args.base_folder, 'events.txt'))

    frame_step = args.fps[0] / args.fps[1]
    time_step = float(frame_step) / float(args.fps[0])
    mask_paths = get_mask_paths(os.path.join(args.base_folder, 'rendered', 'masks'), frame_step, 1.0 / float(args.fps[0]))
    exr_paths  = get_exr_paths(os.path.join(args.base_folder, 'rendered', 'exr'), frame_step, 1.0 / float(args.fps[0]))

    print ("Using every", frame_step, '\'th frame, step is', time_step, 'seconds')

    nframes = len(exr_paths)
    oids = mask_paths.keys()
    for i in range(nframes):
        time = exr_paths[i][1]
        exr_img = OpenEXR.InputFile(exr_paths[i][0])
        print ("Processing time", time, "frame", i, "out of", nframes)

        sl = get_slice(cloud, idx, time, args.width)
        cmb = dvs_img(sl, global_shape) 
    
        masks = []
        for id_ in sorted(oids):
            ts = mask_paths[id_][i][1]
            if (abs(ts - time) > 0.01):
                print ("Critical Error! gt timestamps do not match; time =", time, "but ts = ", ts, "mask id is", id_)
            mask = cv2.imread(mask_paths[id_][i][0], 0)
            masks.append(mask)

        z = extract_depth(exr_img)
        img = extract_grayscale(exr_img)



    cv2.imwrite(args.img_name, cmb)

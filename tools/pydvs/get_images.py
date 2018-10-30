#!/usr/bin/python3

import argparse
import numpy as np
import cv2
import os, sys, signal, glob, time
import matplotlib.colors as colors

# The dvs-related stuff, implemented in C. 
sys.path.insert(0, './build/lib.linux-x86_64-3.5') #The pydvs.so should be in PYTHONPATH!
import pydvs

global_scale_pn = 100
global_scale_pp = 100
global_shape = (200, 346)


def undistort_img(img, K, D):
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted


def dvs_img(cloud, shape, K, D):
    fcloud = cloud.astype(np.float32) # Important!

    slice_width = cloud[-1][0] - cloud[0][0]

    cmb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    pydvs.dvs_img(fcloud, cmb)
    cnt = cmb[:,:,0] + cmb[:,:,2]

    cmb[:,:,0] *= global_scale_pp
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= global_scale_pn

    cmb = undistort_img(cmb, K, D)
    return cmb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice',
                        type=str,
                        required=True)
    parser.add_argument('--odir',
                        type=str,
                        required=True)
    parser.add_argument('--dims',
                        nargs='+',
                        type=int,
                        required=True)
    parser.add_argument('--width', 
                        type=int, 
                        required=True)

    args = parser.parse_args()

    print ("Opening", args.slice)

    sl_npz = np.load(args.slice)
    cloud = sl_npz['events']
    idx   = sl_npz['index']
    K              = sl_npz['K']
    D              = sl_npz['D']

    sl = cloud[idx[0]:idx[args.width]]

    if (True):
        width = cloud[-1][0] - cloud[0][0]
        print ("Input cloud:")
        print ("\tWidth: ", width, "seconds and", len(cloud), "events.")
        print ("\tIndex size: ", len(idx), "points, step = ", width / float(len(idx) + 1), "seconds.")
        print ("")
        width = sl[-1][0] - sl[0][0]
        print ("Chosen slice:")
        print ("\tWidth: ", width, "seconds and", len(sl), "events.")
        print ("")

    n = 0
    rb = args.width
    while (rb < len(idx)):
        ver = []
        for i in range(args.dims[1]):
            hor = []
            for j in range(args.dims[0]):
                cmb = []
                if (rb < len(idx)):
                    sl = cloud[idx[rb - args.width]:idx[rb]]
                    cmb = dvs_img(sl, global_shape, K, D)
                else:
                    cmb = np.zeros((global_shape[0], global_shape[1], 3), dtype=np.float32) 
                if (hor == []):
                    hor = cmb
                else:
                    hor = np.hstack((hor, cmb))
                rb += args.width
            if (ver == []):
                ver = hor
            else:
                ver = np.vstack((ver, hor))
        cv2.imwrite(args.odir + '/image_' + str(n).rjust(10, '0') + '.png', ver)
        n += 1

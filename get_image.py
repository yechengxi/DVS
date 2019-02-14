#!/usr/bin/python3

import argparse
import numpy as np
import cv2
import os, sys, signal, glob, time

# The dvs-related stuff, implemented in C. 
sys.path.insert(0, './pydvs/build/lib.linux-x86_64-3.6') #The libdvs.so should be in PYTHONPATH!
import libdvs

global_scale_t = 20 * 255
global_scale_pn = 100
global_scale_pp = 100
global_shape = (200, 346)
slice_width = 1

def dvs_img(cloud, shape):
    fcloud = cloud.astype(np.float32) # Important!

    c_start = time.time()

    cmb = np.zeros((global_shape[0], global_shape[1], 3), dtype=np.float32)
    libdvs.dvs_img(fcloud, cmb)
    cmb[:,:,0] *= global_scale_pp
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= global_scale_pn

    c_end = time.time()
    return cmb

    py_start = time.time()

    t0 = min(cloud[0][0], cloud[-1][0])
    timg = np.zeros(shape, dtype=np.float)
    cimg = np.zeros(shape, dtype=np.float)
    nimg = np.zeros(shape, dtype=np.float)
    pimg = np.zeros(shape, dtype=np.float)

    for e in cloud:
        x = int(e[1])
        y = int(e[2])
        p = 0
        if (e[3] > 0.5): 
            p = 1

        if (y >= shape[0] or x >= shape[1]):
            continue

        cimg[y, x] += 1
        timg[y, x] += (e[0] - t0)
        if (p > 0):
            nimg[y, x] += 1
        else:
            pimg[y, x] += 1

    timg = np.divide(timg, cimg, out=np.zeros_like(timg), where=cimg!=0)

    cmb = np.dstack((nimg * global_scale_pp, timg * 255 / slice_width, pimg * global_scale_pn))

    py_end = time.time()
    
    
    print ("Python:", py_end - py_start, "C:", c_end - c_start)


    return cmb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice',
                        type=str,
                        required=True)
    parser.add_argument('--img_name',
                        type=str,
                        required=True)
    parser.add_argument('--bounds',
                        nargs='+',
                        type=int,
                        default=[0, -1],
                        required=False)
    parser.add_argument('--info', 
                        action='store_true', 
                        required=False)

    args = parser.parse_args()

    print ("Opening", args.slice)

    sl_npz = np.load(args.slice)
    cloud = sl_npz['events']
    idx   = sl_npz['index']

    if (len(args.bounds) != 2 or (args.bounds[0] > args.bounds[1] and args.bounds[1] != -1) 
        or (args.bounds[0] < 0) or (args.bounds[1] < -1)):
        print ("Invalid bounds: ", args.bounds)
        print ("Bounds have to specify two points in the index array, possible values are 0 -", len(idx) - 1)
        exit(0)

    sl = cloud[idx[args.bounds[0]]:idx[args.bounds[1]]]

    if (args.info):
        width = cloud[-1][0] - cloud[0][0]
        print ("Input cloud:")
        print ("\tWidth: ", width, "seconds and", len(cloud), "events.")
        print ("\tIndex size: ", len(idx), "points, step = ", width / float(len(idx) + 1), "seconds.")
        print ("")
        width = sl[-1][0] - sl[0][0]
        print ("Chosen slice:")
        print ("\tWidth: ", width, "seconds and", len(sl), "events.")
        print ("")


    slice_width = sl[-1][0] - sl[0][0]

    #for i in range(100000):
    cmb = dvs_img(sl, global_shape)
    cv2.imwrite(args.img_name, cmb)

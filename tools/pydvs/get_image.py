#!/usr/bin/python3

import argparse
import numpy as np
import cv2
import os, sys, signal, glob, time
import matplotlib.colors as colors

# The dvs-related stuff, implemented in C. 
sys.path.insert(0, './build/lib.linux-x86_64-3.5') #The pydvs.so should be in PYTHONPATH!
import pydvs

global_scale_t = 20 * 255
global_scale_pn = 100
global_scale_pp = 100
global_shape = (200, 346)
slice_width = 1



def dvs_img(cloud, shape):
    fcloud = cloud.astype(np.float32) # Important!

    cmb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    pydvs.dvs_img(fcloud, cmb)
    cmb[:,:,0] *= global_scale_pp
    cmb[:,:,1] *= 255.0 / slice_width
    cmb[:,:,2] *= global_scale_pn

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

    cmb = dvs_img(sl, global_shape)
    cv2.imwrite(args.img_name, cmb)

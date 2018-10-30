#!/usr/bin/python3

import argparse
import numpy as np
import cv2
import os, sys, signal, glob, time
import matplotlib.colors as colors

# The dvs-related stuff, implemented in C. 
sys.path.insert(0, './build/lib.linux-x86_64-3.5') #The pydvs.so should be in PYTHONPATH!
import pydvs

global_scale_pn = 50
global_scale_pp = 50
global_shape = (200, 346)
slice_width = 1



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

    #cmb = undistort_img(cmb, K, D)
    return cmb
    return cmb.astype(np.uint8)


# 'Correct' stacking function
def stack(timestamped_images):
    cmb = np.zeros(timestamped_images[0][0].shape, dtype=np.float32)
    ts0 = timestamped_images[0][1]

    for i, (img, ts) in enumerate(timestamped_images):
        offset = (ts - ts0) * (255.0 / slice_width) 
        cmb[:,:,0] += img[:,:,0]
        cmb[:,:,2] += img[:,:,2]
        cmb[:,:,1] += (img[:,:,1] + offset) * (img[:,:,0] + img[:,:,2])

    cnt_img = cmb[:,:,0] + cmb[:,:,2]
    cmb[:,:,1] = np.divide(cmb[:,:,1], cnt_img, out=np.zeros_like(cmb[:,:,1]), where=cnt_img > 0.5)
    return cmb
    return cmb.astype(np.uint8)


# Approximated stacking function, but the difference is not visible
def stack_approx(images, width=0.01):
    cmb = np.zeros(images[0].shape, dtype=np.float32)

    for i, img in enumerate(images):
        offset = i * width * (255.0 / slice_width)
        cmb[:,:,0] += img[:,:,0]
        cmb[:,:,2] += img[:,:,2]
        cmb[:,:,1] += (img[:,:,1] + offset) * (img[:,:,0] + img[:,:,2])

    cnt_img = cmb[:,:,0] + cmb[:,:,2]
    cmb[:,:,1] = np.divide(cmb[:,:,1], cnt_img, out=np.zeros_like(cmb[:,:,1]), where=cnt_img > 0.5)
    return cmb
    return cmb.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice',
                        type=str,
                        required=True)
    parser.add_argument('--folder_name',
                        type=str,
                        required=True)
    parser.add_argument('--bounds',
                        nargs='+',
                        type=int,
                        default=[0, -1],
                        required=False)

    args = parser.parse_args()

    print ("Opening", args.slice)

    sl_npz = np.load(args.slice)
    cloud = sl_npz['events']
    idx   = sl_npz['index']
    K     = sl_npz['K']
    D     = sl_npz['D']

    if (len(args.bounds) != 2 or (args.bounds[0] > args.bounds[1] and args.bounds[1] != -1) 
        or (args.bounds[0] < 0) or (args.bounds[1] < -1)):
        print ("Invalid bounds: ", args.bounds)
        print ("Bounds have to specify two points in the index array, possible values are 0 -", len(idx) - 1)
        exit(0)

    sl = cloud[idx[args.bounds[0]]:idx[args.bounds[1]]]

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

    full = dvs_img(sl, global_shape, K, D) 
    cv2.imwrite(args.folder_name + '/full.png', full)

    step = int((args.bounds[1] - args.bounds[0]) / 5)
    print ("Step = ", step)

    slices = []
    timestamped_slices = []
    for i, b in enumerate(range(args.bounds[0], args.bounds[1], step)):
        sl_ = cloud[idx[b]:idx[b + step]]
        cmb = dvs_img(sl_, global_shape, K, D)
        
        slices.append(cmb)
        timestamped_slices.append([cmb, sl_[0][0]]) # DVS image + the timestamp where it started (global or local)
        #cv2.imwrite(args.folder_name + '/full_' + str(i) + '.png', cmb)

    print ("Generated", len(timestamped_slices), "images")

    # 'Correct' stacking
    stacked = stack(timestamped_slices)
    cv2.imwrite(args.folder_name + '/stacked.png', stacked)
    cv2.imwrite(args.folder_name + '/error.png', np.abs(stacked - full))

    # 'Approximated' and a bit simpler stacking
    stacked_approx = stack_approx(slices, 0.01)
    cv2.imwrite(args.folder_name + '/stacked_approx.png', stacked_approx)
    cv2.imwrite(args.folder_name + '/error_approx.png', np.abs(stacked_approx - full))

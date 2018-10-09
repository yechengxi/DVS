import argparse
import numpy as np
import cv2
import os, sys, signal, glob
from multiprocessing import Pool


global_scale_t = 20 * 255
global_scale_pn = 100
global_scale_pp = 100
global_shape = (200, 346)


def dvs_img(cloud, shape):
    t0 = min(cloud[0][0], cloud[-1][0])
    timg = np.zeros(shape, dtype=np.float)
    cimg = np.zeros(shape, dtype=np.float)
    pimg = np.zeros(shape, dtype=np.float)

    for e in cloud:
        x = int(e[1])
        y = int(e[2])
        p = 0
        if (e[3] > 0.5):
            p = 1

        if (y >= shape[0] or x >= shape[1]):
            continue

        timg[y, x] += (e[0] - t0)
        if (p > 0):
            cimg[y, x] += 1
        else:
            pimg[y, x] += 1

    timg = np.divide(timg, cimg+pimg, out=np.zeros_like(timg), where=(cimg+pimg)!=0)
    return timg, cimg, pimg




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice',
                        type=str,
                        required=True)
    parser.add_argument('--info',
                        action='store_true',
                        required=False)

    args = parser.parse_args()

    print("Opening", args.slice)

    sl_npz = np.load(args.slice)
    cloud = sl_npz['events']
    idx = sl_npz['index']

    slices=[]

    for i in range(len(idx)):
        slices.append(None)

    list(map(get_slice, range(len(idx)-1)))

    for i in range(len(idx)-1):
        if i%100==0:
            print(i)
        sl = cloud[idx[i]:idx[i+1]]
        timg, cimg, pimg = dvs_img(sl, global_shape)
        slice=np.dstack((cimg,timg,pimg))
        slice[i]=slice

        #print(slice.shape)



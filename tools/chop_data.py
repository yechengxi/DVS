#!/usr/bin/python

import argparse
import numpy as np
import cv2
import os, sys, signal, glob, time


def get_slice(cloud, idx, ts, width, mode=0, idx_step=0.01):
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

    return sl, idx_, t0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slice_in',
                        type=str,
                        required=True)
    parser.add_argument('--slice_out',
                        type=str,
                        required=True)
    parser.add_argument('--t1',
                        type=float,
                        required=True)
    parser.add_argument('--t2',
                        type=float,
                        required=True)

    args = parser.parse_args()

    print "Opening", args.slice_in

    sl_npz = np.load(args.slice_in)
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']
    K              = sl_npz['K']
    D              = sl_npz['D']
    depth_gt       = sl_npz['depth']
    flow_gt        = sl_npz['flow']
    gt_ts          = sl_npz['gt_ts']

    first_ts = cloud[0][0]
    last_ts = cloud[-1][0]

    print "The recording range:", first_ts, "-", last_ts
    print "The gt range:", gt_ts[0], "-", gt_ts[-1]
    print "gt frame count:", len(gt_ts)
    print "Discretization resolution:", discretization
    if (args.t1 < first_ts or args.t2 > last_ts):
        print "The time boundaries have to be within range"
        exit(0)

    width = args.t2 - args.t1
    sl, idx_, t0 = get_slice(cloud, idx, args.t1, width, 0, discretization)
    t1 = t0 + sl[-1][0] - sl[0][0] # The t1 - t2 ragne can be shifted due to discretization

    idx_lo = 0
    for i, t in enumerate(gt_ts):
        if t > t0:
           idx_lo = i
           break
    idx_hi = 0
    for i, t in enumerate(gt_ts):
        if t > t1:
           idx_hi = i
           break

    depth_gt_ = depth_gt[idx_lo : idx_hi]
    flow_gt_  = flow_gt[idx_lo : idx_hi]
    gt_ts_    = np.copy(gt_ts[idx_lo : idx_hi])
    gt_ts_ -= t0

    print "Saving", depth_gt_.shape[0], "gt slices"

    np.savez_compressed(args.slice_out, events=sl, index=idx_, 
        discretization=discretization, K=K, D=D, depth=depth_gt_, flow=flow_gt_, gt_ts=gt_ts_)

import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import os
import glob
import fnmatch


import sys
sys.path.insert(0, './build/lib.linux-x86_64-3.6') #The libdvs.so should be in PYTHONPATH!
import pydvs


global_scale_pn = 20
global_scale_pp = 20
global_shape = (200, 346)
slice_width = 1



import cv2

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

    cmb = np.uint8(cmb)
    cmb = undistort_img(cmb, K, D)
    return cmb


class NewCloudSequenceFolder(data.Dataset):

    def __init__(self, root, train=True, sequence_length=5,slices=25, transform=None,gt=True):
        self.root = Path(root)
        self.gt = gt
        self.sequence_length=sequence_length
        self.slices=slices
        self.scenes = []
        self.transform = transform

        if train:
            self.train = True
        else:
            self.train = False
        for root, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, '*.npz'):
                self.scenes.append(os.path.join(root, filename))
        self.scenes = sorted(self.scenes)
        self.scenes=[np.load(scene) for scene in self.scenes]


        self.cloud=[]
        self.idx=[]
        self.discretization=[]
        self.K=[]
        self.D=[]
        self.depth=[]
        self.gt_ts=[]
        self.flow=[]

        self.train_idx = []
        self.test_idx = []

        for s, scene in enumerate(self.scenes):
            self.cloud.append(scene['events'].astype(np.float32))
            self.idx.append(scene['index'])
            self.discretization.append(scene['discretization'])
            self.K.append(scene['K'].astype(np.float32))
            self.D.append(scene['D'].astype(np.float32))
            #self.depth.append(scene['depth'])
            self.gt_ts.append(scene['gt_ts'])
            #self.flow.append(scene['flow'])

            split=int(len(self.gt_ts[-1]) * .8)

            tmp=self.scenes[s]['gt_ts'][:split]
            self.train_idx += list(zip([s for i in range(len(tmp))],tmp))

            tmp=self.scenes[s]['gt_ts'][split:]
            self.test_idx += list(zip([s for i in range(len(tmp))],tmp))


    def __getitem__(self, index):
        if self.train:
            scene_id,gt_ts=self.train_idx[index]
        else:
            scene_id,gt_ts=self.test_idx[index]

        cloud = self.cloud[scene_id]
        cloud_idx =self.idx[scene_id]
        sl, idx = get_slice(cloud, cloud_idx, gt_ts, 0.25, 0, self.discretization[scene_id])

        cmb = dvs_img(sl, global_shape, self.K[scene_id], self.D[scene_id])
        tgt_img=cmb


        n_slice = len(idx)
        idx = list(idx) + [len(sl)]

        T = int(n_slice / self.sequence_length)

        seqs = []
        for i in range(self.sequence_length):
            mini_slice = sl[idx[i * T]:idx[(i + 1) * T]]
            cmb = dvs_img(mini_slice, global_shape, self.K[scene_id], self.D[scene_id])
            seqs.append(cmb)

        slices = []
        if self.slices>0:
            T = int(n_slice / self.slices)

            # store slices

            for i in range(self.slices):
                mini_slice = sl[idx[i*T]:idx[(i + 1)*T]]
                cmb = dvs_img(mini_slice, global_shape, self.K[scene_id], self.D[scene_id])
                slices.append(cmb)


        if self.transform is not None:
            imgs, intrinsics = self.transform(seqs +slices, np.copy(self.K[scene_id]))
            seqs = imgs[:self.sequence_length]
            slices=imgs[self.sequence_length:]

        else:
            intrinsics = np.copy(self.K[scene_id])

        return seqs, slices, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        if self.train:
            return len(self.train_idx)
        else:
            return len(self.test_idx)






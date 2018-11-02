import torch.utils.data as data
import numpy as np
from path import Path
import random
import os
import glob
import fnmatch


import sys
sys.path.insert(0, './build/lib.linux-x86_64-3.6') #The libdvs.so should be in PYTHONPATH!
import pydvs

from multiprocessing import Pool

global_shape = (200, 346)


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

    if (sl.shape[0] > 0):
        t0 = sl[0][0]
        sl[:,0] -= t0

    return sl, idx_


def nz_avg(cmb_img, ch_id):
    pcnt = np.copy(cmb_img[:,:,0])
    ncnt = np.copy(cmb_img[:,:,2])
    target = np.copy(cmb_img[:,:,ch_id]).astype(np.float32)
    cnt = pcnt + ncnt
    target[cnt <= 0.5] = np.nan
    target[target <= 0.00000001] = np.nan
    return (np.nanmin(target), np.nanmean(target), np.nanmax(target))


def undistort_img(img, K, D):
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted


def dvs_img(cloud, shape, K, D):
    cmb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    if (cloud.shape[0] == 0):
        return cmb

    fcloud = cloud.astype(np.float32)  # Important!
    pydvs.dvs_img(fcloud, cmb)

    cmb = undistort_img(cmb, K, D)

    cnt_img = cmb[:, :, 0] + cmb[:, :, 2] + 1e-8
    timg = cmb[:, :, 1]

    timg[cnt_img < 0.99] = 0
    timg /= cnt_img

    cmb[:, :, 0] *= 50
    cmb[:, :, 1] *= 255.0 / 0.05
    cmb[:, :, 2] *= 50

    return cmb
    return cmb.astype(np.uint8)


def load_scene_p(queue,scene_path,with_gt=False):
    scene_npz = np.load(scene_path)
    scene = {}
    scene['events']= scene_npz['events']#.astype(np.float32)
    scene['index']= scene_npz['index']
    scene['discretization']= scene_npz['discretization']
    scene['K']= scene_npz['K'].astype(np.float32)
    scene['D']= scene_npz['D'].astype(np.float32)
    if with_gt:
        scene['depth'] = scene_npz['depth'].astype(np.float32)
    scene['gt_ts']= scene_npz['gt_ts']
    #scene['flow']= scene_npz['flow']
    queue.put(scene)


def load_scene_s(scene_path, with_gt=False):
    scene_npz = np.load(scene_path)
    scene = {}
    scene['events'] = scene_npz['events']  # .astype(np.float32)
    scene['index'] = scene_npz['index']
    scene['discretization'] = scene_npz['discretization']
    scene['K'] = scene_npz['K'].astype(np.float32)
    scene['D'] = scene_npz['D'].astype(np.float32)
    if with_gt:
        scene['depth'] = scene_npz['depth'].astype(np.float32)
    scene['gt_ts'] = scene_npz['gt_ts']
    # scene['flow']= scene_npz['flow']
    return scene

class NewCloudSequenceFolder(data.Dataset):

    def __init__(self, root, train=True, sequence_length=5,slices=25, train_transform=None,test_transform=None,gt=False):
        self.root = Path(root)
        self.gt = gt
        self.sequence_length=sequence_length
        self.slices=slices
        self.scenes = []
        self.train_transform = train_transform
        self.test_transform = test_transform

        scenes=[]
        if train:
            self.train = True
        else:
            self.train = False
        for root, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, '*.npz'):
                scenes.append(os.path.join(root, filename))
        scenes = sorted(scenes)
        #self.scenes=self.scenes[:1]
        self.n_scenes = len(scenes)

        self.scenes=[None] * self.n_scenes

        self.train_idx = []
        self.test_idx = []


        import time

        if True:
            t_s = time.time()
            for id in range(self.n_scenes):
                self.scenes[id]=load_scene_s(scenes[id],self.gt)
            t_e  = time.time()
            print('loading time:', t_e-t_s)

        else:
            from multiprocessing import Process,Queue
            t_s=time.time()
            queue = Queue()
            procs = []
            for id in range(self.n_scenes):
                print('scene: ',id)
                proc = Process(target=load_scene_p, args=(queue,scenes[id],self.gt))
                procs.append(proc)
                proc.start()
            self.scenes=[queue.get() for p in procs]
            for p in procs:
                p.join()
                print('join')
            t_e  = time.time()

            print('loading time:', t_e-t_s)

        for id in range(self.n_scenes):
            split = int(len(self.scenes[id]['gt_ts']) * .8)
            tmp=[i for i in range(len(self.scenes[id]['gt_ts']))]
            self.scenes[id]['n_train']=len(tmp[:split])
            self.scenes[id]['n_test'] = len(tmp[split:])

            self.train_idx += list(zip([id for i in range(len(tmp[:split]))],tmp[:split]))
            self.test_idx += list(zip([id for i in range(len(tmp[split:]))],tmp[split:]))

            if self.gt:#only save the portion we use
                self.scenes[id]['depth'] = self.scenes[id]['depth'][split:]
                self.scenes[id]['depth'][np.isnan(self.scenes[id]['depth'])]=0. #set nan to 0



    def __getitem__(self, index):
        if self.train:
            scene_id, index = self.train_idx[index]
        else:
            scene_id, index = self.test_idx[index]
        gt_ts=self.scenes[scene_id]['gt_ts'][index]

        cloud = self.scenes[scene_id]['events']
        cloud_idx =self.scenes[scene_id]['index']
        sl, idx = get_slice(cloud, cloud_idx, gt_ts, 0.25, 2, self.scenes[scene_id]['discretization'])

        n_slice = len(idx)
        idx = list(idx) + [len(sl)]

        T = int(n_slice / self.sequence_length)
        seqs = []
        for i in range(self.sequence_length):
            mini_slice = sl[idx[i * T]:idx[(i + 1) * T]]
            cmb = dvs_img(mini_slice, global_shape, self.scenes[scene_id]['K'], self.scenes[scene_id]['D'])
            seqs.append(cmb)

        slices = []
        if self.slices>0:
            T = int(n_slice / self.slices)
            # store slices
            for i in range(self.slices):
                mini_slice = sl[idx[i*T]:idx[(i + 1)*T]]
                cmb = dvs_img(mini_slice, global_shape,  self.scenes[scene_id]['K'], self.scenes[scene_id]['D'])
                slices.append(cmb)

        if  self.train:
            self.transform=self.train_transform
        else:
            self.transform=self.test_transform

        if self.transform is not None:
            imgs, intrinsics = self.transform(seqs +slices, np.copy(self.scenes[scene_id]['K']))
            seqs = imgs[:self.sequence_length]
            slices=imgs[self.sequence_length:]
        else:
            intrinsics = np.copy(self.scenes[scene_id]['K'])


        if self.gt and (not self.train):
            index-=self.scenes[scene_id]['n_train']
            depth=self.scenes[scene_id]['depth'][index]
            return seqs, slices, intrinsics, np.linalg.inv(intrinsics),depth
        else:
            return seqs, slices, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        if self.train:
            return len(self.train_idx)
        else:
            return len(self.test_idx)


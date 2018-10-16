import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import os
import glob
import fnmatch

def load_as_float(path):
    return imread(path).astype(np.float32)


from functools import partial
from multiprocessing import Pool
from scipy.misc import imresize



global_scale_pn = 20
global_scale_pp = 20
global_shape = (200, 346)
slice_width = 1


import sys
sys.path.insert(0, './build/lib.linux-x86_64-3.6') #The libdvs.so should be in PYTHONPATH!
import libdvs

import cv2
def undistort_img(img, K, dist):
    D = dist
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 0.87 * Knew[(0,1), (0,1)]
    img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted

class CloudSequenceFolder(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=25, transform=None, LoadToRam=False, scale=1.,
                 gt=True):
        np.random.seed(seed)
        random.seed(seed)
        self.LoadToRam = False
        self.scale = scale
        self.root = Path(root)
        self.gt = gt
        self.sequence_length=sequence_length
        cam_lst = []
        for root, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, 'cam.txt'):
                cam_lst.append(os.path.join(root, filename))
            cam_lst = sorted(cam_lst)
        # print(cam_lst)

        self.scenes = cam_lst
        self.scenes = [scene.replace('cam.txt', '') for scene in self.scenes]

        if train:
            self.train = True
            self.scenes = [scene for scene in self.scenes if 'train' in scene]
        else:
            self.train = False
            self.scenes = [scene for scene in self.scenes if 'train' in scene]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            f = open(os.path.join(scene, 'cam.txt'), 'r')

            import re
            non_decimal = re.compile(r'[^\d. ]+')
            l = [[float(num) for num in non_decimal.sub('', line).split()] for line in f]
            intrinsics = np.asarray(l[:3]).astype(np.float32)

            distort=None
            if os.path.exists(os.path.join(scene, 'distort.txt')):
                f = open(os.path.join(scene, 'distort.txt'), 'r')
                l = [[float(num) for num in non_decimal.sub('', line).split()] for line in f]
                distort = np.asarray(l[:1]).astype(np.float32)

            imgs = sorted(glob.glob(os.path.join(scene, '*.npz')))

            split = int(len(imgs) * .8)
            if self.train:
                imgs = imgs[:split]
            else:
                imgs = imgs[split:-1]

            self.depths = [img.replace('frame', 'depth').replace('.jpg', '.npy') for img in imgs]


            if self.scale != 1.:
                intrinsics[0] *= self.scale
                intrinsics[1] *= self.scale

            for i in range( len(imgs) ):
                sample = {'intrinsics': intrinsics,'distort':distort, 'tgt': imgs[i], 'ref_imgs': [], 'depth': self.depths[i]}
                if self.train or os.path.exists(self.depths[i]) or (not self.gt):
                    sequence_set.append(sample)

        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        sl_npz = np.load(sample['tgt'])
        cloud = sl_npz['events']
        fcloud = cloud.astype(np.float32)  # Important!

        idx = sl_npz['index']
        n_slice=len(idx)+1
        idx=[0]+list(idx)+[len(fcloud)]

        T=int(n_slice/self.sequence_length)

        cmb = np.zeros((200, 346, 3), dtype=np.float32)
        libdvs.dvs_img(fcloud, cmb)

        cmb[np.isnan(cmb)]=0.
        cmb=np.clip(cmb,0.,255.)
        cmb[:, :, 0] *= global_scale_pp
        cmb[:, :, 1] *= 255.0 / slice_width
        cmb[:, :, 2] *= global_scale_pn
        cmb=np.uint8(cmb)
        cmb=undistort_img(cmb,sample['intrinsics'],sample['distort'])
        #print(cmb[..., 0].mean(), cmb[..., 1].mean(), cmb[..., 2].mean())
        tgt_img=cmb


        # store slices
        slices=[]
        for i in range(self.sequence_length):
            tmp = fcloud[idx[i*T]:idx[(i + 1)*T]]
            cmb = np.zeros((200, 346, 3), dtype=np.float32)
            libdvs.dvs_img(tmp, cmb)

            cmb[np.isnan(cmb)] = 0.
            cmb = np.clip(cmb, 0., 255.)
            cmb[:, :, 0] *= global_scale_pp
            cmb[:, :, 1] *= 255.0 / slice_width
            cmb[:, :, 2] *= global_scale_pn

            cmb = np.uint8(cmb)
            cmb = undistort_img(cmb, sample['intrinsics'], sample['distort'])

            slices.append(cmb)


        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] +slices, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            slices=imgs[1:]

        else:
            intrinsics = np.copy(sample['intrinsics'])
        if self.train or (not self.gt):
            return tgt_img, slices, intrinsics, np.linalg.inv(intrinsics)
        else:
            depth = np.load(sample['depth']).astype(np.float32)
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), depth

    def __len__(self):
        return len(self.samples)



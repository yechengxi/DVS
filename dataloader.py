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



global_scale_t = 20 * 255
global_scale_pn = 100
global_scale_pp = 100
global_shape = (200, 346)
slice_width = 1


def dvs_img(cloud, shape):
    fcloud = cloud.astype(np.float32)  # Important!

    t0 = min(cloud[0][0], cloud[-1][0])
    timg = np.zeros(shape, dtype=np.float32)
    cimg = np.zeros(shape, dtype=np.float32)
    nimg = np.zeros(shape, dtype=np.float32)
    pimg = np.zeros(shape, dtype=np.float32)

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

    timg = np.divide(timg, cimg, out=np.zeros_like(timg), where=cimg != 0)

    cmb = np.dstack((nimg * global_scale_pp, timg * 255 / slice_width, pimg * global_scale_pn))

    return cmb

import sys
sys.path.insert(0, './build/lib.linux-x86_64-3.6') #The libdvs.so should be in PYTHONPATH!
import libdvs


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
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 'depth': self.depths[i]}
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

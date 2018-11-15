import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import os
import glob
import fnmatch
import cv2
def load_as_float(path):
    return imread(path).astype(np.float32)


from functools import partial
from multiprocessing import Pool
from scipy.misc import imresize

num_workers=12

def load_image(file_path,scale):
    img = load_as_float(file_path)
    if scale!=1.:
        img=imresize(img,scale)
    return img



class LabSequenceFolder2(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, LoadToRam=False, scale=1.,
                 gt=True):
        np.random.seed(seed)
        random.seed(seed)
        self.LoadToRam = LoadToRam
        self.scale = scale
        self.root = Path(root)
        self.gt = gt
        cam_lst = []
        for root, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, 'calib.txt'):
                cam_lst.append(os.path.join(root, filename))
            cam_lst = sorted(cam_lst)
        # print(cam_lst)

        self.scenes = cam_lst
        self.scenes = [scene.replace('calib.txt', '') for scene in self.scenes]

        if train:
            self.train = True
        else:
            self.train = False

        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            f = open(os.path.join(scene, 'calib.txt'), 'r')
            import re
            non_decimal = re.compile(r'[^\d. ]+')
            l = [[float(num) for num in non_decimal.sub('', line).split()] for line in f]
            intrinsics = np.asarray(l[:3]).astype(np.float32)
            distortion = np.asarray(l[4]).astype(np.float32)

            imgs = sorted(glob.glob(os.path.join(scene,'slices', 'frame*.png')))
            depths = sorted(glob.glob(os.path.join(scene,'slices', 'depth*.png')))
            masks = sorted(glob.glob(os.path.join(scene, 'slices', 'mask*.png')))

            split = int(len(imgs) * .8)
            if self.train:
                imgs = imgs[:split]
                depths = depths[:split]
                masks = masks[:split]

            else:
                imgs = imgs[split:]
                depths = depths[split:]
                masks = masks[split:]

            self.depths = depths
            self.masks = masks

            if len(imgs) < sequence_length:
                continue

            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 'depth': self.depths[i],'D':distortion,'mask':self.masks[i]}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                if self.train or os.path.exists(self.depths[i]) or (not self.gt):
                    sequence_set.append(sample)

        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]

        tgt_img = load_as_float(sample['tgt'])

        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        depth = cv2.imread(sample['depth'], -1)
        depth[depth<100]=10000
        depth = depth.astype(np.float32) / 6000*255
        depth=np.expand_dims(depth,axis=2)

        obj_mask = cv2.imread(sample['mask'], -1)
        obj_mask = obj_mask.astype(np.float32)/1000*255
        obj_mask=np.expand_dims(obj_mask,axis=2)

        depth=np.concatenate((depth,obj_mask),axis=2)

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs+[depth], np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:-1]
            depth=imgs[-1]
        else:
            intrinsics = np.copy(sample['intrinsics'])


        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), depth

    def __len__(self):
        return len(self.samples)


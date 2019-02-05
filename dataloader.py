import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import os
import glob
import fnmatch
import cv2

import sys
sys.path.insert(0, './pydvs/build/lib.linux-x86_64-3.6') #The libdvs.so should be in PYTHONPATH!
import pydvs


def load_as_float(path):
    return imread(path).astype(np.float32)

from scipy.misc import imresize

global_shape = (260, 346)



def get_slice(cloud, idx, ts, width, mode=1, idx_step=0.01):
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

def get_mask(shape, K, D):
    mask = np.ones((shape[0], shape[1]), dtype=np.float32)
    mask = undistort_img(mask, K, D)
    return mask



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
        scene['mask'] = scene_npz['mask'].astype(np.float32)

    scene['gt_ts'] = scene_npz['gt_ts']
    # scene['flow']= scene_npz['flow']
    return scene

class ImageSequenceFolder(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, LoadToRam=False, scale=1.,
                 gt=True):
        np.random.seed(seed)
        random.seed(seed)
        self.sequence_length=sequence_length
        self.LoadToRam = LoadToRam
        self.scale = scale
        self.root = Path(root)
        self.gt = gt
        cam_lst = []
        for root, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, 'calib.txt'):
                cam_lst.append(os.path.join(root, filename))

        self.scenes = cam_lst
        self.scenes = [scene.replace('calib.txt', '') for scene in self.scenes]

        self.n_scenes=len(self.scenes)


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


        for id, scene in enumerate(self.scenes):
            f = open(os.path.join(scene, 'calib.txt'), 'r')

            """
            import re
            non_decimal = re.compile(r'[^\d. ]+')
            l = [[float(num) for num in non_decimal.sub('', line).split()] for line in f]
            intrinsics = np.asarray(l[:3]).astype(np.float32)
            distortion = np.asarray(l[4]).astype(np.float32)
            """
            line = f.readline()
            l = [float(num) for num in line.split()]
            intrinsics = np.asarray([[l[1],0,l[3]],[0,l[0],l[2]],[0,0,1]]).astype(np.float32)
            distortion = np.asarray(l[4:]).astype(np.float32)

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

        #random.shuffle(sequence_set)

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
            ref_imgs = imgs[1:self.sequence_length]
            depth=imgs[-1]
        else:
            intrinsics = np.copy(sample['intrinsics'])

        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), depth

    def __len__(self):
        return len(self.samples)




class CloudSequenceFolder(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, slices=0,duration=0.05, transform=None, LoadToRam=False, scale=1.,
                 gt=True):
        np.random.seed(seed)
        random.seed(seed)
        self.sequence_length=sequence_length
        self.slices=slices
        self.duration=duration
        self.LoadToRam = LoadToRam
        self.scale = scale
        self.root = Path(root)
        self.gt = gt
        cam_lst = []
        for root, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, 'calib.txt'):
                cam_lst.append(os.path.join(root, filename))

        self.scenes = cam_lst
        self.scenes = [scene.replace('calib.txt', '') for scene in self.scenes]

        self.n_scenes=len(self.scenes)


        if train:
            self.train = True

        else:
            self.train = False

        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):

        if self.slices>0 and self.train:
            import time
            t_s = time.time()
            self.raw_data = [os.path.join(scene, 'recording.npz') for scene in self.scenes]
            for id in range(self.n_scenes):
                self.raw_data[id] = load_scene_s(self.raw_data[id],with_gt=True)


        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)


        for id, scene in enumerate(self.scenes):
            f = open(os.path.join(scene, 'calib.txt'), 'r')
            import re
            non_decimal = re.compile(r'[^\d. ]+')

            """
            l = [[float(num) for num in non_decimal.sub('', line).split()] for line in f]
            intrinsics = np.asarray(l[:3]).astype(np.float32)
            distortion = np.asarray(l[4]).astype(np.float32)

            """
            line = f.readline()
            l = [float(num) for num in line.split()]
            intrinsics = np.asarray([[l[1],0,l[3]],[0,l[0],l[2]],[0,0,1]]).astype(np.float32)
            distortion = np.asarray(l[4:]).astype(np.float32)

            imgs = sorted(glob.glob(os.path.join(scene,'slices', 'frame*.png')))
            depths = sorted(glob.glob(os.path.join(scene,'slices', 'depth*.png')))
            masks = sorted(glob.glob(os.path.join(scene, 'slices', 'mask*.png')))


            split = int(len(imgs) * .8)


            if self.slices>0:
                print('raw data:', id, len(self.raw_data[id]['gt_ts']), len(imgs))
                tmp = [i for i in range(len(self.raw_data[id]['gt_ts']))]
                self.raw_data[id]['n_train'] = len(tmp[:split])
                self.raw_data[id]['n_test'] = len(tmp[split:])
                cloud_idx = list(zip([id for i in range(len(tmp))], tmp))

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
                if self.slices>0:
                    sample['cloud_idx']=cloud_idx[i]
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
                if self.train or os.path.exists(self.depths[i]) or (not self.gt):
                    sequence_set.append(sample)

        #random.shuffle(sequence_set)

        self.samples = sequence_set

    def dvs_img(self,cloud, shape, K, D):
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
        #print(nz_avg(cmb, 0))

        cmb[:, :, 0] *= 50/self.duration*0.05
        cmb[:, :, 1] *= 255.0 / self.duration
        # cmb[:, :, 1] *= 255.0 / 0.1
        cmb[:, :, 2] *= 50/self.duration*0.05
        return cmb
        return cmb.astype(np.uint8)
    def __getitem__(self, index):
        sample = self.samples[index]

        slices = []


        scene_id, index = sample['cloud_idx']
        gt_ts=self.raw_data[scene_id]['gt_ts'][index]

        cloud = self.raw_data[scene_id]['events']
        cloud_idx =self.raw_data[scene_id]['index']
        sl, idx = get_slice(cloud, cloud_idx, gt_ts, self.duration*self.sequence_length, 0, self.raw_data[scene_id]['discretization'])

        n_slice = len(idx)
        idx = list(idx) + [len(sl)]

        T = int(n_slice / self.sequence_length)
        seqs = []
        for i in range(self.sequence_length):
            mini_slice = sl[idx[i * T]:idx[(i + 1) * T]]
            cmb = self.dvs_img(mini_slice, global_shape, self.raw_data[scene_id]['K'], self.raw_data[scene_id]['D'])
            seqs.append(cmb)

        T = int(n_slice / self.slices)
        # store slices
        for i in range(self.slices):
            mini_slice = sl[idx[i*T]:idx[(i + 1)*T]]
            cmb = self.dvs_img(mini_slice, global_shape,  self.raw_data[scene_id]['K'], self.raw_data[scene_id]['D'])
            slices.append(cmb)

        tgt_img = seqs.pop((len(seqs)-1)//2)
        ref_imgs = seqs
        depth = self.raw_data[scene_id]['depth'][index]
        obj_mask = self.raw_data[scene_id]['mask'][index]
        depth[np.isnan(depth)] = 10000

        depth[depth<100]=10000
        depth = depth.astype(np.float32) / 6000*255
        depth=np.expand_dims(depth,axis=2)

        obj_mask = obj_mask.astype(np.float32)/1000*255
        obj_mask=np.expand_dims(obj_mask,axis=2)

        depth=np.concatenate((depth,obj_mask),axis=2)


        if self.transform is not None:

            imgs, intrinsics = self.transform([tgt_img] + ref_imgs+slices+[depth], np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:self.sequence_length]
            slices=imgs[self.sequence_length:-1]
            depth=imgs[-1]
        else:
            intrinsics = np.copy(sample['intrinsics'])

        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), depth,slices

    def __len__(self):
        return len(self.samples)

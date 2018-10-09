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

num_workers=12

def load_image(file_path,scale):
    img = load_as_float(file_path)
    if scale!=1.:
        img=imresize(img,scale)
    return img


##########################
class SequenceFolder(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, LoadToRam=False,scale=1.,gt=True):
        np.random.seed(seed)
        random.seed(seed)
        self.LoadToRam=LoadToRam
        self.scale=scale
        self.root = Path(root)
        self.gt=gt
        cam_lst = []
        for root, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, '*_cam.txt'):
                cam_lst.append(os.path.join(root, filename))
            cam_lst = sorted(cam_lst)
        #print(cam_lst)

        self.scenes = cam_lst
        self.scenes=[scene.replace('_cam.txt','') for scene in self.scenes]

        split=int(len(self.scenes)*0.75)
        if train:
            self.train=True
            #self.scenes=self.scenes[:split]
        else:
            self.train=False
            #self.scenes = self.scenes[split:]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:

            f = open(scene+'_cam.txt', 'r')

            import re
            non_decimal = re.compile(r'[^\d. ]+')
            l = [[float(num) for num in non_decimal.sub('', line).split()] for line in f]
            intrinsics=np.asarray(l[:3]).astype(np.float32)
            cnt_imgs=sorted(glob.glob(scene + '_cnt_*.jpg'))
            split = int(len(cnt_imgs) * .8)
            if self.train:
                cnt_imgs=cnt_imgs[:split]
            else:
                cnt_imgs=cnt_imgs[split:]
            time_imgs = [img.replace('_cnt','_time') for img in cnt_imgs]
            self.depths=[img.replace('_cnt','_depth').replace('.jpg','.npy') for img in cnt_imgs]

            if len(cnt_imgs) < sequence_length:
                continue

            if self.scale!=1.:
                intrinsics[0] *= self.scale
                intrinsics[1] *= self.scale

            if self.LoadToRam:
                load = partial(load_image, scale=self.scale)
                p = Pool(num_workers)
                cnt_imgs= p.map(load, cnt_imgs)
                time_imgs = p.map(load, time_imgs)
                p.close()


            for i in range(demi_length, len(cnt_imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt_cnt': cnt_imgs[i],'tgt_time': time_imgs[i], 'ref_imgs_cnt': [],'ref_imgs_time': [],'depth':self.depths[i]}
                for j in shifts:
                    sample['ref_imgs_cnt'].append(cnt_imgs[i+j])
                    sample['ref_imgs_time'].append(time_imgs[i+j])
                if self.train or os.path.exists(self.depths[i]) or (not self.gt):
                    sequence_set.append(sample)

        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        if not self.LoadToRam:
            tgt_img_cnt = load_as_float(sample['tgt_cnt'])
            tgt_img_time = load_as_float(sample['tgt_time'])

            ref_imgs_cnt = [load_as_float(ref_img) for ref_img in sample['ref_imgs_cnt']]
            ref_imgs_time = [load_as_float(ref_img) for ref_img in sample['ref_imgs_time']]

            if self.scale != 1.:
                tgt_img_cnt = imresize(tgt_img_cnt, self.scale)
                tgt_img_time = imresize(tgt_img_time, self.scale)

                ref_imgs_cnt = [imresize(ref_img, self.scale) for ref_img in ref_imgs_cnt]
                ref_imgs_time = [imresize(ref_img, self.scale) for ref_img in ref_imgs_time]

        else:
            tgt_img_time=sample['tgt_time']
            ref_imgs_cnt=[ref_img for ref_img in sample['ref_imgs_cnt']]
            ref_imgs_time=[ref_img for ref_img in sample['ref_imgs_time']]


        tgt_img=np.stack([tgt_img_cnt,tgt_img_time],axis=2)
        ref_imgs=[np.stack([ref_img_cnt,ref_img_time],axis=2) for (ref_img_cnt,ref_img_time) in zip(ref_imgs_cnt,ref_imgs_time)]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        if self.train or (not self.gt):
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)
        else:
            depth = np.load(sample['depth']).astype(np.float32)
            depth=depth[depth.shape[0]//4:-depth.shape[0]//4,:]
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics),depth

    def __len__(self):
        return len(self.samples)




class ImSequenceFolder(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, LoadToRam=False,scale=1.,gt=True):
        np.random.seed(seed)
        random.seed(seed)
        self.LoadToRam=LoadToRam
        self.scale=scale
        self.root = Path(root)
        self.gt=gt
        cam_lst = []
        for root, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, '*_cam.txt'):
                cam_lst.append(os.path.join(root, filename))
            cam_lst = sorted(cam_lst)
        #print(cam_lst)

        self.scenes = cam_lst
        self.scenes=[scene.replace('_cam.txt','') for scene in self.scenes]

        split=int(len(self.scenes)*0.75)
        if train:
            self.train=True
            #self.scenes=self.scenes[:split]
        else:
            self.train=False
            #self.scenes = self.scenes[split:]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:

            f = open(scene+'_cam.txt', 'r')

            import re
            non_decimal = re.compile(r'[^\d. ]+')
            l = [[float(num) for num in non_decimal.sub('', line).split()] for line in f]
            intrinsics=np.asarray(l[:3]).astype(np.float32)
            imgs=sorted(glob.glob(scene + '_cmb_*.jpg'))
            split = int(len(imgs) * .8)
            if self.train:
                imgs=imgs[:split]
            else:
                imgs=imgs[split:]
            self.depths=[img.replace('_cmb','_depth').replace('.jpg','.npy') for img in imgs]

            if len(imgs) < sequence_length:
                continue

            if self.scale!=1.:
                intrinsics[0] *= self.scale
                intrinsics[1] *= self.scale

            if self.LoadToRam:
                load = partial(load_image, scale=self.scale)
                p = Pool(num_workers)
                imgs= p.map(load, imgs)
                p.close()


            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [],'depth':self.depths[i]}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                if self.train or os.path.exists(self.depths[i]) or (not self.gt):
                    sequence_set.append(sample)

        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        if not self.LoadToRam:
            tgt_img = load_as_float(sample['tgt'])

            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

            if self.scale != 1.:
                tgt_img = imresize(tgt_img, self.scale)
                ref_imgs = [imresize(ref_img, self.scale) for ref_img in ref_imgs]

        else:
            tgt_img=sample['tgt']
            ref_imgs=[ref_img for ref_img in sample['ref_imgs']]

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        if self.train or (not self.gt):
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)
        else:
            depth = np.load(sample['depth']).astype(np.float32)
            depth=depth[:-60,:]
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics),depth

    def __len__(self):
        return len(self.samples)




class StackedSequenceFolder(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, LoadToRam=False,scale=1.,gt=True):
        np.random.seed(seed)
        random.seed(seed)
        self.LoadToRam=LoadToRam
        self.scale=scale
        self.root = Path(root)
        self.gt=gt
        cam_lst = []
        for root, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, 'cam.txt'):
                cam_lst.append(os.path.join(root, filename))
            cam_lst = sorted(cam_lst)
        #print(cam_lst)

        self.scenes = cam_lst
        self.scenes=[scene.replace('cam.txt','') for scene in self.scenes]

        if train:
            self.train=True
            self.scenes = [scene for scene in self.scenes if 'train' in scene]
        else:
            self.train=False
            self.scenes = [scene for scene in self.scenes if 'eval' in scene]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            f = open(os.path.join(scene,'cam.txt'), 'r')
            import re
            non_decimal = re.compile(r'[^\d. ]+')
            l = [[float(num) for num in non_decimal.sub('', line).split()] for line in f]
            intrinsics=np.asarray(l[:3]).astype(np.float32)
            imgs=sorted(glob.glob(os.path.join(scene, '*.jpg')))
            

            split = int(len(imgs) * .8)
            if self.train:
                imgs=imgs[:split]
            else:
                imgs=imgs[split:]

            self.depths=[img.replace('frame','depth').replace('.jpg','.npy') for img in imgs]

            if len(imgs) < sequence_length:
                continue

            if self.scale!=1.:
                intrinsics[0] *= self.scale
                intrinsics[1] *= self.scale

            if self.LoadToRam:
                load = partial(load_image, scale=self.scale)
                p = Pool(num_workers)
                imgs= p.map(load, imgs)
                p.close()

            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [],'depth':self.depths[i]}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                if self.train or os.path.exists(self.depths[i]) or (not self.gt):
                    sequence_set.append(sample)

        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        if not self.LoadToRam:
            tgt_img = load_as_float(sample['tgt'])

            ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

            if self.scale != 1.:
                tgt_img = imresize(tgt_img, self.scale)
                ref_imgs = [imresize(ref_img, self.scale) for ref_img in ref_imgs]

        else:
            tgt_img=sample['tgt']
            ref_imgs=[ref_img for ref_img in sample['ref_imgs']]

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        if self.train or (not self.gt):
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)
        else:
            depth = np.load(sample['depth']).astype(np.float32)
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics),depth

    def __len__(self):
        return len(self.samples)


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

    def __init__(self, root, seed=None, train=True, sequence_length=5, transform=None, LoadToRam=False, scale=1.,
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
                imgs = imgs[split:]

            self.depths = [img.replace('frame', 'depth').replace('.jpg', '.npy') for img in imgs]

            if len(imgs) < sequence_length:
                continue

            if self.scale != 1.:
                intrinsics[0] *= self.scale
                intrinsics[1] *= self.scale


            for i in range(demi_length, len(imgs) - demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 'depth': self.depths[i]}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i + j])
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
        ref_imgs=[]
        for i in range(self.sequence_length):
            tmp=fcloud[idx[i*T]:idx[(i+1)*T]]

            #cmb=dvs_img(tmp, (200,346))
            cmb = np.zeros((200, 346, 3), dtype=np.float32)
            libdvs.dvs_img(tmp, cmb)

            cmb[np.isnan(cmb)]=0.
            cmb=np.clip(cmb,0.,255.)
            cmb[:, :, 0] *= global_scale_pp
            cmb[:, :, 1] *= 255.0 / slice_width
            cmb[:, :, 2] *= global_scale_pn

            ref_imgs.append(cmb)
        tgt_img=ref_imgs.pop(int((self.sequence_length-1)/2))

        # store slices
        slices=[]
        """
        for i in range(n_slice):
            tmp = fcloud[idx[i]:idx[i + 1]]
            cmb = np.zeros((200, 346, 3), dtype=np.float32)
            libdvs.dvs_img(tmp, cmb)

            cmb[np.isnan(cmb)] = 0.
            cmb = np.clip(cmb, 0., 255.)
            cmb[:, :, 0] *= global_scale_pp
            cmb[:, :, 1] *= 255.0 / slice_width
            cmb[:, :, 2] *= global_scale_pn
            slices.append(cmb)
        """

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs+slices, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:self.sequence_length]
            slices=imgs[self.sequence_length:]

        else:
            intrinsics = np.copy(sample['intrinsics'])
        if self.train or (not self.gt):
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics),slices
        else:
            depth = np.load(sample['depth']).astype(np.float32)
            return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), depth

    def __len__(self):
        return len(self.samples)

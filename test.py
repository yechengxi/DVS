import numpy as np
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt

img = imread('indoor_flying1_cnt_0000000368.jpg').astype(np.float32)


import glob
path="/home/cxy/Data/DVS/indoor_flying/"
import os
cam_lst=sorted(glob.glob(os.path.join(path,"indoor_*_cam.txt")))
print(cam_lst)

scenes=[scene.replace('_cam.txt','') for scene in cam_lst]

#print(np.genfromtxt(cam_lst[0]))

for i in range(len(cam_lst)):
    print(len(sorted(glob.glob(cam_lst[i].replace('_cam.txt','')+'_cnt_*.jpg'))))

f = open(cam_lst[0], 'r')

import re
non_decimal = re.compile(r'[^\d. ]+')
l = [[float(num) for num in non_decimal.sub('',line).split()] for line in f]
print(np.asarray(l[:3]))

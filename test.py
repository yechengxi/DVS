import numpy as np
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt


import cv2
image = imread('frame_0000000284.jpg')



import glob
path="/home/cxy/Data/DVS/MVSEC/"
import os

import fnmatch
import os

cam_lst = []
for root, dirnames, filenames in os.walk(path):
    for filename in fnmatch.filter(filenames, '*_cam.txt'):
        cam_lst.append(os.path.join(root, filename))
    cam_lst=sorted(cam_lst)
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

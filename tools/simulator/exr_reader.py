#!/usr/bin/python

from utils import *


exr_img = OpenEXR.InputFile('./ev_datasets/data/exr/0002.exr')

print exr_img.header()

z = extract_depth(exr_img)
img = extract_grayscale(exr_img)

print np.min(z), np.max(z)

cv2.imwrite('/home/alice/Desktop/depth.png', z * 10)
cv2.imwrite('/home/alice/Desktop/grayscale.png', img * 255)

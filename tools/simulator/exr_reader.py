#!/usr/bin/python

import argparse
from utils import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',
                        type=str,
                        required=True)

    args = parser.parse_args()

    exr_img = OpenEXR.InputFile(args.file)

    print exr_img.header()

    z = extract_depth(exr_img)
    img = extract_grayscale(exr_img)

    print np.min(z), np.max(z)

    cv2.imwrite('/home/alice/Desktop/depth.png', z * 10)
    cv2.imwrite('/home/alice/Desktop/grayscale.png', img * 255)

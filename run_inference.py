import torch

from scipy.misc import imread, imsave, imresize
import cv2
import numpy as np
from path import Path
import argparse

import models
from utils import tensor2array

from inverse_warp import *
from flowlib import *

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
#parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=260, type=int, help="Image height")
parser.add_argument("--img-width", default=346, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--multi", action='store_true', help="multi image depth")
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")


parser.add_argument('--arch', default='ecn', help='architecture')
parser.add_argument('--norm-type', default='gn', help='normalization type')

parser.add_argument('--n-channel', '--init-channel', default=32, type=int,
                    help='initial feature channels(32|64|128).')
parser.add_argument('--growth-rate', default=32, type=int, help='feature channel growth rate.')
parser.add_argument('--scale-factor', default=1. / 2.,
                    type=float, help='scaling factor of each layer(0.5|0.75|0.875)')
parser.add_argument('--final-map-size', default=1, type=int, help='final map size')



def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return


    disp_net = models.ECN_Disp(input_size=args.img_height,init_planes=args.n_channel,scale_factor=args.scale_factor,growth_rate=args.growth_rate,final_map_size=args.final_map_size,norm_type=args.norm_type).cuda()

    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if args.pretrained_posenet:

        pose_net = models.ECN_Pose(input_size=args.img_height, nb_ref_imgs=args.sequence_length - 1,
                                       init_planes=args.n_channel // 2, scale_factor=args.scale_factor,
                                       growth_rate=args.growth_rate // 2, final_map_size=args.final_map_size,
                                       norm_type=args.norm_type).cuda()

        weights = torch.load(args.pretrained_posenet)
        pose_net.load_state_dict(weights['state_dict'])
        pose_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()


    import os
    import glob
    import time

    scene=os.path.join(args.dataset_dir)
    intrinsics = np.genfromtxt(dataset_dir / 'calib.txt',max_rows=3).astype(np.float32).reshape((3, 3))

    intrinsics_inv = np.linalg.inv(intrinsics)

    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).cuda()
    intrinsics_inv = torch.from_numpy(intrinsics_inv).unsqueeze(0).cuda()
    imgs = sorted(glob.glob(os.path.join(scene ,'slices' ,'frame*.png')))
    masks = sorted(glob.glob(os.path.join(scene, 'slices', 'mask*.png')))

    print('{} files to test'.format(len(imgs)))

    #for file in tqdm(test_files):

    class File:
        basename=None
        ext=None

    demi_length = (args.sequence_length - 1) // 2
    shifts = list(range(-demi_length, demi_length + 1))
    shifts.pop(demi_length)

    #for i in range(demi_length,len(imgs)-demi_length):
    for i in range(round(len(imgs)*.9), len(imgs) - demi_length):

        file =File()
        file.namebase=os.path.basename(imgs[i]).replace('.png','')
        file.ext='.jpg'

        img = imread(imgs[i]).astype(np.float32)
        obj_mask = cv2.imread(masks[i], -1)
        obj_mask=np.round(obj_mask/1000)
        ref_imgs=[]
        for j in shifts:
            ref_imgs.append(imread(imgs[i + j]).astype(np.float32))

        h, w, _ = img.shape
        img0=img

        with torch.no_grad():

            img = np.transpose(img, (2, 0, 1))
            ref_imgs = [np.transpose(im, (2, 0, 1)) for im in ref_imgs]
            img = torch.from_numpy(img).unsqueeze(0)
            ref_imgs = [torch.from_numpy(im).unsqueeze(0) for im in ref_imgs]
            img = (img / 255 ).cuda()
            ref_imgs = [(im / 255 ).cuda() for im in ref_imgs]


            output= disp_net(img)#,raw_disp

            output_depth = 1 / output

            if args.pretrained_posenet is not None:
                explainability_mask,pose, pixel_pose,final_pose= pose_net(img, ref_imgs)#,raw_disp

                _, ego_flow = get_new_grid(output_depth[0], pose[:1,:], intrinsics, intrinsics_inv)
                _, rigid_flow = get_new_grid(output_depth[0], pixel_pose[:1, :], intrinsics, intrinsics_inv)
                _, final_flow = get_new_grid(output_depth[0], final_pose[:1, :], intrinsics, intrinsics_inv)

                exp = (255*tensor2array(explainability_mask[0].data.cpu(), max_value=None, colormap='bone')).astype(np.uint8).transpose(1,2,0)

                final_flow=final_flow[0].data.cpu().numpy()
                rigid_flow=rigid_flow[0].data.cpu().numpy()
                ego_flow=ego_flow[0].data.cpu().numpy()


                mask=explainability_mask[0].data.cpu().numpy().transpose((1,2,0))

                write_flow(final_flow, output_dir / 'final_flow_{}{}'.format(file.namebase, '.flo'))
                final_flow = flow_to_image(final_flow)
                imsave(output_dir / 'final_flow_{}{}'.format(file.namebase, file.ext), final_flow)

                write_flow(ego_flow,output_dir / 'ego_flow_{}{}'.format(file.namebase, '.flo'))
                ego_flow = flow_to_image(ego_flow)
                imsave(output_dir / 'ego_flow_{}{}'.format(file.namebase, file.ext), ego_flow)

            output=output[0].cpu()
            output_depth=output_depth[0,0].cpu()

            if args.output_disp:
                disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8).transpose(1,2,0)
                imsave(output_dir/'disp_{}{}'.format(file.namebase,file.ext), disp)
                np.save(output_dir/'depth_{}{}'.format(file.namebase,'.npy'),output_depth.data.numpy())
                np.save(output_dir/'pixel_pose_{}{}'.format(file.namebase,'.npy'),pixel_pose[0].cpu().data.numpy().transpose((1,2,0)))
                np.save(output_dir/'motion_mask_{}{}'.format(file.namebase,'.npy'),explainability_mask[0,0].cpu().data.numpy())
                np.save(output_dir/'ego_pose_{}{}'.format(file.namebase,'.npy'),pose[0,0].cpu().data.numpy())

                if args.pretrained_posenet is not None:
                    cat_im=np.concatenate((img0,disp,ego_flow,exp,final_flow),axis=1)
                else:
                    cat_im=np.concatenate((img0,disp),axis=1)
                imsave(output_dir / 'cat_{}{}'.format(file.namebase, file.ext), cat_im)


if __name__ == '__main__':
    main()

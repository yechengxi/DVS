import torch

from scipy.misc import imread, imsave, imresize
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
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
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


    if args.arch=='ecn':
        disp_net = models.ECN_Disp(input_size=args.img_height,init_planes=args.n_channel,scale_factor=args.scale_factor,growth_rate=args.growth_rate,final_map_size=args.final_map_size,norm_type=args.norm_type).cuda()
    else:
        disp_net = models.DispNetS().cuda()

    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if args.pretrained_posenet:
        output_disp = args.multi

        if args.arch == 'ecn':
            pose_net = models.ECN_Pose(input_size=args.img_height, nb_ref_imgs=args.sequence_length - 1,
                                           init_planes=args.n_channel // 2, scale_factor=args.scale_factor,
                                           growth_rate=args.growth_rate // 2, final_map_size=args.final_map_size,
                                           output_exp=True, output_exp2=True,
                                           output_pixel_pose=True,
                                           output_disp=args.multi, norm_type=args.norm_type).cuda()
        else:
            pose_net = models.PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp=True,
                                             output_pixel_pose=True, output_disp=True).cuda()


        weights = torch.load(args.pretrained_posenet)
        pose_net.load_state_dict(weights['state_dict'])
        pose_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()





    import os
    import glob


    scene=os.path.join(args.dataset_dir,args.dataset_list)
    intrinsics = np.genfromtxt(dataset_dir / args.dataset_list+'_cam.txt').astype(np.float32).reshape((3, 3))
    intrinsics_inv = np.linalg.inv(intrinsics)

    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).cuda()
    intrinsics_inv = torch.from_numpy(intrinsics_inv).unsqueeze(0).cuda()
    cnt_imgs = sorted(glob.glob(scene + '_cnt_*.jpg'))
    time_imgs = [img.replace('_cnt', '_time') for img in cnt_imgs]


    print('{} files to test'.format(len(cnt_imgs)))

    output_s_old=0
    ch2_old=0
    #for file in tqdm(test_files):

    class File:
        basename=None
        ext=None

    for i in range(len(cnt_imgs)-2):

        file =File()
        file.namebase=os.path.basename(cnt_imgs[i + 1]).replace('_cnt','').replace('.jpg','')
        file.ext='.jpg'

        cnt_img = imread(cnt_imgs[i + 1]).astype(np.float32)
        target_im1_cnt = imread(cnt_imgs[i + 2]).astype(np.float32)
        target_im2_cnt = imread(cnt_imgs[i]).astype(np.float32)

        time_img = imread(cnt_imgs[i + 1]).astype(np.float32)
        target_im1_time = imread(time_imgs[i + 2]).astype(np.float32)
        target_im2_time = imread(time_imgs[i]).astype(np.float32)

        h, w = cnt_img.shape

        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            cnt_img = imresize(cnt_img, (args.img_height, args.img_width)).astype(np.float32)
            target_im1_cnt = imresize(target_im1_cnt, (args.img_height, args.img_width)).astype(np.float32)
            target_im2_cnt = imresize(target_im2_cnt, (args.img_height, args.img_width)).astype(np.float32)
            time_img = imresize(time_img, (args.img_height, args.img_width)).astype(np.float32)
            target_im1_time = imresize(target_im1_time, (args.img_height, args.img_width)).astype(np.float32)
            target_im2_time = imresize(target_im2_time, (args.img_height, args.img_width)).astype(np.float32)

        img = np.stack([cnt_img, time_img], axis=2)
        target_im1 = np.stack([target_im1_cnt, target_im1_time], axis=2)
        target_im2 = np.stack([target_im2_cnt, target_im2_time], axis=2)


        with torch.no_grad():

            img = np.transpose(img, (2, 0, 1))
            target_im1 = np.transpose(target_im1, (2, 0, 1))
            target_im2 = np.transpose(target_im2, (2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0)
            tmp = img.mean(1)/255*10
            target_im1 = torch.from_numpy(target_im1).unsqueeze(0)
            target_im2 = torch.from_numpy(target_im2).unsqueeze(0)

            """
            img = ((img / 255 ) ).cuda()
            target_im1 = ((target_im1 / 255 )).cuda()
            target_im2 = ((target_im2 / 255 )).cuda()
            """

            img = ((img / 255 - 0.5) / 0.5).cuda()
            target_im1 = ((target_im1 / 255 - 0.5) / 0.5).cuda()
            target_im2 = ((target_im2 / 255 - 0.5) / 0.5).cuda()

            ref_imgs = [target_im1,target_im2]

            output_s= disp_net(img)#,raw_disp
            output_depth = 1 / output_s

            if args.pretrained_posenet is not None:
                explainability_mask, explainability_mask2, pixel_pose, output_m, pose= pose_net(img, ref_imgs)#,raw_disp

            _, ego_flow = get_new_grid(output_depth[0], pose[:,1], intrinsics, intrinsics_inv)

            ego_flow = flow_to_image(ego_flow[0].data.cpu().numpy())

            imsave(output_dir / 'ego_flow_{}{}'.format(file.namebase, file.ext), ego_flow)

            write_flow(ego_flow,output_dir / 'ego_flow_{}{}'.format(file.namebase, '.flo'))
            output_s=output_s[0].cpu()
            output_depth=output_depth[0,0].cpu()


            if args.output_disp:
                disp = (255*tensor2array(output_s, max_value=None, colormap='bone')).astype(np.uint8)
                imsave(output_dir/'disp_{}{}'.format(file.namebase,file.ext), disp)
                np.save(output_dir/'depth_{}{}'.format(file.namebase,'.npy'),output_depth.data.numpy())

            if args.output_depth:
                depth = (255*tensor2array(output_depth, max_value=10, colormap='rainbow')).astype(np.uint8)
                imsave(output_dir/'depth_{}{}'.format(file.namebase,file.ext), depth)



if __name__ == '__main__':
    main()

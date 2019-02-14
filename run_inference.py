import torch

from scipy.misc import imread, imsave, imresize
import numpy as np
from path import Path
import argparse

#from models.ECN_old import *
from models.ECN import *
from models.DispNetS import *
from models.PoseExpNet import *

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
parser.add_argument("--img-height", default=200, type=int, help="Image height")
parser.add_argument("--img-width", default=346, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")


parser.add_argument('--arch', default='ecn', help='architecture')
parser.add_argument('--norm-type', default='fd', help='normalization type')

parser.add_argument('--n-channel', '--init-channel', default=32, type=int,
                    help='initial feature channels(32|64|128).')
parser.add_argument('--growth-rate', default=32, type=int, help='feature channel growth rate.')
parser.add_argument('--scale-factor', default=1. / 2.,
                    type=float, help='scaling factor of each layer(0.5|0.75|0.875)')
parser.add_argument('--final-map-size', default=4, type=int, help='final map size')



def main():
    args = parser.parse_args()

    if args.arch=='ecn':
        disp_net = ECN_Disp(input_size=args.img_height,init_planes=args.n_channel,scale_factor=args.scale_factor,growth_rate=args.growth_rate,final_map_size=args.final_map_size,norm_type=args.norm_type).cuda()
    else:
        disp_net = DispNetS().cuda()

    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    if args.pretrained_posenet:

        if args.arch == 'ecn':
            pose_net = ECN_Pose(input_size=args.img_height, nb_ref_imgs=args.sequence_length - 1,
                                           init_planes=args.n_channel // 2, scale_factor=args.scale_factor,
                                           growth_rate=args.growth_rate // 2, final_map_size=args.final_map_size,
                                           output_exp=True,
                                           #output_exp2=True,
                                           #output_pixel_pose=True,
                                           #output_disp=args.multi,
                                           norm_type=args.norm_type).cuda()
        else:
            pose_net = PoseExpNet(nb_ref_imgs=args.sequence_length - 1, output_exp=True,
                                             output_pixel_pose=False, output_disp=False).cuda()


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
    #intrinsics = np.genfromtxt(dataset_dir / args.dataset_list+'_cam.txt').astype(np.float32).reshape((3, 3))
    intrinsics = np.genfromtxt(dataset_dir / 'cam.txt').astype(np.float32).reshape((3, 3))

    intrinsics_inv = np.linalg.inv(intrinsics)

    intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).cuda()
    intrinsics_inv = torch.from_numpy(intrinsics_inv).unsqueeze(0).cuda()
    #imgs = sorted(glob.glob(scene + '_cmb_*.jpg'))
    imgs = sorted(glob.glob(os.path.join(scene , '*.jpg')))

    print('{} files to test'.format(len(imgs)))

    #for file in tqdm(test_files):

    class File:
        basename=None
        ext=None

    demi_length = (args.sequence_length - 1) // 2
    shifts = list(range(-demi_length, demi_length + 1))
    shifts.pop(demi_length)


    for i in range(demi_length,len(imgs)-demi_length):
    #for i in range(828,829):

        file =File()
        file.namebase=os.path.basename(imgs[i]).replace('.jpg','')
        file.ext='.jpg'

        img0 = imread(imgs[i]).astype(np.float32)

        ref_imgs=[]
        for j in shifts:
            ref_imgs.append(imread(imgs[i + j]).astype(np.float32))

        h, w, _ = img0.shape

        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img0 = imresize(img0, (args.img_height, args.img_width)).astype(np.float32)
            ref_imgs=[imresize(im, (args.img_height, args.img_width)).astype(np.float32) for im in ref_imgs]


        with torch.no_grad():

            img = np.transpose(img0, (2, 0, 1))
            ref_imgs = [np.transpose(im, (2, 0, 1)) for im in ref_imgs]
            img = torch.from_numpy(img).unsqueeze(0)
            ref_imgs = [torch.from_numpy(im).unsqueeze(0) for im in ref_imgs]
            img = (img / 255 ).cuda()
            ref_imgs = [(im / 255 ).cuda() for im in ref_imgs]


            msg=None
            #msg = 'test'
            if args.arch=='ecn':
                output_s= disp_net(img,msg)#,msg#,raw_disp
            else:
                output_s = disp_net(img)

            output_depth = 1 / output_s

            if args.pretrained_posenet is not None:
                if args.arch=='ecn':
                    explainability_mask, pose = pose_net(img, ref_imgs)  # ,raw_disp
                else:
                    explainability_mask, explainability_mask2, pixel_pose, output_m, pose= pose_net(img, ref_imgs)#,raw_disp

                if args.arch=='ecn':
                    _, ego_flow = get_new_grid(output_depth[0], pose[:,int((args.sequence_length-1)/2)], intrinsics, intrinsics_inv)
                else:
                    _, ego_flow = inverse_warp(ref_imgs[int((args.sequence_length-1)/2)], output_depth[:, 0], pose[:,int((args.sequence_length-1)/2)], intrinsics,
                                               intrinsics_inv, 'euler', 'border')

                np.save(output_dir / 'ego_pose_{}{}'.format(file.namebase, '.npy'),pose[0,(args.sequence_length-1)//2].cpu().data.numpy())


                ego_flow=ego_flow[0].data.cpu().numpy()
                write_flow(ego_flow,output_dir / 'ego_flow_{}{}'.format(file.namebase, '.flo'))
                #tmp=read_flow(output_dir / 'ego_flow_{}{}'.format(file.namebase, '.flo'))
                ego_flow = flow_to_image(ego_flow)
                imsave(output_dir / 'ego_flow_{}{}'.format(file.namebase, file.ext), ego_flow)


            output_s=output_s[0].cpu()
            output_depth=output_depth[0,0].cpu()

            disp = (255*tensor2array(output_s, max_value=None, colormap='bone')).astype(np.uint8).transpose((1,2,0))
            imsave(output_dir/'disp_{}{}'.format(file.namebase,file.ext), disp)
            np.save(output_dir/'depth_{}{}'.format(file.namebase,'.npy'),output_depth.data.numpy())
            if args.pretrained_posenet is not None:
                cat_im=np.concatenate((img0,disp,ego_flow),axis=1)
            else:
                cat_im=np.concatenate((img0,disp),axis=1)
            imsave(output_dir / 'cat_{}{}'.format(file.namebase, file.ext), cat_im)


if __name__ == '__main__':
    main()

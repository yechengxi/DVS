#!/usr/bin/python3

import argparse
import sys, time
import csv

# OpenCV for Python 3 does not like ROS
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
import custom_transforms
from itertools import chain
from tensorboardX import SummaryWriter
import models

from dataloader import *
from flowlib import *
from utils import *


from inverse_warp import inverse_warp,simple_inverse_warp

from loss_functions import *
from logger import AverageMeter
from itertools import chain
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Structure from Motion Learner',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('--slices', type=int, metavar='N', help='slice length for training', default=0)
parser.add_argument('--duration', type=float, metavar='N', help='duration for a big slice', default=0.05)

parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')


parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers')

parser.add_argument('-c','--n-motions', default=4, type=int, metavar='N',
                    help='number of independent motions')

parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained-dispnet', dest='pretrained_dispnet', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-posenet', dest='pretrained_posenet', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-pixelnet', dest='pretrained_pixelnet', default=None, metavar='PATH',
                    help='path to pre-trained Pixel net model')
parser.add_argument('--seed', default=1, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=1)
parser.add_argument('-d', '--depth-loss-weight', type=float, help='weight for depth loss', metavar='W', default=.5)
parser.add_argument('--still-loss-weight', type=float, help='weight for still mask loss', metavar='W', default=0.0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=1.0)
parser.add_argument('-p','--pose-loss-weight', type=float, help='weight for pose smoothness loss', metavar='W', default=1)
parser.add_argument('-o','--flow-smooth-loss-weight', type=float, help='weight for optical flow smoothness loss', metavar='W', default=0.0)
parser.add_argument('--ssim-weight', type=float, help='weight for ssim loss', metavar='W', default=0.)
parser.add_argument('--pixelpose', action='store_true', help='use binary mask and pixel wise pose.')


parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')

parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)
parser.add_argument('--sharp', action='store_true',help='use sharpness loss')

parser.add_argument('--simple', action='store_true',help='use simple warping')

#optimizer
parser.add_argument('--optimizer', default='Adam', help='optimizer(SGD|Adam)')
parser.add_argument('--lr-scheduler', default='cosine', help='learning rate scheduler(multistep|cosine)')
parser.add_argument('--milestone', default=0.4, type=float, help='milestone in multistep scheduler')
parser.add_argument('--multistep-gamma', default=0.1, type=float,
                    help='the gamma parameter in multistep|plateau scheduler')

parser.add_argument('--arch', default='ecn', help='architecture')
parser.add_argument('--norm-type', default='gn', help='normalization type')


parser.add_argument('--n-channel', '--init-channel', default=32, type=int,
                    help='initial feature channels(32|64|128).')
parser.add_argument('--growth-rate', default=32, type=int, help='feature channel growth rate.')
parser.add_argument('--scale-factor', default=1. / 2.,
                    type=float, help='scaling factor of each layer(0.5|0.75|0.875)')
parser.add_argument('--final-map-size', default=4, type=int, help='final map size')

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")

parser.add_argument("--scale", default=1., type=float, help="rescaling factor")

best_error = -1
n_iter = 0


def main():
    global best_error, n_iter
    args = parser.parse_args()
    args.with_gt=True
    save_path = save_path_formatter(args, parser)
    args.save_path = 'checkpoints'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    #normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[.5,.5,.5])
    #normalize = custom_transforms.Normalize(mean=[0.5, 0.5],std=[.5,.5])

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        #normalize
    ])

    valid_transform = custom_transforms.Compose([#custom_transforms.CropBottom(),
                                                 custom_transforms.ArrayToTensor(),
                                                #normalize
                                                ])

    print("=> fetching scenes in '{}'".format(args.data))
    if args.slices>0:
        train_set = CloudSequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            slices=args.slices,
            duration=args.duration,
            scale=args.scale
        )

        # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
        val_set = ImageSequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
            scale=args.scale,
            gt=args.with_gt
        )
    else:
        train_set = ImageSequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            scale=args.scale
        )

        # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
        val_set = ImageSequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
            scale=args.scale,
            gt=args.with_gt
        )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")


    if args.arch=='ecn':
        disp_net = models.ECN_Disp(input_size=260,#260*args.scale*.5,
                                   init_planes=args.n_channel,scale_factor=args.scale_factor,growth_rate=args.growth_rate,final_map_size=args.final_map_size,norm_type=args.norm_type).cuda()


        if args.pixelpose:
            pose_exp_net = models.ECN_PixelPose(input_size=260,#260*args.scale*.5,
                                           nb_ref_imgs=args.sequence_length - 1,init_planes=args.n_channel//2,scale_factor=args.scale_factor,growth_rate=args.growth_rate//2,final_map_size=args.final_map_size,
                                              norm_type=args.norm_type).cuda()

        else:
            pose_exp_net = models.ECN_Pose(input_size=260,#260*args.scale*.5,
                                           nb_ref_imgs=args.sequence_length - 1,init_planes=args.n_channel//2,scale_factor=args.scale_factor,growth_rate=args.growth_rate//2,final_map_size=args.final_map_size,
                                              norm_type=args.norm_type,n_motions=args.n_motions).cuda()
    else:
        disp_net = models.DispNetS().cuda()
        pose_exp_net=models.PoseExpNet( nb_ref_imgs=args.sequence_length - 1).cuda()

    if args.pretrained_posenet:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_posenet)
        pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
    else:
        pose_exp_net.init_weights()

    if args.pretrained_dispnet:
        print("=> using pre-trained weights for Dispnet")
        weights = torch.load(args.pretrained_dispnet)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()



    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_exp_net = torch.nn.DataParallel(pose_exp_net)

    args.sharpness_loss=sharpness_loss().cuda()
    args.sharpness_loss=torch.nn.DataParallel(args.sharpness_loss)

    args.simple_photometric_reconstruction_loss = simple_photometric_reconstruction_loss().cuda()
    args.simple_photometric_reconstruction_loss = torch.nn.DataParallel(args.simple_photometric_reconstruction_loss)

    args.smooth_loss=smooth_loss().cuda()
    args.smooth_loss = torch.nn.DataParallel(args.smooth_loss)

    args.explainability_loss=explainability_loss().cuda()
    args.explainability_loss = torch.nn.DataParallel(args.explainability_loss)

    args.explainability_loss_new=explainability_loss_new().cuda()
    args.explainability_loss_new = torch.nn.DataParallel(args.explainability_loss_new)

    args.pose_smooth_loss=pose_smooth_loss().cuda()
    args.pose_smooth_loss = torch.nn.DataParallel(args.pose_smooth_loss)

    args.depth_loss=depth_loss().cuda()
    args.depth_loss=torch.nn.DataParallel(args.depth_loss)

    print('=> setting adam solver')

    parameters = chain(disp_net.parameters(), pose_exp_net.parameters())
    parameters = filter(lambda p: p.requires_grad, parameters)
    params = sum([np.prod(p.size()) for p in parameters])
    print(params,'trainable parameters in the network.')

    parameters = chain(disp_net.parameters(), pose_exp_net.parameters())
    parameters = filter(lambda p: p.requires_grad, parameters)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr,betas=(args.momentum, args.beta), weight_decay=args.weight_decay)


    if args.lr_scheduler=='multistep':
        milestones=[int(args.milestone*args.epochs)]
        while milestones[-1]+milestones[0]<args.epochs:
            milestones.append(milestones[-1]+milestones[0])
        args.current_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.multistep_gamma)

    if args.lr_scheduler=='cosine':
        import math
        total_steps = math.ceil(len(train_set)/args.batch_size)*args.epochs
        args.current_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=0, last_epoch=-1)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss'])

    for epoch in range(args.epochs):
        print('epoch: %d'%epoch)
        if args.lr_scheduler == 'multistep':
            args.current_scheduler.step()
        if args.lr_scheduler == 'multistep' or args.lr_scheduler == 'cosine':
            print('Current learning rate:', args.current_scheduler.get_lr()[0])

        #errors, error_names = validate_with_gt(args, val_loader, disp_net,pose_exp_net, epoch, output_writers)

        # train for one epoch
        train_loss = train(args, train_loader, disp_net, pose_exp_net, optimizer, args.epoch_size, training_writer)
        # evaluate on validation set
        errors, error_names = validate_with_gt(args, val_loader, disp_net, pose_exp_net, epoch, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_exp_net.module.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])

from random import random

def train(args, train_loader, disp_net, pose_exp_net,optimizer, epoch_size,  train_writer):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 ,w4, w5, w6 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight,args.pose_loss_weight,args.flow_smooth_loss_weight,args.depth_loss_weight

    loss,loss_1,loss_2,loss_3,loss_4,loss_5,loss_6=0,0,0,0,0,0,0
    # switch to train mode
    disp_net.train()
    pose_exp_net.train()

    start_time = time.time()
    end = time.time()

    for i, data in enumerate(train_loader):
        if len(data)==4:
            tgt_img, ref_imgs, intrinsics, intrinsics_inv=data
        if len(data)>=5:
            if len(data) == 5:
                tgt_img, ref_imgs, intrinsics, intrinsics_inv, gt=data
            if len(data) == 6:
                tgt_img, ref_imgs, intrinsics, intrinsics_inv, gt,slices = data
                if args.sharp:
                    slices = [img.cuda() for img in slices]
            if gt.shape[1]==2:
                gt_depth=gt[:,:1].cuda()
                gt_mask=torch.round(gt[:,1:].cuda())
            else:
                gt_mask=(gt[:,:1]<0.99).cuda()

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = tgt_img.cuda()
        ref_imgs_var = [img.cuda() for img in ref_imgs]
        intrinsics_var = intrinsics.cuda()
        intrinsics_inv_var = intrinsics_inv.cuda()


        # compute output
        disparities = disp_net(tgt_img_var)

        # normalize
        b = tgt_img.shape[0]
        mean_disp = disparities[0].view(b, -1).mean(-1).view(b, 1, 1, 1) * 0.1
        disparities = [disp / mean_disp for disp in disparities]

        depth = [1/disp for disp in disparities]
        if args.pixelpose:
            explainability_mask, pose, pixel_pose, final_pose = pose_exp_net(tgt_img_var, ref_imgs_var)
        else:
            explainability_mask, pose, final_pose = pose_exp_net(tgt_img_var, ref_imgs_var)

        final_flows=None
        pixel_flows=None



        loss_1,warped_refs, final_flows = args.simple_photometric_reconstruction_loss(tgt_img_var, ref_imgs_var,
                                                             intrinsics_var, intrinsics_inv_var,
                                                             depth, explainability_mask, final_pose,
                                                             args.ssim_weight, args.padding_mode)

        if args.pixelpose:
            loss_1_2, warped_refs_2, pixel_flows = args.simple_photometric_reconstruction_loss(tgt_img_var, ref_imgs_var,
                                                                                         intrinsics_var, intrinsics_inv_var,
                                                                                         depth, explainability_mask, pixel_pose,
                                                                                         args.ssim_weight,
                                                                                         args.padding_mode)

            loss_1=loss_1+loss_1_2


        if args.sharp:

            loss_1_slices,warped_slices,ego_flows_slices = args.sharpness_loss(slices,
                                                    intrinsics_var, intrinsics_inv_var,
                                                    depth, explainability_mask, final_pose,
                                                    args.padding_mode)
            loss_1=loss_1_slices

        loss_1=loss_1.mean()

        if args.n_motions==1:
            w2=0

        if w2 > 0:
            #loss_2 = args.explainability_loss(explainability_mask,gt_mask).mean()
            loss_2 = args.explainability_loss_new(explainability_mask,gt_mask).mean()
        else:
            loss_2 = 0

        if w3 > 0:

            loss_3 = args.smooth_loss(depth)#args.smooth_loss(depth)

            loss_3=loss_3.mean()

        else:
            loss_3=0.

        if w4 > 0:
            #loss_4 = args.pose_smooth_loss(pixel_pose,pose,gt_mask)
            loss_4 = args.pose_smooth_loss(final_pose,pose,gt_mask)

            loss_4=loss_4.mean()
        else:
            loss_4=0.

        loss_5 = 0
        if w5>0 and final_flows is not None:
            stacked_final_flow=[]
            #stacked_pixel_flow=[]
            if final_flows is not None:
                for i in range(len(final_flows)):
                    if final_flows is not None:
                        tmp=torch.cat(final_flows[i],dim=3).permute(0,3,1,2)
                        stacked_final_flow.append(tmp)
            if len(stacked_final_flow)>0:
                loss_5+=args.smooth_loss(stacked_final_flow).mean()
            #if len(stacked_pixel_flow) > 0:
            #    loss_5 += args.smooth_loss(stacked_pixel_flow).mean()
        else:
            w5=0

        loss_6=0

        gt_disp = 1 / gt_depth
        mean_disp = gt_disp.view(b, -1).mean(-1).view(b, 1, 1, 1)
        gt_disp = gt_disp / mean_disp

        if w6>0 and args.with_gt:
            #gt_depth=gt_depth.cuda()
            loss_6+=args.depth_loss(gt_depth,depth).mean()
        else:
            w6=0

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_4 +w5*loss_5+w6*loss_6

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            if w2 > 0:
                train_writer.add_scalar('explanability_loss', loss_2.item(), n_iter)
            if w3 > 0:
                train_writer.add_scalar('disparity_smoothness_loss', loss_3.item(), n_iter)
            if w4 > 0:
                train_writer.add_scalar('pose_smooth_loss', loss_4.item(), n_iter)
            if w5 > 0:
                train_writer.add_scalar('flow_smooth_loss', loss_5.item(), n_iter)
            if w6 > 0 and args.with_gt:
                train_writer.add_scalar('depth_loss', loss_6.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:

            train_writer.add_image('train Input', tensor2array(tgt_img[0],max_value=1,colormap='bone'), n_iter)

            if args.with_gt:
                train_writer.add_image('train gt disp', tensor2array(gt_disp[0].cpu().data, max_value=None, colormap='bone'), n_iter)
                train_writer.add_image('train gt depth', tensor2array(gt_depth[0].cpu().data, max_value=None), n_iter)

            if args.sharp:
                for j in range(len(warped_slices[0])):
                    if j == 0 or j == len(warped_slices[0]) - 1 or j == ((len(warped_slices[0]) - 1) // 2):
                        train_writer.add_image('warped slice {}'.format(j), tensor2array(warped_slices[0][j][0].data.cpu(), max_value=1, colormap='bone'),n_iter)

                #stack

                stacked_im = 0.
                counter_im = 0

                for j in range(len(warped_slices[0])):
                    ref_warped = warped_slices[0][j][0]  # slices[j][0]#

                    event_im = ref_warped[0].abs() + ref_warped[2].abs()
                    ref_warped[1] = (ref_warped[1]/ args.slices + j / args.slices) * event_im

                    counter_im += event_im
                    stacked_im = stacked_im + ref_warped

                stacked_im[1][counter_im < 0.99 * 50/args.duration*0.05 / 255.] = 0
                stacked_im[1] = stacked_im[1] / (counter_im + 1e-3)

                mask = (counter_im > (0.)).type_as(counter_im)  # 4./255*50
                stacked_im[1] = stacked_im[1] * mask
                #stacked_im[1]=0
                train_writer.add_image('train stacked slices',
                                       tensor2array(stacked_im.abs().data.cpu(), colormap='bone', max_value=None), n_iter)

                train_writer.add_image('train event mask',
                                       tensor2array(mask.abs().data.cpu(), colormap='bone', max_value=None), n_iter)

            for k,scaled_depth in enumerate(depth):
                train_writer.add_image('train Dispnet Output Normalized {}'.format(k),tensor2array(disparities[k].data[0].cpu(), max_value=None, colormap='bone'),n_iter)
                b, _, h, w = scaled_depth.size()

                warped_refs_scaled = warped_refs[k]

                # log warped images along with explainability mask
                stacked_im = 0.

                tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img_var, (h, w))
                middle_slice = tgt_img_scaled[0]

                for j in range(len(warped_refs_scaled)):

                    if j==0:
                        if explainability_mask[k] is not None:
                            if args.pixelpose:
                                mask=explainability_mask[k][0, 0].data.cpu()
                            else:
                                mask=1-explainability_mask[k][0, 0].data.cpu()
                            train_writer.add_image('train Exp mask Outputs {}'.format(k),
                                                   tensor2array(mask, max_value=1,
                                                                colormap='bone'), n_iter)
                    if k==0:
                        if (explainability_mask[0].shape[1]>=3):
                            mask=explainability_mask[0][0, 0:3].data.cpu()
                            train_writer.add_image('train Exp components',
                                                   tensor2array(mask, max_value=1,
                                                                colormap='bone'), n_iter)


                    ref_warped = warped_refs_scaled[j][0]
                    stacked_im = stacked_im + ref_warped

                    if final_flows is not None:
                        final_flow = flow_to_image(final_flows[k][j][0].data.cpu().numpy()).transpose(2,0,1)
                        train_writer.add_image('final flow {} {}'.format(k, j), final_flow / 255, n_iter)

                    if pixel_flows is not None:
                        pixel_flow = flow_to_image(pixel_flows[k][j][0].data.cpu().numpy()).transpose(2, 0, 1)
                        train_writer.add_image('pixel flow {} {}'.format(k, j), pixel_flow / 255, n_iter)

                    train_writer.add_image('train Warped Outputs {} {}'.format(k, j),
                                           tensor2array(ref_warped.data.cpu(), colormap='bone', max_value=1), n_iter)

                    train_writer.add_image('train Diff Outputs {} {}'.format(k, j),
                                           tensor2array((middle_slice - ref_warped).abs().data.cpu(), colormap='bone',
                                                        max_value=1.), n_iter)

                stacked_im[1]=0
                train_writer.add_image('train stacked Outputs {}'.format(k), tensor2array(stacked_im.abs().data.cpu(),colormap='bone',max_value=None), n_iter)



        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.lr_scheduler == 'cosine':
            args.current_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item() if w3 > 0 else 0])
        if i >= epoch_size - 1:
            break

        n_iter += 1

    end_time = time.time()
    print('Training loss: %.5f, elasped time: %3.f seconds.'% (losses.avg[0], end_time-start_time))

    return losses.avg[0]



def validate_with_gt(args, val_loader, disp_net, pose_exp_net, epoch, output_writers=[]):
    batch_time = AverageMeter()
    error_names = ['Total loss', 'Photo loss', 'Exp loss','abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0
    if log_outputs:
        log_freq=len(val_loader)//len(output_writers)

    # switch to evaluate mode
    disp_net.eval()
    pose_exp_net.eval()
    w1, w2, w3 ,w4, w5, w6 = args.photo_loss_weight, args.mask_loss_weight, args.smooth_loss_weight,args.pose_loss_weight,args.flow_smooth_loss_weight,args.depth_loss_weight

    start_time = time.time()

    end = time.time()

    for i, data in enumerate(val_loader):
        if len(data) == 4:
            tgt_img, ref_imgs, intrinsics, intrinsics_inv = data
        if len(data) >= 5:
            if len(data) == 5:
                tgt_img, ref_imgs, intrinsics, intrinsics_inv, gt = data
            if len(data) == 6:
                tgt_img, ref_imgs, intrinsics, intrinsics_inv, gt, slices = data
                if args.sharp:
                    slices = [img.cuda() for img in slices]
            if gt.shape[1] == 2:
                gt_depth = gt[:, :1].cuda()
                gt_mask = torch.round(gt[:, 1:].cuda())
            else:
                gt_mask = (gt[:, :1] < 0.99).cuda()


        with torch.no_grad():
            tgt_img_var = tgt_img.cuda()
            ref_imgs_var = [img.cuda() for img in ref_imgs]
            intrinsics_var = intrinsics.cuda()
            intrinsics_inv_var = intrinsics_inv.cuda()
            if gt.shape[1]==2:
                gt_depth=gt[:,:1].cuda()
                gt_mask=gt[:,1:].cuda()
            else:
                gt_mask=(gt[:,:1]<0.99).cuda()

            gt_depth = gt_depth.cuda()
            if len(gt_depth.shape)==4:
                gt_depth=gt_depth.squeeze(1)
            # compute output
            output_disp = disp_net(tgt_img_var)

            # normalize
            b = tgt_img.shape[0]
            mean_disp = output_disp.view(b, -1).mean(-1).view(b, 1, 1, 1) * 0.1
            output_disp = output_disp / mean_disp

            output_disp=F.adaptive_avg_pool2d(output_disp,gt_depth.shape[-2:])

            output_depth = 1/output_disp

            if args.pixelpose:
                explainability_mask, pose,pixel_pose, final_pose = pose_exp_net(tgt_img_var, ref_imgs_var)
            else:
                explainability_mask, pose, final_pose = pose_exp_net(tgt_img_var, ref_imgs_var)

            loss_1, warped_refs, final_flows = args.simple_photometric_reconstruction_loss(tgt_img_var, ref_imgs_var,
                                                                                         intrinsics_var,
                                                                                         intrinsics_inv_var,
                                                                                         output_depth, explainability_mask,
                                                                                         final_pose,
                                                                                         args.ssim_weight,
                                                                                         args.padding_mode)

            loss_1 = loss_1.mean()

            if args.n_motions==1:
                w2=0

            if w2 > 0:
                loss_2 = args.explainability_loss(explainability_mask, gt_mask).mean()
            else:
                loss_2 = 0

            loss = w1*loss_1 + w2*loss_2

            if log_outputs and i % log_freq == 0 and i/log_freq < len(output_writers):
                index = int(i//log_freq)
                if epoch == 0:
                    output_writers[index].add_image('val Input', tensor2array(tgt_img[0],colormap='bone',max_value=1), 0)
                    depth_to_show = gt_depth[0].cpu()
                    output_writers[index].add_image('val target Depth', tensor2array(depth_to_show, max_value=10), epoch)
                    depth_to_show[depth_to_show == 0] = 1000
                    disp_to_show = (1/depth_to_show).clamp(0,10)
                    output_writers[index].add_image('val target Disparity Normalized', tensor2array(disp_to_show, max_value=None, colormap='bone'), epoch)

                output_writers[index].add_image('val Dispnet Output Normalized', tensor2array(output_disp.data[0].cpu(), max_value=None, colormap='bone'), epoch)

                output_writers[index].add_image('val Depth Output', tensor2array(output_depth.data[0].cpu(), max_value=3), epoch)


                if explainability_mask is not None:
                    output_writers[index].add_image('val Exp mask Outputs',tensor2array(explainability_mask[0, 0].data.cpu(), max_value=1,colormap='bone'), epoch)


                for j,ref in enumerate(ref_imgs_var):
                    final_flow = final_flows[0][j][0]
                    final_flow = flow_to_image(final_flow.data.cpu().numpy()).transpose(2, 0, 1)
                    output_writers[index].add_image('val final flow {}'.format(j), final_flow / 255, n_iter)

                    ref_warped = warped_refs[0][j][0]
                    output_writers[index].add_image('val Warped Outputs {}'.format(j), tensor2array(ref_warped.data.cpu(),colormap='bone',max_value=1), n_iter)
                    output_writers[index].add_image('val Diff Outputs {}'.format(j), tensor2array((tgt_img_var[0] - ref_warped).abs().data.cpu(),colormap='bone',max_value=1), n_iter)



            errors_tmp=compute_errors(gt_depth, output_depth.data)
            errors_tmp=[loss, loss_1, loss_2]+errors_tmp
            errors_tmp = [e.item() if isinstance(e, torch.Tensor) else e for e in errors_tmp]
            errors.update(errors_tmp)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    end_time = time.time()

    msg = 'Evaluation. '

    for i in range(len(error_names)):
        msg += error_names[i]
        msg += ': '
        msg += str(round(errors.avg[i], 3))

        msg += '; '

    msg += ' Elapsed time: '
    msg += str(round(end_time - start_time, 3))
    msg += 'secs.'
    print(msg)

    return errors.avg, error_names




if __name__ == '__main__':
    main()

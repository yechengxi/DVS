from __future__ import division
import torch
from torch import nn
from torch.autograd import Variable
from inverse_warp import inverse_warp

import torch.nn.functional as F


import numpy as np
from inverse_warp import *

class simple_photometric_reconstruction_loss(nn.Module):
    def __init__(self):
        super(simple_photometric_reconstruction_loss, self).__init__()

    def forward(self, tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose,ssim_w=0.,padding_mode='zeros'):
        def one_scale(depth,explainability_mask,pose):
            reconstruction_loss = 0
            b, _, h, w = depth.size()
            downscale = tgt_img.size(2)/h
            ego_flows_scaled=[]
            refs_warped_scaled = []
            grids=[]
            tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
            ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)
            if pose.size(1)==1 or pose.size(1)==6:
                refs_warped_scaled, grids, ego_flows_scaled = multi_inverse_warp(ref_imgs_scaled, depth[:, 0], pose,
                                                                              intrinsics_scaled,
                                                                              intrinsics_scaled_inv, padding_mode)

            else:

                for i, ref_img in enumerate(ref_imgs_scaled):
                    if pose.size(1) == len(ref_imgs):
                        current_pose = pose[:, i]
                    elif pose.size(1)==len(ref_imgs)*6:
                        current_pose=pose[:,i*6:(i+1)*6]
                    ref_img_warped,grid,ego_flow = simple_inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, padding_mode)
                    refs_warped_scaled.append(ref_img_warped)
                    grids.append(grid)
                    ego_flows_scaled.append(ego_flow)

            for i in range(len(refs_warped_scaled)):
                #grid = grids[i]
                diff = (tgt_img_scaled - refs_warped_scaled[i])
                reconstruction_loss += diff.abs().view(b,-1).mean(1)
            return reconstruction_loss,refs_warped_scaled,ego_flows_scaled

        if type(explainability_mask) not in [tuple, list]:
            explainability_mask = [explainability_mask]
        if type(depth) not in [list, tuple]:
            depth = [depth]
        if type(pose) in [tuple, list]:
            assert len(pose)==len(depth)
        else:
            pose=[pose for i in range(len(depth))]
        loss = 0
        ego_flows=[]
        warped_refs=[]

        weight=0
        for d, mask, p in zip(depth, explainability_mask,pose):
            current_loss,refs_warped_scaled,ego_flows_scaled= one_scale(d, mask,p)
            _, _, h, w = d.size()
            weight+=h*w
            loss=loss+current_loss*h*w
            ego_flows.append(ego_flows_scaled)
            warped_refs.append(refs_warped_scaled)
        loss=loss/weight

        return loss,warped_refs,ego_flows




class explainability_loss(nn.Module):
    def __init__(self):
        super(explainability_loss, self).__init__()
    def forward(self,mask,gt_mask):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(mask) not in [tuple, list]:
            mask = [mask]
        loss = 0
        weight=0
        for mask_scaled in mask:
            N,_,H,W=mask_scaled.shape
            dx, dy = gradient(mask_scaled)
            loss += (dx.abs().view(N, -1).mean(1) + dy.abs().view(N, -1).mean(1)) * H * W
            ones_var = F.adaptive_avg_pool2d(gt_mask.type_as(mask_scaled),(H,W))
            loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)*H*W
            weight+=H*W
        return loss/weight


class depth_loss(nn.Module):
    def __init__(self):
        super(depth_loss, self).__init__()
    def forward(self, gt, predicts,eps=1e-5):
        weight=0
        abs_rel=0.
        acc=0.
        for pred in predicts:
            N, C, H, W = pred.shape
            current_gt = F.adaptive_avg_pool2d(gt, (H, W))
            weight += H * W
            valid = ((current_gt > 1/255) * (current_gt < 1000/255)).type_as(gt)
            masked_gt=current_gt*valid
            masked_pred=pred*valid
            pred = pred * (torch.mean(masked_gt.view(N,-1),1) / (eps+torch.mean(masked_pred.view(N,-1),1))).view(N,1,1,1)
            thresh = torch.max((masked_gt / (eps+pred)), (pred / (eps+masked_gt)))*valid
            cost=(torch.abs(current_gt - pred) / current_gt)*valid
            abs_rel += cost.view(N, -1).mean(1)*H*W
            acc+=thresh.view(N,-1).mean(1)*H*W
        return (abs_rel+acc)/weight

class smooth_loss(nn.Module):
    def __init__(self):
        super(smooth_loss, self).__init__()

    def forward(self, pred_map,p=.5,eps=1e-4):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight =0

        for scaled_map in pred_map:
            N,C,H,W = scaled_map.shape
            if min(H,W)<4:
                continue
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            #loss += (dx2.abs().view(N,-1).mean(1) + dxdy.abs().view(N,-1).mean(1) + dydx.abs().view(N,-1).mean(1) + dy2.abs().view(N,-1).mean(1))*weight
            loss += (torch.pow(torch.clamp(dx2.abs(), min=eps),p).view(N, -1).mean(1)
                     + torch.pow(torch.clamp(dxdy.abs(), min=eps),p).view(N, -1).mean(1)
                        + torch.pow(torch.clamp(dydx.abs(), min=eps),p).view(N, -1).mean(1)
                           + torch.pow(torch.clamp(dy2.abs(), min=eps),p).view(N, -1).mean(1)) * H*W

            weight += H*W

        return loss/weight





class pose_smooth_loss(nn.Module):
    def __init__(self):
        super(pose_smooth_loss, self).__init__()

    def forward(self, pred_map,pose,mask):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight = 0
        pose = pose.view(-1, 6, 1, 1)

        for i,scaled_map in enumerate(pred_map):
            N, _, H, W = scaled_map.shape

            if H > 3 and W > 3:
                dx, dy = gradient(scaled_map)
                loss += (dx.abs().view(N, -1).mean(1) + dy.abs().view(N, -1).mean(1)) * H*W
                loss += 10*(((scaled_map-pose).abs()*F.adaptive_avg_pool2d(mask,(H,W))).view(N, -1)).mean(1)
                weight += H*W
        return loss/weight





def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[0][valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


from __future__ import division
import torch
from torch import nn
from torch.autograd import Variable
from inverse_warp import *

import torch.nn.functional as F

import numpy as np

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
            tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
            ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

            for i, ref_img in enumerate(ref_imgs_scaled):
                if pose.size(1) == len(ref_imgs):
                    current_pose = pose[:, i]
                elif pose.size(1)==len(ref_imgs)*6:
                    current_pose=pose[:,i*6:(i+1)*6]

                ref_img_warped,_,ego_flow = simple_inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, padding_mode)
                out_of_bound = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)

                diff = (tgt_img_scaled - ref_img_warped) * out_of_bound
                if explainability_mask is not None:
                    diff = diff * explainability_mask[:,i:i+1].expand_as(diff)
                if ssim_w>0 and min(ref_img_warped.shape[2:])>11:
                    mask=1
                    if explainability_mask is not None:
                        mask=explainability_mask[:, i:i + 1]
                    ssim_loss = ssim(tgt_img_scaled,ref_img_warped,size_average=False,mask=out_of_bound*mask)
                else:
                    ssim_loss=0.

                reconstruction_loss += diff.abs().view(b,-1).mean(1)+ssim_w*ssim_loss
                ego_flows_scaled.append(ego_flow)
                refs_warped_scaled.append(ref_img_warped)
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




class sharpness_loss(nn.Module):
    def __init__(self):
        super(sharpness_loss, self).__init__()

    def forward(self, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, padding_mode='zeros'):
        def one_scale(depth,explainability_mask,pose):

            sharpness_loss = 0
            b, _, h, w = depth.size()
            downscale = ref_imgs[0].size(2)/h
            ego_flows_scaled=[]
            ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]
            intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

            stacked_im=0.

            ref_imgs_warped, grids, ego_flows_scaled = multi_inverse_warp(ref_imgs_scaled, depth[:, 0], pose, intrinsics_scaled,
                                                              intrinsics_scaled_inv, padding_mode)
            for i in range(len(ref_imgs)):
                ref_img=ref_imgs_scaled[i]
                ref_img_warped=ref_imgs_warped[i]
                new_grid=grids[i]
                in_bound = (new_grid[:,:,:,0]!=2).type_as(ref_img_warped).unsqueeze(1)
                #print(ref_img.min(),ref_img.mean(),ref_img.max(),ref_img_warped.min(),ref_img_warped.mean(),ref_img_warped.max())
                scaling = ref_img.view(b, 3, -1).mean(-1) / (1e-5 + ref_img_warped.view(b, 3, -1).mean(-1))
                #print(scaling.view(1,-1))
                stacked_im = stacked_im + ref_img_warped #* in_bound* scaling.view(b, 3, 1, 1)
            stacked_im=torch.pow(stacked_im.abs()+1e-4, .5)
            if explainability_mask is not None:
                stacked_im = stacked_im * explainability_mask[:, 0:1]
            stacked_im=stacked_im[:,0]+stacked_im[:,2]#take the event channels
            sharpness_loss += stacked_im.view(b, -1).mean(1)

            return sharpness_loss,ref_imgs_warped,ego_flows_scaled

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
            current_loss,ref_imgs_warped,ego_flows_scaled= one_scale(d, mask,p)
            _, _, h, w = d.size()
            weight += h * w
            loss = loss + current_loss * h * w
            ego_flows.append(ego_flows_scaled)
            warped_refs.append(ref_imgs_warped)

        loss = loss / weight
        return loss,warped_refs,ego_flows




def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    weight=0
    for mask_scaled in mask:
        N,C,H,W=mask_scaled.shape
        weight += H * W
        ones_var = mask_scaled.new_ones(1).expand_as(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)*H*W
    return loss/weight




class joint_smooth_loss(nn.Module):
    def __init__(self):
        super(joint_smooth_loss, self).__init__()

    def forward(self, pred_map,joint,p=1,eps=5e-2):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight =0

        mask=((joint[:,:1].abs()+joint[:,2:].abs())>0).type_as(joint)
        for scaled_map in pred_map:
            dx, dy = gradient(scaled_map)
            dx=dx[:,:,1:-1,:-1]
            dy=dy[:,:,:-1,1:-1]
            N, _, H, W = dx.shape

            """
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            dx2=dx2[:,:,1:-1,:]
            dxdy=dxdy[:,:,:-1,:-1]
            dydx = dydx[:, :, :-1, :-1]
            dy2 = dy2[:, :, :, 1:-1]
            N,_,H,W=dx2.shape
            """

            scaled_mask = (F.adaptive_avg_pool2d(mask, (H, W))>0.01).type_as(joint)

            loss += ((dx.abs()+dy.abs())*(1-scaled_mask)).view(N, -1).mean(1)*H*W
            weight += H * W

        return loss/weight


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



class non_local_smooth_loss(nn.Module):
    def __init__(self):
        super(non_local_smooth_loss, self).__init__()

    def forward(self, pred_map,p=.5,eps=1e-4):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight = 1.

        for scaled_map in pred_map:
            N = scaled_map.shape[0]
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            #loss += (dx2.abs().view(N,-1).mean(1) + dxdy.abs().view(N,-1).mean(1) + dydx.abs().view(N,-1).mean(1) + dy2.abs().view(N,-1).mean(1))*weight
            #loss += JointSmoothnessLoss(dx2,dx2).view(N, -1).mean(1)+JointSmoothnessLoss(dxdy,dxdy).view(N, -1).mean(1)+JointSmoothnessLoss(dy2,dy2).view(N, -1).mean(1)+JointSmoothnessLoss(dydx,dydx).view(N, -1).mean(1)
            dd=torch.cat((dx2[:,:,1:-1,:],dy2[:,:,:,1:-1],dxdy[:,:,:-1,:-1],dydx[:,:,:-1,:-1]),dim=1)
            loss+=NonLocalSmoothnessLoss(dd,p=0.8,eps=1e-4, R=0.1, B=10)
            weight /= 2.3 # don't ask me why it works better

        return loss




def pose_smooth_loss(pred_map,pose_ego):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1
    for scaled_map in pred_map:
        N, _, H, W = scaled_map.shape
        if H > 3 and W > 3:
            scaled_map=scaled_map-pose_ego.view(N,-1,1,1)
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss += (dx2.abs().view(N, -1).mean(1) + dxdy.abs().view(N, -1).mean(1) + dydx.abs().view(N, -1).mean(
                1) + dy2.abs().view(N, -1).mean(1)) * weight

            weight /= 2.3 # don't ask me why it works better
    return loss






def SimpleSmoothnessLoss(I,p=1,eps=1e-2):
    loss=0.
    N, C, H, W= I.shape

    W1=torch.pow(torch.abs(I[:, :, 2:, 1:-1] - I[:, :, 1:-1, 1:-1])+ eps, p-2)
    I1=(I[:, :, 2:, 1:-1] - I[:, :, 1:-1, 1:-1]) ** 2
    W2=torch.pow(torch.abs(I[:, :, 1:-1, 2:] - I[:, :, 1:-1, 1:-1])+ eps,p-2)
    I2=(I[:, :, 1:-1, 2:] - I[:, :, 1:-1, 1:-1]) ** 2
    W3=torch.pow(torch.abs(I[:, :, :-2, 1:-1] - I[:, :, 1:-1, 1:-1]) + eps, p - 2)
    I3=(I[:, :, :-2, 1:-1] - I[:, :, 1:-1, 1:-1]) ** 2
    W4=torch.pow(torch.abs(I[:, :, 1:-1, :-2] - I[:, :, 1:-1, 1:-1]) + eps, p - 2)
    I4=(I[:, :, 1:-1, :-2] - I[:, :, 1:-1, 1:-1]) ** 2
    loss=(W1*I1+W2*I2+W3*I3+W4*I4)/(W1+W2+W3+W4)
    loss = torch.mean(loss.view(N,-1),1)
    return loss


def box_filter(tensor,R):
    N,C,H,W=tensor.shape
    cumsum = torch.cumsum(tensor, dim=2)
    slidesum=torch.cat((cumsum[:, :, R:2 * R + 1, :],cumsum[:, :, 2 * R + 1:H, :] - cumsum[:, :, 0:H - 2 * R - 1, :],cumsum[:, :, -1:, :] - cumsum[:, :, H - 2 * R - 1:H - R - 1, :]),dim=2)
    cumsum = torch.cumsum(slidesum, dim=3)
    slidesum=torch.cat((cumsum[:, :, :, R:2 * R + 1],cumsum[:, :, :, 2 * R + 1:W] - cumsum[:, :, :, 0:W - 2 * R - 1],cumsum[:, :, :, -1:] - cumsum[:, :, :, W - 2 * R - 1:W - R - 1]),dim=3)
    return slidesum




def NonLocalSmoothnessLoss(I, p=1.0,eps=1e-4, R=0.1, B=10 ):

    N, C, H, W = I.shape

    R=int(min(H,W)*R)
    if H<10 or W<10 or R<2:
        return SimpleSmoothnessLoss(I,p,eps)

    loss = 0.

    J=I

    min_J, _ = torch.min(J.view(N, C, -1), dim=2)
    max_J, _ = torch.max(J.view(N, C, -1), dim=2)
    min_J = min_J.view(N, C, 1, 1, 1)
    max_J = max_J.view(N, C, 1, 1, 1)
    Q = torch.from_numpy(np.linspace(0.0, 1.0, B + 1)).type_as(min_J).view(1, 1, 1, 1, B + 1)
    Q = Q * (max_J - min_J + 1e-5) + min_J
    min_J = min_J.view(N, C, 1, 1)
    max_J = max_J.view(N, C, 1, 1)
    Bin1 = torch.floor((J - min_J) / (max_J - min_J + 1e-5) * B).long()
    Bin2 = torch.ceil((J - min_J) / (max_J - min_J + 1e-5) * B).long()

    I_old = I#.detach()

    W1 = (torch.abs(J - Q[:, :, :, :, 0]) + eps) ** (p - 2)
    W_sum1 = box_filter(W1, R)
    WI_sum1 = box_filter(W1 * I_old, R)
    WI2_sum1 = box_filter(W1 * (I_old ** 2), R)
    loss1 = W_sum1 * (I ** 2) - 2 * I * WI_sum1 + WI2_sum1

    W_sum = 0

    for i in range(1, B + 1):

        W2 = (torch.abs(J - Q[:, :, :, :, i]) + eps) ** (p - 2)
        W_sum2 = box_filter(W2, R)
        WI_sum2 = box_filter(W2 * I_old, R)
        WI2_sum2 = box_filter(W2 * (I_old ** 2), R)
        loss2 = W_sum2 * (I ** 2) - 2 * I * WI_sum2 + WI2_sum2

        mask1 = (Bin1 == (i - 1)).float()
        mask2 = (Bin2 == i).float()


        slice_loss = (loss1 * (Q[:, :, :, :, i] - J) * mask1 + loss2 * (J - Q[:, :, :, :, i - 1]) * mask2) / (
                    Q[:, :, :, :, i] - Q[:, :, :, :, i - 1])

        loss = loss + slice_loss

        W_sum = W_sum + (
                    W_sum1 * (Q[:, :, :, :, i] - J) * mask1 + W_sum2 * (J - Q[:, :, :, :, i - 1]) * mask2) / (
                            Q[:, :, :, :, i] - Q[:, :, :, :, i - 1])

        W_sum1 = W_sum2
        loss1 = loss2

    loss = torch.mean((loss / W_sum).view(N,-1),1)

    return loss



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


#for ssim

from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2,mask=None, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ssim_map=1-ssim_map #convert to a loss function

    if mask is not None:
        crop=(real_size)//2
        ssim_map=ssim_map*mask[:,:,crop:-crop,crop:-crop]

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

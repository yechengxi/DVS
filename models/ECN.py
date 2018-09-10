import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .normalization import *

class SingleConvBlock(nn.Module):
    # basic convolution block v1
    def __init__(self, in_planes, out_planes, kernel_size, padding, norm_type='gn', norm_group=16):
        super(SingleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1s = None
        if norm_type == 'bn':
            self.bn1s = nn.BatchNorm2d(in_planes,momentum=0.1)
        if norm_type == 'gn':
            self.bn1s = GroupNorm(in_planes, num_groups=norm_group)
        if norm_type == 'fd':
            self.bn1s = FeatureDecorr(in_planes, num_groups=norm_group)

    def forward(self, x):
        if self.bn1s is not None:
            x = self.bn1s(x)
        out = self.conv1(F.relu(x))
        return out



class DoubleConvBlock(nn.Module):
    # basic convolution block v2
    def __init__(self, in_planes, out_planes, kernel_size, padding, norm_type='gn', norm_group=16):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False)

        self.bn1s = None
        self.bn2s = None
        if norm_type == 'bn':
            self.bn1s = nn.BatchNorm2d(in_planes,momentum=0.1)
            self.bn2s = nn.BatchNorm2d(out_planes, momentum=0.1)
        if norm_type == 'gn':
            self.bn1s = GroupNorm(in_planes, num_groups=norm_group)
            self.bn2s = GroupNorm(out_planes, num_groups=norm_group)
        if norm_type == 'fd':
            self.bn1s = FeatureDecorr(in_planes, num_groups=norm_group)
            self.bn2s = FeatureDecorr(out_planes, num_groups=norm_group)

    def forward(self, x):
        if self.bn1s is not None:
            x = self.bn1s(x)
        out = self.conv1(F.relu(x))
        if self.bn2s is not None:
            out = self.bn2s(out)
        out = self.conv2(F.relu(out))
        return out




class RecurrentConvBlockC(nn.Module):
    # basic convolution block v12
    def __init__(self, in_planes, out_planes, kernel_size, padding,dropout_rate=0.,norm_type='gn', norm_group=16, n_iter=3):
        super(RecurrentConvBlockC, self).__init__()
        self.n_iter=n_iter
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1s=nn.ModuleList()

        if n_iter > 1:
            self.conv0 = nn.Conv2d(out_planes,in_planes, kernel_size=1, padding=0, bias=False)
            self.bn0s=nn.ModuleList()

        self.dropouts=nn.ModuleList()

        for i in range(n_iter):
            self.bn1s.append(nn.BatchNorm2d(in_planes))
            if norm_type == 'bn':
                self.bn1s.append(nn.BatchNorm2d(in_planes, momentum=0.1))
            if norm_type == 'gn':
                self.bn1s.append(GroupNorm(in_planes, num_groups=norm_group))
            if norm_type == 'fd':
                self.bn1s.append(FeatureDecorr(in_planes, num_groups=norm_group))

            if dropout_rate>0:
                self.dropouts.append(nn.Dropout2d(dropout_rate))
            if i+1<n_iter:
                if norm_type == 'bn':
                    self.bn0s.append(nn.BatchNorm2d(out_planes, momentum=0.1))
                if norm_type == 'gn':
                    self.bn0s.append(GroupNorm(out_planes, num_groups=norm_group))
                if norm_type == 'fd':
                    self.bn0s.append(FeatureDecorr(out_planes, num_groups=norm_group))

    def forward(self, x):
        C=x.shape[1]
        for i in range(self.n_iter):
            if i > 0:
                out1 = self.conv0(F.relu(self.bn0s[i-1](out)))
                out1=F.relu(self.bn1s[i](out1))
                if len(self.dropouts)>0:
                    out1=self.dropouts[i](out1)
                out = out + self.conv1(out1)
            else:
                x=F.relu(self.bn1s[i](x))
                if len(self.dropouts) > 0:
                    x = self.dropouts[i](x)
                out = self.conv1(x)
        return out






def scaling(maps, scaling_factor=None, output_size=None):
    N, C, H, W = maps.shape

    if scaling_factor is not None:
        # target map size
        H_t = math.floor(H * scaling_factor)
        W_t = math.floor(W * scaling_factor)
        pool_size = int(math.floor(1. / scaling_factor))
        min_pool_size = pool_size

    if output_size is not None:
        _, _, H_t, W_t = output_size
        scaling_factor = [H_t / H, W_t / W]
        pool_size = [int(math.floor(1. / scaling_factor[0])), int(math.floor(1. / scaling_factor[1]))]
        min_pool_size = min(pool_size)

    if min_pool_size >= 2:
        maps = F.avg_pool2d(maps, pool_size, ceil_mode=True)
        N, C, H, W = maps.shape

    if H != H_t or W != W_t:
        """
        # calculate the coordinates of the sampling grid
        base_grid = maps.new_empty(N, H_t, W_t, 2, requires_grad=False)
        linear_points = torch.from_numpy(np.linspace(-1., 1., W_t)).type_as(base_grid).to(maps.device)
        base_grid[..., 0] = linear_points.view(1, 1, -1).repeat(1, H_t, 1).expand_as(base_grid[..., 0])
        linear_points = torch.from_numpy(np.linspace(-1., 1., H_t)).type_as(base_grid).to(maps.device)
        base_grid[..., 1] = linear_points.view(1, -1, 1).repeat(1, 1, W_t).expand_as(base_grid[..., 0])
        # sampled map
        maps = F.grid_sample(maps, base_grid)
        """
        maps = F.adaptive_avg_pool2d(maps,(H_t,W_t))
    return maps


def cascade(out, x):
    if out.shape[1] > x.shape[1]:
        channels_in = x.shape[1]
        out = torch.cat([out[:, :channels_in] + x, out[:, channels_in:]], dim=1)
    elif out.shape[1] == x.shape[1]:
        out = out + x
    elif out.shape[1] < x.shape[1]:
        channels_out = out.shape[1]
        out = x[:, :channels_out] + out

    return out


class CascadeLayer(nn.Module):
    """
    This function continuously samples the feature maps
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, scale_factor=None, dropout_rate=0. ,norm_type='gn', norm_group=16,n_iter=2):
        super(CascadeLayer, self).__init__()
        self.scale_factor = scale_factor
        self.ConvBlock = DoubleConvBlock(in_planes, out_planes, kernel_size, int((kernel_size - 1) / 2),
                                         norm_type=norm_type)
        #self.ConvBlock=RecurrentConvBlockC(in_planes, out_planes, kernel_size, int((kernel_size - 1) / 2),dropout_rate=dropout_rate,norm_type=norm_type,n_iter=n_iter)
    def forward(self, x, output_size=None):

        out = self.ConvBlock(x)
        out = cascade(out, x)
        if self.scale_factor is not None:
            out = scaling(out, self.scale_factor)
        if output_size is not None:
            out = scaling(out, output_size)
        return out


class InvertedCascadeLayer(nn.Module):
    def __init__(self, in_planes, in_planes2, out_planes, kernel_size=3, padding=1, dropout_rate=0.,norm_type='gn', norm_group=16,n_iter=1):
        super(InvertedCascadeLayer, self).__init__()

        self.ConvBlock1 = SingleConvBlock(in_planes, out_planes, kernel_size, padding, norm_type=norm_type)
        self.ConvBlock2 = SingleConvBlock(in_planes2 + out_planes, out_planes, kernel_size, padding,
                                          norm_type=norm_type)
        #self.ConvBlock1 = RecurrentConvBlockC(in_planes, out_planes, kernel_size, int((kernel_size - 1) / 2),dropout_rate=dropout_rate,norm_type=norm_type,n_iter=n_iter)
        #self.ConvBlock2 = RecurrentConvBlockC(in_planes2 + out_planes, out_planes, kernel_size, int((kernel_size - 1) / 2),dropout_rate=dropout_rate,norm_type=norm_type,n_iter=n_iter)

    def forward(self, x, x2):
        x = scaling(x, output_size=x2.shape)
        out = self.ConvBlock1(x)
        out = cascade(out, x)
        out = self.ConvBlock2(torch.cat([out, x2], dim=1))
        maps = cascade(out, x)
        return maps


class ECN_Disp(nn.Module):
    def __init__(self, input_size, in_planes=3, init_planes=32, scale_factor=0.5, growth_rate=32, final_map_size=1,
                 alpha=10, beta=0.01, norm_type='gn'):
        super(ECN_Disp, self).__init__()
        self.scale_factor = scale_factor
        self.final_map_size = final_map_size
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()
        self.pred_planes = 1
        self.alpha = alpha
        self.beta = beta

        out_planes = init_planes
        output_size = input_size
        self.conv1 = nn.Conv2d(in_planes, init_planes, kernel_size=3, padding=1, stride=2, bias=False)
        output_size = math.floor(output_size / 2)
        while math.floor(output_size * scale_factor) >= final_map_size:
            new_out_planes = out_planes + growth_rate
            if len(self.encoding_layers) == 0:
                kernel_size = 3
            else:
                kernel_size = 3
            self.encoding_layers.append(
                CascadeLayer(in_planes=out_planes, out_planes=new_out_planes, kernel_size=kernel_size,
                             scale_factor=scale_factor, norm_type=norm_type))
            output_size = math.floor(output_size * scale_factor)
            out_planes = out_planes + growth_rate

        print(len(self.encoding_layers), ' encoding layers.')
        print(out_planes, ' encoded feature maps.')

        self.predict_maps = nn.ModuleList()

        in_planes2 = out_planes  # encoder planes

        planes = []
        for i in range(len(self.encoding_layers) + 1):
            if i == len(self.encoding_layers):
                in_planes2 = in_planes  # encoder planes
                new_out_planes = max(in_planes, self.pred_planes)
            else:
                in_planes2 = in_planes2 - growth_rate  # encoder planes
                new_out_planes = max(out_planes - growth_rate, self.pred_planes)
            self.decoding_layers.append(
                InvertedCascadeLayer(in_planes=out_planes, in_planes2=in_planes2, out_planes=new_out_planes,norm_type=norm_type))
            out_planes = new_out_planes
            planes.append(out_planes)

        planes.reverse()

        self.predicts=4
        self.predict_maps = nn.ModuleList()
        for i in range(self.predicts):
            self.predict_maps.append(SingleConvBlock(planes[i], 1, kernel_size=3, padding=1, norm_type=norm_type))

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):

        b, _, h, w = input.shape
        encode = [input]
        encode.append(self.conv1(encode[-1]))

        for i, layer in enumerate(self.encoding_layers):
            encode.append(layer(encode[-1]))

        decode = [encode[-1]]
        predicts = []

        for i, layer in enumerate(self.decoding_layers):
            out = layer(decode[-1], encode[-2 - i])
            decode.append(out)

            j = len(self.decoding_layers) - i
            if j <= self.predicts:
                pred = self.predict_maps[j - 1](decode[-1])
                predicts.append(pred)
                if len(predicts) > 1:
                    if len(predicts) > 1:
                        predicts[-1] = predicts[-1] + scaling(predicts[-2],output_size=predicts[-1].shape)  # residual learning
                    decode[-1] = torch.cat([decode[-1][:, :self.pred_planes] + predicts[-1], decode[-1][:, self.pred_planes:]],dim=1)  # residual learning

        predicts.reverse()

        #for i in range(self.predicts):
        #   print(i,predicts[i].min().item(),predicts[i].mean().item(),predicts[i].max().item())

        disp_predicts = [self.alpha * torch.sigmoid(predicts[i]) + self.beta for i in range(self.predicts)]

        if self.training:
            return disp_predicts
        else:
            return disp_predicts[0]


class ECN_Pose(nn.Module):
    def __init__(self, input_size, nb_ref_imgs=2, init_planes=16, scale_factor=0.5, growth_rate=16,
                 final_map_size=1, output_exp=False,output_exp2=False, output_pixel_pose=False, output_disp=False, alpha=10, beta=0.01,
                 norm_type='gn'):
        super(ECN_Pose, self).__init__()
        self.scale_factor = scale_factor
        self.final_map_size = final_map_size
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp
        self.output_exp2=output_exp2
        self.output_pixel_pose = output_pixel_pose
        self.output_disp = output_disp
        self.alpha = alpha
        self.beta = beta
        self.pred_planes = nb_ref_imgs * 8 + 1

        in_planes = (1 + nb_ref_imgs) *3

        out_planes = init_planes
        output_size = input_size
        self.conv1 = nn.Conv2d(in_planes, init_planes, kernel_size=3, padding=1, stride=2, bias=False)
        output_size = math.floor(output_size / 2)
        while math.floor(output_size * scale_factor) >= final_map_size:
            new_out_planes = out_planes + growth_rate
            if len(self.encoding_layers) == 0:
                kernel_size = 3
            else:
                kernel_size = 3
            self.encoding_layers.append(
                CascadeLayer(in_planes=out_planes, out_planes=new_out_planes, kernel_size=kernel_size,
                             scale_factor=scale_factor,norm_type=norm_type))
            output_size = math.floor(output_size * scale_factor)
            out_planes = out_planes + growth_rate

        print(len(self.encoding_layers), ' encoding layers.')
        print(out_planes, ' encoded feature maps.')

        self.pose_pred = SingleConvBlock(out_planes, 6 * self.nb_ref_imgs, kernel_size=1, padding=0,
                                         norm_type=norm_type)

        in_planes2 = out_planes  # encoder planes
        self.predicts = 4
        if self.output_exp or self.output_pixel_pose or self.output_disp or self.output_exp2:
            planes = []
            for i in range(len(self.encoding_layers) + 1):
                if i == len(self.encoding_layers):
                    in_planes2 = in_planes  # encoder planes
                    new_out_planes = max(in_planes, self.pred_planes)
                else:
                    in_planes2 = in_planes2 - growth_rate  # encoder planes
                    new_out_planes = max(out_planes - growth_rate, self.pred_planes)
                self.decoding_layers.append(
                    InvertedCascadeLayer(in_planes=out_planes, in_planes2=in_planes2, out_planes=new_out_planes,norm_type=norm_type))
                out_planes = new_out_planes
                planes.append(out_planes)

            planes.reverse()


            self.predict_maps = nn.ModuleList()
            for i in range(self.predicts):
                self.predict_maps.append(
                    SingleConvBlock(planes[i], self.pred_planes, kernel_size=3, padding=1, norm_type=norm_type))

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_image, ref_imgs):
        assert (len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)

        b, _, h, w = input.shape
        encode = [input]
        encode.append(self.conv1(encode[-1]))

        for i, layer in enumerate(self.encoding_layers):
            encode.append(layer(encode[-1]))

        out = encode[-1]

        pose = self.pose_pred(out)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        if self.output_exp or self.output_pixel_pose or self.output_disp or self.output_exp2:
            decode = [out]
            predicts = []

            for i, layer in enumerate(self.decoding_layers):
                out = layer(decode[-1], encode[-2 - i])
                decode.append(out)

                j = len(self.decoding_layers) - i
                if j <= self.predicts:
                    pred = self.predict_maps[j - 1](decode[-1])
                    predicts.append(pred)
                    if len(predicts) > 1:
                        predicts[-1] = predicts[-1] + scaling(predicts[-2],output_size=predicts[-1].shape)  # residual learning
                    decode[-1] = torch.cat([decode[-1][:, :self.pred_planes] + predicts[-1], decode[-1][:, self.pred_planes:]],dim=1)  # residual learning
            predicts.reverse()

            #for i in range(self.predicts):
            #    print(i,predicts[i].min().item(),predicts[i].mean().item(),predicts[i].max().item())

        if self.output_exp:
            exps = [torch.sigmoid(predicts[i][:, :self.nb_ref_imgs]) for i in range(self.predicts)]
        else:
            exps = [None for i in range(self.predicts)]


        if self.output_exp2:
            exps2 = [torch.sigmoid(predicts[i][:, self.nb_ref_imgs:2*self.nb_ref_imgs]) for i in range(self.predicts)]
        else:
            exps2 = [None for i in range(self.predicts)]

        if self.output_pixel_pose:
            pose_tmp = pose.view(pose.size(0), -1, 1, 1)
            pixel_poses = [0.01 * predicts[i][:, 2*self.nb_ref_imgs:8 * self.nb_ref_imgs] + pose_tmp for i in range(self.predicts)]
            #for i in range(self.predicts):
            #    b, c, h, w = exps[i].shape
            #    exp = exps[i].view(b, self.nb_ref_imgs, 1, h, w).repeat(1, 1, 6, 1, 1).view(b, self.nb_ref_imgs * 6, h,w)
            #    pixel_poses[i] = pixel_poses[i] * (1 - exp) + exp * pose_tmp
            #print('ego pose', pose_tmp[i].min().item(), pose_tmp[i].mean().item(), pose_tmp[i].max().item())
            #for i in range(self.predicts):
            #   print(i,(pixel_poses[i]- pose_tmp).min().item(),(pixel_poses[i]- pose_tmp).mean().item(),(pixel_poses[i]- pose_tmp).max().item())
        else:
            pixel_poses = [None for i in range(self.predicts)]

        if self.output_disp:
            disps = [self.alpha * torch.sigmoid(predicts[i][:, -1:]) + self.beta for i in range(self.predicts)]
        else:
            disps = [None for i in range(self.predicts)]

        if self.training:
            return exps,exps2, pixel_poses, disps, pose
        else:
            return exps[0],exps2[0], pixel_poses[0], disps[0], pose


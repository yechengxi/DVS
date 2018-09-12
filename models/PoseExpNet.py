import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class PoseExpNet(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False, output_pixel_pose=False,output_disp=False,alpha=10, beta=0.01):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp
        self.output_pixel_pose = output_pixel_pose
        self.output_disp = output_disp
        self.alpha = alpha
        self.beta = beta


        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        if self.output_exp or self.output_pixel_pose or self.output_disp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv(conv_planes[4],   upconv_planes[0])
            self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])

            if self.output_exp:
                self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
                self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
                self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
                self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)

            if self.output_pixel_pose:
                self.predict_pose4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs * 6, kernel_size=3, padding=1)
                self.predict_pose3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs * 6, kernel_size=3, padding=1)
                self.predict_pose2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs * 6, kernel_size=3, padding=1)
                self.predict_pose1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs * 6, kernel_size=3, padding=1)

            if self.output_disp:
                self.predict_disp4 = nn.Conv2d(upconv_planes[1], 1, kernel_size=3, padding=1)
                self.predict_disp3 = nn.Conv2d(upconv_planes[2], 1, kernel_size=3, padding=1)
                self.predict_disp2 = nn.Conv2d(upconv_planes[3], 1, kernel_size=3, padding=1)
                self.predict_disp1 = nn.Conv2d(upconv_planes[4], 1, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        if self.output_exp or self.output_pixel_pose or self.output_disp:
            out_upconv5 = self.upconv5(out_conv5  )[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 0:input.size(3)]

        if self.output_exp:
            exp_mask4 = torch.sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = torch.sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = torch.sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = torch.sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.output_pixel_pose:
            pose_tmp = pose.view(pose.size(0), -1, 1, 1)

            pixel_pose4 = 0.01 * self.predict_pose4(out_upconv4) + pose_tmp
            pixel_pose3 = 0.01 * self.predict_pose3(out_upconv3) + pose_tmp
            pixel_pose2 = 0.01 * self.predict_pose2(out_upconv2) + pose_tmp
            pixel_pose1 = 0.01 * self.predict_pose1(out_upconv1) + pose_tmp
        else:
            pixel_pose4 = None
            pixel_pose3 = None
            pixel_pose2 = None
            pixel_pose1 = None

        if self.output_disp:

            disp4 = self.alpha * torch.sigmoid(self.predict_disp4(out_upconv4)) + self.beta
            disp3 = self.alpha * torch.sigmoid(self.predict_disp3(out_upconv3)) + self.beta
            disp2 = self.alpha * torch.sigmoid(self.predict_disp2(out_upconv2)) + self.beta
            disp1 = self.alpha * torch.sigmoid(self.predict_disp1(out_upconv1)) + self.beta

        else:
            disp4 = None
            disp3 = None
            disp2 = None
            disp1 = None

        if self.training:
            exps=[exp_mask1, exp_mask2, exp_mask3, exp_mask4]
            pixel_poses=[pixel_pose1, pixel_pose2, pixel_pose3, pixel_pose4]
            disps=[disp1, disp2, disp3, disp4]

            return exps,pixel_poses,disps, pose
        else:
            return exp_mask1,pixel_pose1, disp1, pose





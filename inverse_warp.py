from __future__ import division
import torch
from torch.autograd import Variable


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()

    i_range = torch.arange(0, h, dtype=depth.dtype, device=depth.device, requires_grad=False).view(1, h, 1).expand(1, h, w)  # [1, H, W]
    j_range = torch.arange(0, w, dtype=depth.dtype, device=depth.device, requires_grad=False).view(1, 1, w).expand(1, h, w)  # [1, H, W]
    ones = torch.ones(1, h, w, dtype=depth.dtype, device=depth.device, requires_grad=False)
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).contiguous().view(b, 3, -1)  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b,h,w,2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rotMat = xmat.bmm(ymat).bmm(zmat)
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """

    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]
    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    b,c,h,w=img.shape
    i_range = torch.arange(0, h, dtype=depth.dtype, device=depth.device, requires_grad=False).view(1, h, 1).expand(1, h, w)  # [1, H, W]
    j_range = torch.arange(0, w, dtype=depth.dtype, device=depth.device, requires_grad=False).view(1, 1, w).expand(1, h, w)  # [1, H, W]
    i_range = 2 * i_range / (h - 1) - 1
    j_range=2*j_range/(w-1)-1

    pixel_coords = torch.stack((j_range, i_range), dim=1)  # [1, 3, H, W]


    flow=(src_pixel_coords-pixel_coords.permute(0,2,3,1))/2
    flow=torch.stack((flow[...,0]*(h-1),flow[...,1]*(w-1)),dim=3)
    #flow[flow>=2*(h-1)]=0
    #flow[flow >= 2*(w-1)]=0
    return projected_img,flow



def get_new_grid(depth, pose,intrinsics,intrinsics_inv):
    b, h, w = depth.size()

    if len(pose.shape)<=3:
        pose=pose.view(-1,1,1,6,1)
    else:
        pose=pose.permute(0,2,3,1).unsqueeze(4)

    i_range = torch.arange(0, h, dtype=depth.dtype, device=depth.device, requires_grad=False).view(1, h, 1).expand(1, h, w)  # [1, H, W]
    j_range = torch.arange(0, w, dtype=depth.dtype, device=depth.device, requires_grad=False).view(1, 1, w).expand(1, h, w)  # [1, H, W]
    ones = torch.ones(1, h, w, dtype=depth.dtype, device=depth.device, requires_grad=False)
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).contiguous().view(b, 3, -1)  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)

    x = cam_coords[:, 0]
    y = cam_coords[:, 1]


    L= depth.new_zeros(b, h, w, 2, 6, requires_grad=False)
    L[:, :, :, 0, 0] = -1./depth
    L[:, :, :, 0, 2] = x/depth
    L[:, :, :, 1, 1] = -1./depth
    L[:, :, :, 1, 2] = y/depth

    L[:, :, :, 0, 3] = x * y
    L[:, :, :, 0, 4] = - (1 + x ** 2)
    L[:, :, :, 0, 5] = y
    L[:, :, :, 1, 3] = (1 + y ** 2)
    L[:, :, :, 1, 4] = -x * y
    L[:, :, :, 1, 5] = -x

    velocity=torch.matmul(L, pose).squeeze(dim=4)

    #cam2pixel

    new_cam_coords=torch.cat([cam_coords[:,:2]+velocity.permute(0,3,1,2),cam_coords[:,2:]],dim=1).view(b, 3, -1)

    new_pixel_coords = intrinsics.bmm(new_cam_coords).view(b, 3, h, w)

    new_pixel_coords=new_pixel_coords[:,:2].permute(0,2,3,1)

    flow=(new_pixel_coords-pixel_coords[:,:2].permute(0,2,3,1))

    new_grid=torch.stack([2*new_pixel_coords[:, :, :, 0] / (w - 1) -1., 2*new_pixel_coords[:, :, :, 1] / (h - 1) -1.], dim=3)

    return new_grid, flow


def simple_inverse_warp(img, depth, pose, intrinsics,intrinsics_inv, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """

    new_grid,flow =get_new_grid(depth, pose, intrinsics, intrinsics_inv)

    #loss = epipolar_loss(flow, pose, intrinsics, intrinsics_inv)
    #print(loss.item())

    if padding_mode == 'zeros':
        mask=((new_grid>1)+(new_grid<-1)).detach()
        new_grid[mask] = 2

    projected_img = torch.nn.functional.grid_sample(img, new_grid, padding_mode=padding_mode)

    return projected_img,new_grid, flow



def compute_E(vec):
    """
    Convert 6DoF parameters to essential matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        Essential matrix -- [B, 3, 3]
    """
    b=vec.shape[0]
    t = vec[:, :3]  # [B, 3, 1]
    rot = vec[:,3:]
    rot_mat = euler2mat(rot)  # [B, 3, 3]

    #https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    t_x = vec.new_zeros(b, 3, 3, requires_grad=False)
    t_x[:, 0, 1] = -t[:,2]
    t_x[:, 1, 0] = t[:, 2]
    t_x[:, 0, 2] = t[:, 1]
    t_x[:, 2, 0] = -t[:, 1]
    t_x[:, 1, 2] = -t[:, 0]
    t_x[:, 2, 1] = t[:, 0]

    #https://en.wikipedia.org/wiki/Essential_matrix
    E_mat =  rot_mat.bmm(t_x) # [B, 3, 3]
    return E_mat


def epipolar_loss(flow,pose,intrinsics,intrinsics_inv):

    b,  h, w, c = flow.size()

    E=compute_E(pose)

    i_range = torch.arange(0, h, dtype=pose.dtype, device=pose.device, requires_grad=False).view(1, h, 1).expand(1, h, w)  # [1, H, W]
    j_range = torch.arange(0, w, dtype=pose.dtype, device=pose.device, requires_grad=False).view(1, 1, w).expand(1, h, w)  # [1, H, W]
    ones = torch.ones(1, h, w, dtype=pose.dtype, device=pose.device, requires_grad=False)
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).contiguous().view(b,3,-1)  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)

    new_cam_coords=torch.cat([cam_coords[:,:2]+flow.permute(0,3,1,2),cam_coords[:,2:]],dim=1)
    cam_coords=cam_coords.permute(0,2,3,1).unsqueeze(4)

    new_cam_coords=new_cam_coords.permute(0,2,3,1).unsqueeze(3)

    loss=new_cam_coords@E.view(b,1,1,3,3)@cam_coords#check the order --cam_coords@E@new_cam_coords

    return (loss.abs()).mean()


from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict



def save_path_formatter(args, parser):
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['sequence_length'] = 'seq'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr_scheduler'] = ''
    keys_with_prefix['norm_type'] = ''
    keys_with_prefix['norm_group'] = 'g'
    keys_with_prefix['optimizer'] = ''
    keys_with_prefix['lr'] = 'lr'
    keys_with_prefix['mask_loss_weight'] = 'm'
    keys_with_prefix['nls'] = 'nls'
    keys_with_prefix['smooth_loss_weight'] = 's'
    keys_with_prefix['flow_smooth_loss_weight'] = 'o'
    keys_with_prefix['pose_loss_weight'] = 'p'



    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp


def tensor2array(tensor, max_value=255, colormap='rainbow'):
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('3'):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (tensor.squeeze().numpy()*255./max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy().transpose(1, 2, 0)*0.5

    #for tensorboardx 1.4
    array=array.transpose(2,0,1)

    return array


def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar',epoch_id=None):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    if epoch_id:
        for (prefix, state) in zip(file_prefixes, states):
            torch.save(state, save_path/'{}_epoch{}_{}'.format(prefix,str(epoch_id),filename))
        if is_best:
            for prefix in file_prefixes:
                shutil.copyfile(save_path/'{}_epoch{}_{}'.format(prefix,str(epoch_id),filename), save_path/'{}_model_best.pth.tar'.format(prefix))

    else:
        for (prefix, state) in zip(file_prefixes, states):
            torch.save(state, save_path/'{}_{}'.format(prefix,filename))

        if is_best:
            for prefix in file_prefixes:
                shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))
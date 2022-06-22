import os
import copy
import collections
from time import time

import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology
import pickle


from seg.Network import unet, kiunet_org, unet_min


def get_dice(pred, seg):
    intersection = (pred * seg).sum()
    union = pred.sum() + seg.sum()
    return 2*intersection/union, 2*intersection, union


def read_ct(file):
    # 获取ct图像并进行数据预处理
    ct = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
    direction, origin, spacing = ct.GetDirection(), ct.GetOrigin(), ct.GetSpacing()
    ct = sitk.GetArrayFromImage(ct)
    origin_shape = ct.shape

    # # min max归一化
    # ct = ct.astype(np.float32)
    # ct = ct / max(upper, -lower)
    return ct, direction, origin, spacing, origin_shape


def read_seg(file):
    seg = sitk.ReadImage(os.path.join(test_seg_path, file.replace('image', 'label')), sitk.sitkUInt8)
    seg = sitk.GetArrayFromImage(seg)
    return seg


def test_volumn(ct, net):
    with torch.no_grad():
        ct_tensor = torch.FloatTensor(ct).cuda()
        ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
        outputs = net(ct_tensor)
        probability_map = np.squeeze(outputs.cpu().detach().numpy())
        del outputs
    return probability_map

if __name__ == '__main__':

    # 参数
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    net = unet(training=False).cuda()

    # 超参数
    threshold = 0.5

    module_path = f"..\\best_epoch_net200-0.130-0.165-0.609.pth"
    # test_ct_path = "..\\Kits\\Kits_seg\\trainset\\image"
    # test_seg_path = "..\\Kits\\Kits_seg\\trainset\\label"
    test_ct_path = "..\\Kits\\Kits_seg\\valset\\image"
    test_seg_path = "..\\Kits\\Kits_seg\\valset\\label"
    pred_path = f"..\\nets\\0502unet\\pred"
    dice_path = f"..\\nets\\0502unet"
    print(module_path)
    net.load_state_dict(torch.load(module_path))
    net.eval()

    # 为了计算dice_global定义的两个变量
    dice_intersection = 0.0
    dice_union = 0.0
    dice_for_file = {}
    for file_index, file in enumerate(os.listdir(test_ct_path)):
        start = time()
        ct, direction, origin, spacing, origin_shape = read_ct(file)

        probability_map = test_volumn(ct, net)

        pred_seg = np.zeros_like(probability_map)
        pred_seg[probability_map >= threshold] = 1

        seg = read_seg(file)

        pred_seg = pred_seg.astype(np.uint8)


        # 计算分割指标
        dice = get_dice(pred_seg, seg)
        dice_intersection += dice[1]
        dice_union += dice[2]
        dice = dice[0]

        # 将预测的结果保存为nii数据
        pred_seg = sitk.GetImageFromArray(pred_seg)

        pred_seg.SetDirection(direction)
        pred_seg.SetOrigin(origin)
        pred_seg.SetSpacing(spacing)

        # sitk.WriteImage(pred_seg, os.path.join(pred_path, file.replace('volume', 'pred')))

        speed = time() - start
        dice_for_file[file] = dice
        print(file, "dice: ", dice)
        if dice > 0.7:
            sitk.WriteImage(pred_seg, os.path.join(pred_path, file.replace('image', 'pred').replace(".nii", f"_{dice}.nii")))
        print(file_index, 'this case use {:.3f} s'.format(speed))
        print('-----------------------')

    print("dice global:", dice_intersection / dice_union)

import os
import random
import shutil

"""

测试脚本
"""

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

from seg.Network import kiunet_min, unet, kiunet_org
from seg.utils import Metirc

import seg.Parameter as para
# # import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
# 为了计算dice_global定义的两个变量
dice_intersection = 0.0
dice_union = 0.0

file_name = []  # 文件名称
time_pre_case = []  # 单例数据消耗时间

# 定义评价指标
liver_score = collections.OrderedDict()
liver_score['dice'] = []
liver_score['jacard'] = []
liver_score['voe'] = []
liver_score['fnr'] = []
liver_score['fpr'] = []
liver_score['assd'] = []
liver_score['rmsd'] = []
liver_score['msd'] = []

# 定义网络并加载参数
# net = torch.nn.DataParallel(unet(training=False)).cuda()
net = kiunet_org(training=False).cuda()
net.load_state_dict(torch.load(para.module_path))
net.eval()

files = os.listdir(para.test_ct_path)
for file in files:
    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(para.test_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    origin_shape = ct_array.shape

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # min max 归一化
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200

    # 滑动窗口取样预测
    with torch.no_grad():
        ct_tensor = torch.FloatTensor(ct_array).cuda()
        ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

        outputs = net(ct_tensor)

        probability_map = np.squeeze(outputs.cpu().detach().numpy())

        # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
        del outputs

        pred_seg = np.zeros_like(probability_map)
        pred_seg[probability_map >= (para.threshold)] = 1

    # 将金标准读入内存
    seg = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('image', 'label')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array != 2] = 0
    seg_array[seg_array == 2] = 1
    print(np.average(probability_map[seg_array==0]), np.average(probability_map[seg_array==1]))


# 对肝脏进行最大连通域提取,移除细小区域,并进行内部的空洞填充
pred_seg = pred_seg.astype(np.uint8)
liver_seg = copy.deepcopy(pred_seg)
liver_seg = measure.label(liver_seg, 4)
props = measure.regionprops(liver_seg)

max_area = 0
max_index = 0
for index, prop in enumerate(props, start=1):
    if prop.area > max_area:
        max_area = prop.area
        max_index = index

liver_seg[liver_seg != max_index] = 0
liver_seg[liver_seg == max_index] = 1

liver_seg = liver_seg.astype(np.bool)
morphology.remove_small_holes(liver_seg, para.maximum_hole, connectivity=2, in_place=True)
liver_seg = liver_seg.astype(np.uint8)

# 计算分割评价指标
liver_metric = Metirc(seg_array, liver_seg, ct.GetSpacing())
print('dice: ', liver_metric.get_dice_coefficient()[0])
print('jacard: ', liver_metric.get_jaccard_index())

liver_score['dice'].append(liver_metric.get_dice_coefficient()[0])
liver_score['jacard'].append(liver_metric.get_jaccard_index())
liver_score['voe'].append(liver_metric.get_VOE())
liver_score['fnr'].append(liver_metric.get_FNR())
liver_score['fpr'].append(liver_metric.get_FPR())
liver_score['assd'].append(liver_metric.get_ASSD())
liver_score['rmsd'].append(liver_metric.get_RMSD())
liver_score['msd'].append(liver_metric.get_MSD())



if __name__ == '__main__':
    root = './data'
    images = os.path.join(root, 'image')
    labels = os.path.join(root, 'label')

    cts = list(set([x[6:9] for x in os.listdir(images)]))

    random.seed(0)
    random.shuffle(cts)
    train_ct, test_ct = cts[:int(len(cts)*0.6)], cts[int(len(cts)*0.6):]

    for image in os.listdir(images):
        if image[6:9] in train_ct:
            shutil.copy(os.path.join(images, image), os.path.join(root, 'train', 'ct'))
        elif image[6:9] in test_ct:
            shutil.copy(os.path.join(images, image), os.path.join(root, 'test', 'ct'))
        else:
            print("error!!!")

    for label in os.listdir(labels):
        if label[6:9] in train_ct:
            shutil.copy(os.path.join(labels, label), os.path.join(root, 'train', 'seg'))
        elif label[6:9] in test_ct:
            shutil.copy(os.path.join(labels, label), os.path.join(root, 'test', 'seg'))
        else:
            print("error!!!")
"""

torch中的Dataset定义脚本
"""

import os
# import sys
# sys.path.append(os.path.split(sys.path[0])[0])

import random

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset as dataset
from torch.utils.data import Sampler

from seg import Parameter as para


class Dataset(dataset):
    def __init__(self, ct_dir, seg_dir):

        self.ct_list = os.listdir(ct_dir)
        # self.seg_list = list(map(lambda x: x.replace('volume', 'segmentation').replace('.nii', '.nii.gz'), self.ct_list))
        self.seg_list = list(map(lambda x: x.replace('image', 'label'), self.ct_list))
        self.ct_list = list(map(lambda x: os.path.join(ct_dir, x), self.ct_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

    def __getitem__(self, index):

        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
        # ct = sitk.GetArrayFromImage(sitk.ReadImage(imgs[0], sitk.sitkUInt8))

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # ct_array[ct_array > para.upper] = para.upper
        # ct_array[ct_array < para.lower] = para.lower
        #
        # # min max 归一化
        # ct_array = ct_array.astype(np.float32)
        # ct_array = ct_array / 200

        # seg_array[seg_array != 2] = 0
        # seg_array[seg_array == 2] = 1
        # # 在slice平面内随机选取48张slice
        # start_slice = random.randint(0, ct_array.shape[0] - para.size)
        # end_slice = start_slice + para.size - 1

        # z, x, y = ct_array.shape
        # ct_array = ct_array[start_slice:end_slice + 1, ::x//128, ::y//128]
        # # seg_array = seg_array[start_slice:end_slice + 1, ::x//128, ::y//128]
        # ct_array = ct_array[start_slice:end_slice + 1, , ]
        # seg_array = seg_array[start_slice:end_slice + 1, , ]

        # 处理完毕，将array转换为tensor
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return ct_array, seg_array

    def __len__(self):

        return len(self.ct_list)


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, replacement=True, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)
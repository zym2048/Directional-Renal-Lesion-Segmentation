root = "..."
save_path = "..."

import nibabel as nib
import os
import numpy as np
import skimage
import pickle
import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil
import random
import scipy.ndimage as ndimage


def load_file(ct_path, seg_path):
    pass


def main():
    imgRoot = os.path.join(root, "image")
    labelRoot = os.path.join(root, "label")
    volumns = {}
    # 获取所有影像及标签路径，两者在列表中一一对应
    # ..\\case_xxx_0000.nii    use [-12:-9] to get xxx
    imagePaths = [os.path.join(imgRoot, x) for x in os.listdir(imgRoot)]
    labelPaths = [[os.path.join(labelRoot, "label_" + x[-12:-9] + ".nii"), x[-12:-9]]
                  if os.path.isfile(os.path.join(labelRoot, "label_" + x[-12:-9] + ".nii"))
                  else [os.path.join(labelRoot, "case_" + x[-12:-9] + "_0000.nii"), x[-12:-9]] for x in imagePaths]

    node_num = 1
    for i in range(len(imagePaths)):
        ct_path = imagePaths[i]
        seg_path = labelPaths[i][0]
        id = ct_path[-12:-9]

        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_path, sitk.sitkUInt8))
        seg_array[seg_array != 2] = 0
        seg_array[seg_array == 2] = 1

        # 重采样，调整z轴spacing
        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / 1, 1, 1), order=3)
        seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / 1, 1, 1), order=0)

        # 将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array)

        new_seg = sitk.GetImageFromArray(seg_array)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], 1))

        sitk.WriteImage(new_seg, os.path.join(save_path, "label", f"label_{id}.nii"))

        # 获取病灶
        label = skimage.measure.label(seg_array, connectivity=3)
        bboxs = [region.bbox for region in skimage.measure.regionprops(label)]
        if len(bboxs) == 0:
            continue

        bbox_dict = {}
        z_depth, w, h = ct_array.shape
        for bbox in bboxs:
            node_zmin, node_xmin, node_ymin, node_zmax, node_xmax, node_ymax = bbox
            node_width, node_height = node_xmax - node_xmin, node_ymax - node_ymin
            xy_step = max(node_width, node_height) // 64 + 1
            z_step = (node_zmax - node_zmin) // 64 + 1

            x_min = (node_xmin + node_xmax) // 2 - xy_step * 32
            x_max = (node_xmin + node_xmax) // 2 + xy_step * 32
            y_min = (node_ymin + node_ymax) // 2 - xy_step * 32
            y_max = (node_ymin + node_ymax) // 2 + xy_step * 32
            z_min = (node_zmin + node_zmax) // 2 - z_step * 32
            z_max = (node_zmin + node_zmax) // 2 + z_step * 32

            # 假如体积框超过了边界，则让其向周围平移
            if x_min < 0:
                x_max, x_min = x_max + (-x_min), 0
            if x_max > w:
                x_min, x_max = x_min + (w - x_max), w
            if y_min < 0:
                y_max, y_min = y_max + (-y_min), 0
            if y_max > h:
                y_min, y_max = y_min + (h - y_max), h
            if z_min < 0:
                z_max, z_min = z_max + (-z_min), 0
            if z_max > z_depth:
                z_min, z_max = z_min + (z_depth - z_max), z_depth

            x_min, y_min, z_min = max(x_min, 0), max(y_min, 0), max(z_min, 0)
            x_max, y_max, z_max = min(x_max, w), min(y_max, h), min(z_max, z_depth)
            ct_array_tmp = ct_array[z_min:z_max:z_step, x_min:x_max:xy_step, y_min:y_max:xy_step]
            seg_array_tmp = seg_array[z_min:z_max:z_step, x_min:x_max:xy_step, y_min:y_max:xy_step]

            if ct_array_tmp.shape != (64, 64, 64):
                print("WRONG:", "image_{}_{:0>4d}.nii".format(id, node_num))
            new_ct = sitk.GetImageFromArray(ct_array_tmp)

            new_ct.SetDirection(ct.GetDirection())
            new_ct.SetOrigin(ct.GetOrigin())
            new_ct.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], 1))

            new_seg = sitk.GetImageFromArray(seg_array_tmp)

            new_seg.SetDirection(ct.GetDirection())
            new_seg.SetOrigin(ct.GetOrigin())
            new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], 1))

            sitk.WriteImage(new_ct, os.path.join(save_path, "ct", "image_{}_{:0>4d}.nii".format(id, node_num)))
            sitk.WriteImage(new_seg, os.path.join(save_path, "seg","label_{}_{:0>4d}.nii".format(id, node_num)))
            bbox_dict["image_{}_{:0>4d}.nii".format(id, node_num)] = x_min, y_min, z_min, x_max, y_max, z_max, xy_step, z_step
            node_num += 1

        volumns[f"image_{id}"] = {
            "bbox": bbox_dict,
            "spacing": ct.GetSpacing()
        }

        print(f"Finish {imagePaths[i]}")


    with open(os.path.join(save_path, "volumes.pkl"), 'wb') as f:
        pickle.dump(volumns, f, pickle.HIGHEST_PROTOCOL)
    print("Finish ALL")


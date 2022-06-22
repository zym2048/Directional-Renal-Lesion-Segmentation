import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import os
from parameter import Para
import skimage
import pickle
import numpy as np


def change_direction(ct):
    spacing, origin = ct.GetSpacing(), ct.GetOrigin()
    new_ct = sitk.GetImageFromArray(np.flipud(sitk.GetArrayFromImage(ct).transpose((2, 1, 0))))
    new_ct.SetSpacing((spacing[1], spacing[2], spacing[0]))
    new_ct.SetOrigin((origin[1], origin[2], origin[0]))
    return new_ct


def change_seg_array(seg_array):
    seg_array[seg_array != 2] = 0
    seg_array[seg_array > 0] = 1
    return seg_array


def get_HUs(args, image_path, label_path, image, label):
    HU = {}
    seg_pixel = []
    img_HU = []
    for i in range(len(image)):
        ct = sitk.ReadImage(os.path.join(image_path, image[i]), sitk.sitkInt16)
        seg = sitk.ReadImage(os.path.join(label_path, label[i]), sitk.sitkUInt8)

        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        seg_array = change_seg_array(seg_array)

        ct_array = ct_array[seg_array == 1]
        if len(ct_array > 0):
            seg_pixel.extend(ct_array)
            img_HU.append([image[i], ct_array.min(), ct_array.max(), ct_array.mean()])
        print("finish", image[i], f":{i+1}/{len(image)}")
    HU['pixel'] = seg_pixel
    HU['img_HU'] = img_HU
    with open(os.path.join(args['root_path'], "HU_analysis.pkl"), "wb") as f:
        pickle.dump(HU, f, pickle.HIGHEST_PROTOCOL)


def HU_analysis(args):
    with open(os.path.join(args['root_path'], "HU_analysis.pkl"), "rb") as f:
        HU = pickle.load(f)
    pixel, img_HU = HU['pixel'], HU['img_HU']

    pixel = [i for i in pixel if i < 600]
    pixel = pd.DataFrame(pixel)
    pixel.hist()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('病灶区域CT值分布直方图')
    plt.ylabel('像素数量')
    plt.xlabel('CT值')
    # plt.savefig(os.path.join(pics_root, "病灶区域CT值分布直方图.png"))
    plt.show()


def main(args, image_path, label_path, image, label):
    image_dict = {}
    for i in range(len(image)):
        ct = sitk.ReadImage(os.path.join(image_path, image[i]), sitk.sitkInt16)
        seg = sitk.ReadImage(os.path.join(label_path, label[i]), sitk.sitkUInt8)
        id = image[i].split('_')[1]

        change_label = False
        if seg.GetDirection() != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            seg, ct = change_direction(seg), change_direction(ct)
            change_label = True
        # if ct.GetDirection() != (-0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0) or ct.GetSpacing()[1] != ct.GetSpacing()[2]:
        #     print("WRONG: ", image[i])
        #     continue

        seg, ct = change_direction(seg), change_direction(ct)
        change_label = True


        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        seg_array = change_seg_array(seg_array)

        if np.sum(seg_array) == 0:
            print(f"{image[i]} has no nidus!")
            continue

        # ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / 1, ct.GetSpacing()[1] / 1, ct.GetSpacing()[0] / 1), order=3)
        # seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / 1, ct.GetSpacing()[1] / 1, ct.GetSpacing()[0] / 1), order=0)
        if change_label:
            ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / 1, ct.GetSpacing()[1] / 1, ct.GetSpacing()[0] / 1), order=3)
            seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / 1, ct.GetSpacing()[1] / 1, ct.GetSpacing()[0] / 1), order=0)
        else:
            # 重采样，调整z轴spacing
            ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / 1, 1, 1), order=3)
            seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / 1, 1, 1), order=0)

        # 将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array)
        new_seg = sitk.GetImageFromArray(seg_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], 1))

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], 1))

        sitk.WriteImage(new_ct, os.path.join(args['data_path'], "image", f"case_{id}_0000.nii"))
        sitk.WriteImage(new_seg, os.path.join(args['data_path'], "label", f"label_{id}.nii"))

        # 获取病灶 包括每张CT中的病灶数量，每个病灶的大小、中心点位置、三维
        areas = skimage.measure.label(seg_array, connectivity=3)
        areas = [[region.bbox, region.area] for region in skimage.measure.regionprops(areas)]
        if len(areas) == 0:
            continue
        image_dict[id] = areas
        print(f"Finish {image[i]}. {i+1}/{len(image)}")
    with open(os.path.join(args['root_path'], "nidus.pkl"), "wb") as f:
        pickle.dump(image_dict, f, pickle.HIGHEST_PROTOCOL)
    print("Finish ALL")


if __name__ == "__main__":
    args = Para().get_para()

    image_path = os.path.join(args['origin_data_path'], "image")
    label_path = os.path.join(args['origin_data_path'], "label")

    image = os.listdir(image_path)
    label = [f"label_{id.split('_')[1]}.nii" for id in image]

    # # 需要先使用get_HUs，再使用HU_analysis
    # get_HUs(args, image_path, label_path, image, label)
    # HU_analysis(args)

    main(args, image_path, label_path, image, label)


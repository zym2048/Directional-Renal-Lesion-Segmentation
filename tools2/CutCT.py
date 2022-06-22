import nibabel as nib
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import scipy.ndimage as ndimage
import SimpleITK as sitk

HU_LOW, HU_HIGH = -200, 200
CLASS_LABEL = 2
CTROOT = "..\\CT\\Kits_data"
SaveROOT = "..\\dataset"

def createXMLlabel(savedir, folder, filename, bbox, width, height, z_depth,
                   classname="nidus", depth='1', truncated='0', difficult='0'):
    # 创建根节点
    root = ET.Element("annotation")

    # 创建子节点
    # 将子节点数据添加到根节点
    folder_node = ET.Element("folder")
    folder_node.text = folder
    root.append(folder_node)

    filename_node = ET.Element("filename")
    filename_node.text = filename
    root.append(filename_node)

    size_node = ET.Element("size")
    width_node = ET.SubElement(size_node, "width")
    height_node = ET.SubElement(size_node, "height")
    depth_node = ET.SubElement(size_node, "depth")
    z_node = ET.SubElement(size_node, "z_depth")
    width_node.text = str(width)
    height_node.text = str(height)
    depth_node.text = str(height)
    z_node.text = str(z_depth)
    root.append(size_node)

    for i in range(len(bbox)):
        newEle = ET.Element("object")
        name = ET.Element("name")
        name.text = classname
        newEle.append(name)


        boundingbox = ET.Element("bndbox")
        xmin = ET.SubElement(boundingbox, "xmin")
        xmax = ET.SubElement(boundingbox, "xmax")
        ymin = ET.SubElement(boundingbox, "ymin")
        ymax = ET.SubElement(boundingbox, "ymax")
        xmin.text = str(bbox[i][1])
        ymin.text = str(bbox[i][0])
        xmax.text = str(bbox[i][3])
        ymax.text = str(bbox[i][2])
        newEle.append(boundingbox)

        trunc = ET.Element("truncated")
        trunc.text = truncated
        newEle.append(trunc)
        dif = ET.Element("difficult")
        dif.text = difficult
        newEle.append(dif)

        root.append(newEle)

    segmented_node = ET.Element("segmented")
    segmented_node.text = "0"
    root.append(segmented_node)

    ImageID = filename.split('.')[0]
    # 创建elementtree对象，写入文件
    tree = ET.ElementTree(root)
    if not os.path.exists(os.path.join(savedir, folder)):
        os.makedirs(os.path.join(savedir, folder))
    tree.write(os.path.join(savedir, folder, ImageID) + ".xml")


import skimage
# https://www.cnblogs.com/smartweed/p/12153744.html
# 输入图片，生成检测框列表， [[Y_min1, X_min1, Y_max1, X_max1], [Y_min2, X_min2, Y_max2, X_max2], ...]
def build_bounding_boxes(image, label):
    # mask.shape = [image.shape[0], image.shape[1], classnum]
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[image==label] = 1
    # # 删掉小于10像素的目标
    # mask_without_small = skimage.morphology.remove_small_objects(mask, min_size=5, connectivity=1)
    # 连通域标记
    label_image = skimage.measure.label(mask, connectivity=2)
    # 统计object个数
    boundingbox = [region.bbox for region in skimage.measure.regionprops(label_image) if region.area > 5]
    object_num = len(boundingbox)
    return object_num, boundingbox


import matplotlib.pyplot as plt
import matplotlib.patches as patches
def draw_bounding_boxes(image, boundingbox):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box in boundingbox:
        x, y, w, h = box[1], box[0], box[3]-box[1], box[2]-box[0]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    # plt.axis("off")
    # plt.savefig("slice_example2.png", bbox_inches='tight')


# def createXMLlabel(savedir, folder, filename, bbox, width, height,
#                    classname="nidus", depth='3', truncated='0', difficult='0')
if __name__ == '__main__':


    imgRoot = os.path.join(CTROOT, "image")
    labelRoot = os.path.join(CTROOT, "label")

    # 获取所有影像及标签路径，两者在列表中一一对应
    # ..\\case_xxx_0000.nii    use [-12:-9] to get xxx
    imagePaths = [os.path.join(imgRoot, x) for x in os.listdir(imgRoot)]
    labelPaths = [os.path.join(labelRoot, "label_" + x[-12:-9] + ".nii")
                  if os.path.isfile(os.path.join(labelRoot, "label_" + x[-12:-9] + ".nii"))
                  else os.path.join(labelRoot, "case_" + x[-12:-9] + "_0000.nii") for x in imagePaths]

    print("GET Paths.")
    for i in range(len(imagePaths)):
        print(f"Dealing with picture {i}:")
        image = np.array(nib.load(imagePaths[i]).dataobj)
        label = np.array(nib.load(labelPaths[i]).dataobj)

        # ct = sitk.ReadImage(os.path.join(imagePaths[i]), sitk.sitkInt16)
        # image = sitk.GetArrayFromImage(ct)
        #
        # seg = sitk.ReadImage(os.path.join(labelPaths[i]), sitk.sitkUInt8)
        # label = sitk.GetArrayFromImage(seg)

        image[image < HU_LOW] = HU_LOW
        image[image > HU_HIGH] = HU_HIGH
        image = (image - HU_LOW) / (HU_HIGH - HU_LOW) * 255
        x, y, z = label.shape

        # 对CT数据在横断面上进行降采样,并进行重采样,将所有数据的z轴的spacing调整到1mm
        # 暂时不管这个了
        # image = ndimage.zoom(image, (ct.GetSpacing()[-1],0.5, 0.5), order=3)
        # label = ndimage.zoom(label, (seg.GetSpacing()[-1], 1, 1), order=0)

        # 在z轴方向上，计算有像素的连续区域的起始坐标和长度
        # 输出结构为列表：[[ind1, length1], [ind2, length2]...]
        def region_cut_by_z(ct, class_label):
            L = [np.sum(ct[:, :, i] == class_label) for i in range(z)]
            length = len(L)
            count, regions = 0, []
            for i in range(length):
                if L[i] > 0:
                    count += 1
                if count > 0 and (L[i] == 0 or i == length - 1):
                    regions.append([i - count, count])
                    count = 0
            return regions


        regions = region_cut_by_z(ct=label, class_label=CLASS_LABEL)

        # 每个区域采样不超过10张, 采样间隔为 num//11
        idxs = []
        for region in regions:
            idxs += [x + region[0] for x in list(range(region[1])[::region[1] // 11 + 1])]

        for idx in idxs:
            image_slice = image[:, :, idx]
            label_slice = label[:, :, idx]

            object_num, boundingbox = build_bounding_boxes(label_slice, CLASS_LABEL)
            filename = imagePaths[i][-12:-9] + "_{:0>4d}".format(idx) + ".jpg"
            if not os.path.exists(os.path.join(SaveROOT, "images")):
                os.makedirs(os.path.join(SaveROOT, "images"))
            cv2.imwrite(os.path.join(SaveROOT, "images", filename), image_slice)
            createXMLlabel(SaveROOT, "KITS", filename=filename, bbox=boundingbox, width=x, height=y, z_depth=z)
            print(f"finined pic{i} : {filename}")
        print()
    print("Finish all!")
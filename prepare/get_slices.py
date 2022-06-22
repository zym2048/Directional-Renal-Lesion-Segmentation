import skimage

from parameter import Para

import nibabel as nib
import numpy as np
import cv2

import os
import json


def get_z_ids(label, z, is_train: bool):
    if is_train:
        # 在z轴方向上，计算有像素的连续区域的起始坐标和长度
        # 输出结构为列表：[[ind1, length1], [ind2, length2]...]
        def region_cut_by_z(ct):
            L = [np.sum(ct[:, :, i]) for i in range(z)]
            length = len(L)
            count, regions = 0, []
            for i in range(length):
                if L[i] > 0:
                    count += 1
                if count > 0 and (L[i] == 0 or i == length - 1):
                    regions.append([i - count, count])
                    count = 0
            return regions

        regions = region_cut_by_z(ct=label)

        # 每个区域采样不超过10张, 采样间隔为 num//11
        idxs = []
        for region in regions:
            idxs += [t + region[0] for t in list(range(region[1])[::region[1] // 11 + 1])]
    else:
        idxs = [i for i in range(z) if np.sum(label[:, :, i]) > 0]

    # 去除首张和末尾张
    if 0 in idxs:
        idxs.remove(0)
    if z-1 in idxs:
        idxs.remove(z-1)

    return idxs


# https://www.cnblogs.com/smartweed/p/12153744.html
# 输入图片，生成检测框列表， [[Y_min1, X_min1, Y_max1, X_max1], [Y_min2, X_min2, Y_max2, X_max2], ...]
def build_bounding_boxes(label):
    label_image = skimage.measure.label(label, connectivity=2)
    # 统计object个数
    boundingbox = [region.bbox for region in skimage.measure.regionprops(label_image)]
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


def deal_image(args, image):
    HU_HIGH, HU_LOW = args['HU_HIGH'], args['HU_LOW']
    image[image < HU_LOW] = HU_LOW
    image[image > HU_HIGH] = HU_HIGH
    image = (image - HU_LOW) / (HU_HIGH - HU_LOW) * 255
    return image


def get_coco(args, image_path, label_path, image_list, label_list, type):
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": [{"supercategory": "none", "id": 0, "name": "nidus"}]}
    bnd_id = 1

    for i in range(len(image_list)):
        image = np.array(nib.load(os.path.join(image_path, image_list[i])).dataobj)
        label = np.array(nib.load(os.path.join(label_path, label_list[i])).dataobj)
        image = deal_image(args, image)
        x, y, z = label.shape

        if type == "train":
            idxs = get_z_ids(label, z, is_train=True)
        else:
            idxs = get_z_ids(label, z, is_train=False)


        for idx in idxs:
            object_num, boundingbox = build_bounding_boxes(label[:, :, idx])
            filename = image_list[i].split("_")[1] + "_{:0>4d}".format(idx) + ".jpg"

            cv2.imwrite(os.path.join(args['coco_path'], f"{type}2017", filename), image[:, :, idx-1:idx+2])

            image_id = int(filename.split(".")[0])
            image_dict = {"file_name": filename, "height": x, "width": y, "id": image_id}
            json_dict['images'].append(image_dict)
            for box in boundingbox:
                Y_min, X_min, Y_max, X_max = box
                o_width = abs(X_max - X_min)
                o_height = abs(Y_max - Y_min)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                       'bbox': [X_min, Y_min, o_width, o_height],
                       'category_id': 0, 'id': bnd_id, 'ignore': 0, 'segmentation': []}
                json_dict['annotations'].append(ann)
                bnd_id += 1

        print(f"Finish {image_list[i]}.   {i+1}/{len(image_list)}")

    json_file = os.path.join(args['coco_path'], "annotations", f"{type}.json")
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print(f"Finish {type}... \n\n")


def change_nidus_type(args, type):
    json_file = os.path.join(args['coco_path'], "annotations1", f"{type}.json")
    with open(json_file, 'r')as fp:
        json_dict = json.load(fp)
    annotations = []
    # json_dict["categories"] = [{"supercategory": "none", "id": 0, "name": "tiny"}, # <1000
    #                            {"supercategory": "none", "id": 1, "name": "middle"},
    #                            {"supercategory": "none", "id": 2, "name": "huge"}] # >=100,000
    json_dict["categories"] = [{"supercategory": "none", "id": 0, "name": "tiny"}, # <1000
                               {"supercategory": "none", "id": 1, "name": "huge"}] # >=100,000
    # for anno in json_dict["annotations"]:
    #     if anno['area'] < 1000:
    #         anno['category_id'] = 0
    #     elif anno['area'] <= 100000:
    #         anno['category_id'] = 1
    #     else:
    #         anno['category_id'] = 2
    for anno in json_dict["annotations"]:
        if anno['area'] <= 100000:
            anno['category_id'] = 0
        else:
            anno['category_id'] = 1
        annotations.append(anno)
    json_dict["annotations"] = annotations

    json_file = os.path.join(args['coco_path'], "annotations", f"{type}.json")
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    args = Para().get_para()

    image_path = os.path.join(args['data_path'], "image")
    label_path = os.path.join(args['data_path'], "label")

    image = os.listdir(image_path)
    image_train = [img for img in image if (img.split('_')[1] in args['nidus_train'])]
    image_val = [img for img in image if (img.split('_')[1] in args['nidus_val'])]
    image_test = [img for img in image if (img.split('_')[1] in args['nidus_test'])]

    label_train = [f"label_{id.split('_')[1]}.nii" for id in image_train]
    label_val = [f"label_{id.split('_')[1]}.nii" for id in image_val]
    label_test = [f"label_{id.split('_')[1]}.nii" for id in image_test]

    get_coco(args, image_path, label_path, image_val, label_val, "val")
    get_coco(args, image_path, label_path, image_test, label_test, "test")
    get_coco(args, image_path, label_path, image_train, label_train, "train")
    # #
    # all_areas = []
    # count = [0, 0, 0]
    # for type in ["test", "val", "train"]:
    #     json_file = os.path.join(args['coco_path'], "annotations1", f"{type}.json")
    #     with open(json_file, 'r')as fp:
    #         json_dict = json.load(fp)
    #     annotations = []
    #     # json_dict["categories"] = [{"supercategory": "none", "id": 0, "name": "tiny"},  # <500
    #     #                            {"supercategory": "none", "id": 1, "name": "middle"},
    #     #                            {"supercategory": "none", "id": 2, "name": "huge"}]  # >=5,000
    #     json_dict["categories"] = [{"supercategory": "none", "id": 0, "name": "tiny"},  # <1000
    #                                {"supercategory": "none", "id": 1, "name": "huge"}]  # >=100,000
    #     # for anno in json_dict["annotations"]:
    #     #     if anno['area'] < 500:
    #     #         anno['category_id'] = 0
    #     #         count[0] += 1
    #     #     elif anno['area'] <= 5000:
    #     #         anno['category_id'] = 1
    #     #         count[1] += 1
    #     #     else:
    #     #         anno['category_id'] = 2
    #     #         count[2] += 1
    #     #     annotations.append(anno)
    #     #     all_areas.append(anno['area'])
    #     for anno in json_dict["annotations"]:
    #         if anno['area'] <= 5000:
    #             anno['category_id'] = 0
    #             count[1] += 1
    #         else:
    #             anno['category_id'] = 1
    #             count[2] += 1
    #         annotations.append(anno)
    #         all_areas.append(anno['area'])
    #     json_dict["annotations"] = annotations
    #
    #     json_file = os.path.join(args['coco_path'], "annotations", f"{type}.json")
    #     json_fp = open(json_file, 'w')
    #     json_str = json.dumps(json_dict)
    #     json_fp.write(json_str)
    #     json_fp.close()
    #     print(f"finish {type}:    tiny:{count[0]}, middle:{count[1]}, huge:{count[2]}")

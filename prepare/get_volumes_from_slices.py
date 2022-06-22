import os
import numpy as np
import skimage
import pickle
import SimpleITK as sitk
import scipy.ndimage as ndimage
from parameter import Para


# bbox = [x_middle, y_middle, z_middle]
def get_volume_by_bbox(image_id, bbox, xy_step, z_step, z_depth, w, h, ct_array, seg_array, ct, node_num, mode="trainset"):
    x_min = bbox[0] - xy_step * 32
    x_max = bbox[0] + xy_step * 32
    y_min = bbox[1] - xy_step * 32
    y_max = bbox[1] + xy_step * 32
    z_min = bbox[2] - z_step * 32
    z_max = bbox[2] + z_step * 32
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
    # seg_array_tmp = seg_array[z_min:z_max:z_step, x_min:x_max:xy_step, y_min:y_max:xy_step]

    if ct_array_tmp.shape != (64, 64, 64):
        print("WRONG:", "image_{}_{:0>4d}.nii    xy_step:{},   z_step:{}".format(image_id, node_num, xy_step, z_step))
        return False

    new_ct = sitk.GetImageFromArray(ct_array_tmp)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], 1))

    # new_seg = sitk.GetImageFromArray(seg_array_tmp)
    #
    # new_seg.SetDirection(ct.GetDirection())
    # new_seg.SetOrigin(ct.GetOrigin())
    # new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], 1))

    sitk.WriteImage(new_ct, os.path.join(args['seg_path'], mode, "image", "image_{}_{:0>4d}.nii".format(image_id, node_num)))
    # sitk.WriteImage(new_seg, os.path.join(args['seg_path'], mode, "label", "label_{}_{:0>4d}.nii".format(image_id, node_num)))

    return x_min, y_min, z_min, x_max, y_max, z_max, xy_step, z_step


def get_training_set(args, image_path, label_path):
    with open(os.path.join(args['root_path'], "nidus.pkl"), "rb") as f:
        nidus = pickle.load(f)
    volumes = {}
    volumes_info = {}
    for image_id in nidus:
        if image_id not in args['nidus_train']:
            continue

        ct = sitk.ReadImage(os.path.join(image_path, f"case_{image_id}_0000.nii"), sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_path, f"label_{image_id}.nii"),
                                                          sitk.sitkUInt8))
        bbox_dict = {}
        z_depth, w, h = ct_array.shape
        node_num = 1
        for node in nidus[image_id]:
            node_zmin, node_xmin, node_ymin, node_zmax, node_xmax, node_ymax = node[0]
            x_middle = int((node_xmax+node_xmin)//2)
            y_middle = int((node_ymax+node_ymin)//2)
            z_middle = int((node_zmax+node_zmin)//2)
            bbox_change = [[x_middle, y_middle, z_middle, 1, 1],
                           [x_middle+15, y_middle+15, z_middle+15, 1, 1],
                           [x_middle+15, y_middle+15, z_middle-15, 1, 1],
                           [x_middle+15, y_middle-15, z_middle+15, 1, 1],
                           [x_middle+15, y_middle-15, z_middle-15, 1, 1],
                           [x_middle-15, y_middle+15, z_middle+15, 1, 1],
                           [x_middle-15, y_middle+15, z_middle-15, 1, 1],
                           [x_middle-15, y_middle-15, z_middle+15, 1, 1],
                           [x_middle-15, y_middle-15, z_middle-15, 1, 1],
                           [x_middle, y_middle, z_middle, 2, 2],
                           [x_middle, y_middle, z_middle, 3, 3],
                           [x_middle, y_middle, z_middle, 1, 2],
                           [x_middle, y_middle, z_middle, 2, 1],
                           [x_middle, y_middle, z_middle, 2, 3],
                           [x_middle, y_middle, z_middle, 3, 2],
                           ]
            for i in range(len(bbox_change)):
                location = get_volume_by_bbox(image_id, bbox_change[i][:3], bbox_change[i][3], bbox_change[i][4],
                                       z_depth, w, h, ct_array, seg_array, ct, node_num+i)
                if not location:
                    continue
                bbox_dict["image_{}_{:0>4d}.nii".format(image_id, node_num+i)] = location
                volumes_info["image_{}_{:0>4d}.nii".format(image_id, node_num+i)] = {
                    "location": location,
                    "image": image_id,
                    "nudis_areas": node[1],
                    "offset": [bbox_change[i][0]-bbox_change[0][0], bbox_change[i][1]-bbox_change[0][1],
                               bbox_change[i][2]-bbox_change[0][2], bbox_change[i][3], bbox_change[i][4]]
                }
            node_num += len(bbox_change)
        volumes[f"image_{image_id}"] = bbox_dict
        print(f"finish_{image_id}")
    # with open(os.path.join(os.path.join(args['seg_path'], "trainset", "volumes.pkl")), 'wb') as f:
    #     pickle.dump(volumes, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(os.path.join(args['seg_path'], "trainset", "volumes.pkl")), 'wb') as f:
        pickle.dump(volumes_info, f, pickle.HIGHEST_PROTOCOL)
    print("Finish trainset_volumes\n\n")


def deal_single_ct(args, image_path, label_path, image, label, bboxes, mode):
    id = image.split("_")[1]

    ct = sitk.ReadImage(os.path.join(image_path, image), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_path, label), sitk.sitkUInt8))

    bbox_dict = {}
    z_depth, w, h = ct_array.shape
    node_num = 1
    for bbox in bboxes:
        bbox_dict["image_{}_{:0>4d}.nii".format(id, node_num)] = \
            get_volume_by_bbox(id, bbox, 1, 1, z_depth, w, h, ct_array, seg_array, ct, node_num, mode)
        bbox_dict["image_{}_{:0>4d}.nii".format(id, node_num+1)] = \
            get_volume_by_bbox(id, bbox, 2, 2, z_depth, w, h, ct_array, seg_array, ct, node_num+1, mode)
        bbox_dict["image_{}_{:0>4d}.nii".format(id, node_num+2)] = \
            get_volume_by_bbox(id, bbox, 3, 3, z_depth, w, h, ct_array, seg_array, ct, node_num+2, mode)
        node_num += 3
    print("finish", image)
    return bbox_dict


def reduce_volumes(image_list: list):
    def volume_in_list(l, v):
        for i in l:
            if (i[0]-v[0]) < 10 and (i[1]-v[1]) < 10 and (i[2]-v[2]) < 10:
                return True
        return False
    new_list = []
    for i in image_list:
        if not volume_in_list(new_list, i):
            new_list.append(i)
    return new_list


def get_volumes(args, image_path, label_path, image_list, label_list, type):

    with open(os.path.join(args['yolox_output_path'], f"{type}_bboxes.pkl"), "rb") as f:
        bboxes = pickle.load(f)

    image_boxes = dict.fromkeys(list(set(x[:3] for x in bboxes.keys())))
    for key in image_boxes:
        image_boxes[key] = []
    # x0 = int(box[0])
    # y0 = int(box[1])
    # x1 = int(box[2])
    # y1 = int(box[3])
    for key in bboxes:
        for bbox in bboxes[key]:
            image_boxes[key[:3]].append([int((bbox[0]+bbox[2])//2), int((bbox[1]+bbox[3])//2), int(key[-4:])])

    # for key in image_boxes:
    #     image_boxes[key] = reduce_volumes(image_boxes[key])

    volumns = {}

    for i in range(len(image_list)):
        id = image_list[i].split("_")[1]
        bbox_dict = deal_single_ct(args, image_path, label_path, image_list[i], label_list[i], image_boxes[id], type)
        volumns[f"image_{id}"] = bbox_dict

    with open(os.path.join(os.path.join(args['seg_path'], type, "volumes.pkl")), 'wb') as f:
        pickle.dump(volumns, f, pickle.HIGHEST_PROTOCOL)
    print("Finish ", type, "\n\n")


def make_seg_dirs(args):
    for mode_type in ["train", "test", "val"]:
        os.makedirs(os.path.join(args["seg_path"], mode_type))
        os.makedirs(os.path.join(args["seg_path"], mode_type, "image"))
        os.makedirs(os.path.join(args["seg_path"], mode_type, "label"))
        os.makedirs(os.path.join(args["seg_path"], mode_type, "pred"))
        os.makedirs(os.path.join(args["seg_path"], mode_type, "pred", "maps"))

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

    # make_seg_dirs(args)

    get_volumes(args, image_path, label_path, image_test, label_test, "test")
    get_volumes(args, image_path, label_path, image_val, label_val, "val")
    get_volumes(args, image_path, label_path, image_train, label_train, "train")
    # get_training_set(args, image_path, label_path)
"""
针对不同阈值，对通过YOLOX生成的不同step的volumes的dice, P, R进行测试
其中step采用填充后的
最终得到一个字典，其结构如下：
{
    "train": {
        "images": {
            "xxx"(image_id) : {
                "volumes": {
                    "image_xxx_xxxx.nii": [  [[threshold1, dice, dice1, dice2, P, R, intersection, pred, seg], ...],
                    [x_min, y_min, z_min, x_max, y_max, z_max, xy_step, z_step]],
                    ...
                },
                "areas": areas,
                "nidus_num": nidus_num,
                "volumes_num": [volumes_num_step1, volumes_num_step2, volumes_num_step3],
                "dice_step1":[[threshold1, dice, dice1, dice2, P, R, intersection, pred, seg], ...],
                "dice_step2":[[threshold1, dice, dice1, dice2, P, R, intersection, pred, seg], ...],
                "dice_step3":[[threshold1, dice, dice1, dice2, P, R, intersection, pred, seg], ...]
            },
            ...
        },
        "volumes_global_dice_step1":[[threshold1, dice], ...],
        "volumes_global_dice_step2":[[threshold1, dice], ...],
        "volumes_global_dice_step3":[[threshold1, dice], ...],
        "global_dice_step1":[[threshold1, dice], ...],
        "global_dice_step2":[[threshold1, dice], ...],
        "global_dice_step3":[[threshold1, dice], ...]
    },
    "val": {...},
    "test": {...}
}
"""

import os
import scipy.ndimage as ndimage
import torch
import numpy as np
import SimpleITK as sitk
import pickle
from parameter import Para

from seg.Network import unet, kiunet_org, unet_min


def get_volumes_by_id_and_step(volumes, key, step):
    return {v_key:v_val for v_key, v_val in volumes[key].items() if (v_val and (v_val[-1] == step))}


def get_ct_by_id_and_step(volumes, key, step, ct_volumes, probability_map, count_map,
                          pred_path, threshold_list, origin_label_array):
    print("step: ", step)
    for volume in get_volumes_by_id_and_step(volumes, key, step):
        x_min, y_min, z_min, x_max, y_max, z_max, xy_step, z_step = volumes[key][volume]
        out = np.load(os.path.join(pred_path, "maps", volume.replace("nii", "npy").replace("image", "map")))
        if out.shape != origin_label_array[z_min:z_max, x_min:x_max, y_min:y_max].shape:
            continue
        pred_volume_seg = np.zeros_like(out)
        volume_dice_list = []
        for th in threshold_list:
            pred_volume_seg[out > th] = 1
            pred_volume_seg = pred_volume_seg.astype(np.uint8)
            volume_dice = get_dice(pred_volume_seg, origin_label_array[z_min:z_max, x_min:x_max, y_min:y_max])
            volume_dice_list.append([th] + list(volume_dice))
            if th == 0.5:
                print(f"th 0.5: {volume}   dice:{volume_dice[0]},  P:{volume_dice[3]},     R:{volume_dice[4]}")
                print(f"intersection: {volume_dice[5]}, pred: {volume_dice[6]}, seg: {volume_dice[7]}")
        ct_volumes[volume] = [volume_dice_list, [x_min, y_min, z_min, x_max, y_max, z_max, xy_step, z_step]]

        probability_map[z_min:z_max, x_min:x_max, y_min:y_max] += (out / (xy_step + z_step))
        count_map[z_min:z_max, x_min:x_max, y_min:y_max] += (1 / (xy_step + z_step))

    probability_map_step = probability_map.copy()
    np.divide(probability_map_step, count_map, np.zeros_like(probability_map), where=count_map != 0)
    pred_seg = np.zeros_like(probability_map_step)
    pred_seg = pred_seg.astype(np.uint8)
    dice_step = []
    for th in threshold_list:
        pred_seg[probability_map >= th] = 1
        ct_dice = get_dice(pred_seg, origin_label_array)
        dice_step.append([th] + list(ct_dice))
    return ct_volumes, probability_map, count_map, dice_step


def get_path_and_volumes(args, type):
    image_path = os.path.join(args['seg_path'], type, "image")
    label_path = os.path.join(args['seg_path'], type, "label")
    pred_path = os.path.join(args['seg_path'], type, "pred")
    origin_label_path = os.path.join(args['data_path'], "label")
    origin_label_list = os.listdir(origin_label_path)
    with open(os.path.join(args['seg_path'], type, "volumes.pkl"), 'rb') as f:
        volumes = pickle.load(f)
    return image_path, label_path, pred_path, origin_label_path, origin_label_list, volumes


def load_network(net_path, net_type):
    # 参数
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    if net_type == "unet_mini":
        net = unet_min(training=False).cuda()
    else:
        net = unet(training=False).cuda()

    net.load_state_dict(torch.load(net_path))
    net.eval()

    return net


def get_dice(pred, seg):
    intersection = (pred * seg).sum()
    union = pred.sum() + seg.sum()
    # return 2*intersection/union, 2*intersection, union, intersection/(pred.sum()), intersection/(seg.sum()), intersection, pred.sum(), seg.sum()
    return 2*intersection/union, 2*intersection, union, intersection/(pred.sum()), intersection/(seg.sum()), intersection, pred.sum(), seg.sum()


def test_volume(ct):
    with torch.no_grad():
        ct_tensor = torch.FloatTensor(ct).cuda()
        ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
        outputs = net(ct_tensor)
        probability_map = np.squeeze(outputs.cpu().detach().numpy())
        del outputs
    return probability_map


def get_global_dice(images, th_num):
    volumes_global_dice_step1 = []
    volumes_global_dice_step2 = []
    volumes_global_dice_step3 = []
    global_dice_step1 = []
    global_dice_step2 = []
    global_dice_step3 = []

    for i in range(th_num):
        dice_intersection = [0.0] * 6
        dice_union = [0.0] * 6
        for image_id in images:
            image = images[image_id]
            for v_key in image["volumes"]:
                v_list = image["volumes"][v_key]

                # # 考虑排除在没有结节的情况下，误检测到结节的情况
                # if v_list[0][i][-1] == 0:
                #     continue

                if v_list[1][-1] == 1:
                    dice_intersection[0] += v_list[0][i][2]
                    dice_union[0] += v_list[0][i][3]
                elif v_list[1][-1] == 2:
                    dice_intersection[1] += v_list[0][i][2]
                    dice_union[1] += v_list[0][i][3]
                elif v_list[1][-1] == 3:
                    dice_intersection[2] += v_list[0][i][2]
                    dice_union[2] += v_list[0][i][3]

            dice_intersection[3] += image["dice_step1"][i][2]
            dice_union[3] += image["dice_step1"][i][3]
            dice_intersection[4] += image["dice_step2"][i][2]
            dice_union[4] += image["dice_step2"][i][3]
            dice_intersection[5] += image["dice_step3"][i][2]
            dice_union[5] += image["dice_step3"][i][3]

        volumes_global_dice_step1.append(dice_intersection[0] / dice_union[0])
        volumes_global_dice_step2.append(dice_intersection[1] / dice_union[1])
        volumes_global_dice_step3.append(dice_intersection[2] / dice_union[2])
        global_dice_step1.append(dice_intersection[3] / dice_union[3])
        global_dice_step2.append(dice_intersection[4] / dice_union[4])
        global_dice_step3.append(dice_intersection[5] / dice_union[5])
    return volumes_global_dice_step1, volumes_global_dice_step2, volumes_global_dice_step3, \
           global_dice_step1, global_dice_step2, global_dice_step3


def exclude_one_image(images, th_num):
    ex_dice = {}
    for image_id in images:
        t_images = images.copy()
        t_images.pop(image_id)
        ret = get_global_dice(t_images, th_num)
        ex_dice[image_id] = {
            "image": image_id,
            "volumes_global_dice_step1": ret[0],
            "volumes_global_dice_step2": ret[1],
            "volumes_global_dice_step3": ret[2],
            "global_dice_step1": ret[3],
            "global_dice_step2": ret[4],
            "global_dice_step3": ret[5]
        }
    return ex_dice


if __name__ == '__main__':
    args = Para().get_para()
    net = load_network(os.path.join(args['root_path'], "unet.pth"), "unet")
    result = {}
    type_list = ["val", "test", "train"]
    threshold_list = [i*0.05 for i in range(19, 0, -1)] # 0.95, 0.9, ..., 0.05
    th_num = len(threshold_list)
    with open(os.path.join(args['root_path'], "nidus.pkl"), "rb") as f:
        nidus = pickle.load(f)
    for data_type in type_list:
        print(f"{data_type}: ...")
        image_path, label_path, pred_path, origin_label_path, origin_label_list, volumes = \
            get_path_and_volumes(args, data_type)
        dice_intersection = [[0.0]*th_num, [0.0]*th_num, [0.0]*th_num]
        dice_union = [[0.0]*th_num, [0.0]*th_num, [0.0]*th_num]
        images = {}
        for key in volumes:     # 针对每幅CT影像
            ct_id = key.split("_")[1]
            origin_label = sitk.ReadImage(os.path.join(origin_label_path, f"label_{ct_id}.nii"), sitk.sitkUInt8)
            origin_label_array = sitk.GetArrayFromImage(origin_label)
            probability_map = np.zeros_like(origin_label_array, dtype=np.float64)
            count_map = np.zeros_like(origin_label_array, dtype=np.float32)

            ct_volumes = {}
            ct_volumes, probability_map, count_map, dice_step1 = \
                get_ct_by_id_and_step(volumes, key, 1, ct_volumes, probability_map, count_map,
                                      pred_path, threshold_list, origin_label_array)
            ct_volumes, probability_map, count_map, dice_step2 = \
                get_ct_by_id_and_step(volumes, key, 2, ct_volumes, probability_map, count_map,
                                      pred_path, threshold_list, origin_label_array)
            ct_volumes, probability_map, count_map, dice_step3 = \
                get_ct_by_id_and_step(volumes, key, 3, ct_volumes, probability_map, count_map,
                                      pred_path, threshold_list, origin_label_array)
            images[ct_id] = {
                "volumes": ct_volumes,
                "areas": sum([i[1] for i in nidus[ct_id]]),
                "nidus_num": len(nidus[ct_id]),
                "volumes_num": [len(get_volumes_by_id_and_step(volumes, key, 1)),
                                len(get_volumes_by_id_and_step(volumes, key, 2)),
                                len(get_volumes_by_id_and_step(volumes, key, 3))],
                "dice_step1": dice_step1,
                "dice_step2": dice_step2,
                "dice_step3": dice_step3
            }
            print(f"finish {ct_id}")
            print("dice step1")
            for i in dice_step1:
                print(i)
            print()
            print("dice step2")
            for i in dice_step2:
                print(i)
            print()
            print("dice step3")
            for i in dice_step3:
                print(i)
            print()

        with open(os.path.join(args['seg_path'], f"image_{data_type}_volumes_info.pkl"), 'wb') as f:
            pickle.dump(images, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(args['seg_path'], f"image_{data_type}_volumes_info.pkl"), 'rb') as f:
            images = pickle.load(f)

        volumes_global_dice_step1, volumes_global_dice_step2, volumes_global_dice_step3, \
        global_dice_step1, global_dice_step2, global_dice_step3 = get_global_dice(images, th_num)

        result[data_type] = {
            "image": images,
            "volumes_global_dice_step1": volumes_global_dice_step1,
            "volumes_global_dice_step2": volumes_global_dice_step2,
            "volumes_global_dice_step3": volumes_global_dice_step3,
            "global_dice_step1": global_dice_step1,
            "global_dice_step2": global_dice_step2,
            "global_dice_step3": global_dice_step3
        }
        print(f"{data_type}    RESULT")
        print("volumes_global_dice_step1:")
        for i in volumes_global_dice_step1:
            print(i)
        print("volumes_global_dice_step2:")
        for i in volumes_global_dice_step2:
            print(i)
        print("volumes_global_dice_step3:")
        for i in volumes_global_dice_step3:
            print(i)
        print("volumes_global_dice_step1:")
        for i in global_dice_step1:
            print(i)
        print("volumes_global_dice_step2:")
        for i in global_dice_step2:
            print(i)
        print("volumes_global_dice_step3:")
        for i in global_dice_step3:
            print(i)
        print("\n\n\n")
    with open(os.path.join(args['seg_path'], "test_volumes_info.pkl"), 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


a = {}
for image_id in images.keys():
    tmp = images.copy()
    tmp.pop(image_id)
    volumes_global_dice_step1, volumes_global_dice_step2, volumes_global_dice_step3, \
    global_dice_step1, global_dice_step2, global_dice_step3 = get_global_dice(tmp, th_num)
    a[image_id] = {
        "volumes_global_dice_step1": volumes_global_dice_step1,
        "volumes_global_dice_step2": volumes_global_dice_step2,
        "volumes_global_dice_step3": volumes_global_dice_step3,
        "global_dice_step1": global_dice_step1,
        "global_dice_step2": global_dice_step2,
        "global_dice_step3": global_dice_step3
    }


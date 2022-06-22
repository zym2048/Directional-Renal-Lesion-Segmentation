import os
from time import time

import scipy.ndimage as ndimage
import torch
import numpy as np
import SimpleITK as sitk
import pickle
from parameter import Para

from seg.Network import unet, kiunet_org, unet_min


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
    return 2*intersection/union, 2*intersection, union, intersection/(pred.sum()), intersection/(seg.sum())


def test_volume(net, ct):
    with torch.no_grad():
        ct_tensor = torch.FloatTensor(ct).cuda()
        ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
        outputs = net(ct_tensor)
        probability_map = np.squeeze(outputs.cpu().detach().numpy())
        del outputs
    return probability_map


if __name__ == '__main__':
    args = Para().get_para()
    for data_type in ["train", "test", "val"]:
        # data_type = "val"
        image_path, label_path, pred_path, origin_label_path, origin_label_list, volumes = \
            get_path_and_volumes(args, data_type)
        # net = load_network(os.path.join(args['root_path'], "unet_mini.pth"), "unet_mini")
        net = load_network(os.path.join(args['root_path'], "unet.pth"), "unet")

        dice_intersection = 0.0
        dice_union = 0.0
        dice_info = {
            "info": "yolox_s, conf 0.25, unet, kits19",
        }
        images = []
        for key in volumes:
            ct_id = key.split("_")[1]
            origin_label = sitk.ReadImage(os.path.join(origin_label_path, f"label_{ct_id}.nii"), sitk.sitkUInt8)
            origin_label_array = sitk.GetArrayFromImage(origin_label)
            probability_map = np.zeros_like(origin_label_array, dtype=np.float64)
            count_map = np.zeros_like(origin_label_array, dtype=np.float32)
            count = 0
            for volume in volumes[key]:
                if not os.path.exists(os.path.join(image_path, volume)):
                    continue
                ct = sitk.ReadImage(os.path.join(image_path, volume), sitk.sitkInt16)
                ct = sitk.GetArrayFromImage(ct)

                x_min, y_min, z_min, x_max, y_max, z_max, xy_step, z_step = volumes[key][volume]

                # if xy_step > 2 or z_step > 2:
                #     continue
                # if xy_step > 1 or z_step > 1:
                #     continue
                count += 1
                # out = np.load(os.path.join(pred_path, "maps", volume.replace("nii", "npy").replace("image", "map")))

                out = test_volume(net, ct)
                out = ndimage.zoom(out, (xy_step, xy_step, z_step), order=1)
                np.save(os.path.join(pred_path, "maps", volume.replace("nii", "npy").replace("image", "map")), out)

                # probability_map[z_min:z_max, x_min:x_max, y_min:y_max] += out
                # count_map[z_min:z_max, x_min:x_max, y_min:y_max] += 1
                #
                # out = ndimage.zoom(out, (xy_step, xy_step, z_step), order=3)
                #
                # probability_map[z_min:z_max, x_min:x_max, y_min:y_max] += (out / (xy_step + z_step))
                # count_map[z_min:z_max, x_min:x_max, y_min:y_max] += (1 / (xy_step + z_step))
            print(f"finish image_{ct_id}.")
        #
        #     np.divide(probability_map, count_map, np.zeros_like(probability_map), where=count_map != 0)
        #     pred_seg = np.zeros_like(probability_map)
        #     # np.save(os.path.join(pred_path, f"pred_{ct_id}_probability_map.npy"), probability_map)
        #
        #     pred_seg[probability_map >= 0.5] = 1
        #     pred_seg = pred_seg.astype(np.uint8)
        #
        #     # 计算分割指标
        #     dice = get_dice(pred_seg, origin_label_array)
        #     dice_intersection += dice[1]
        #     dice_union += dice[2]
        #     images.append({
        #         "dice": dice[0],
        #         "num": count,
        #         "P": dice[3],
        #         "R": dice[4]
        #     })
        #     print(f"image_{ct_id} dice: {dice[0]}.     P:{dice[3]}.     R:{dice[4]}")
        #
        #     # 保存预测结果
        #     pred_seg = sitk.GetImageFromArray(pred_seg)
        #     pred_seg.SetDirection(origin_label.GetDirection())
        #     pred_seg.SetOrigin(origin_label.GetOrigin())
        #     pred_seg.SetSpacing(origin_label.GetSpacing())
        #
        #     sitk.WriteImage(pred_seg, os.path.join(pred_path, f"pred_{ct_id}.nii"))
        #
        # dice_info['dice_global'] = dice_intersection / dice_union
        # dice_info['images'] = images
        # with open(os.path.join(args['seg_path'], data_type, "dice_info.pkl"), 'wb') as f:
        #     pickle.dump(images, f, pickle.HIGHEST_PROTOCOL)
        # print("dice global:", dice_intersection / dice_union)

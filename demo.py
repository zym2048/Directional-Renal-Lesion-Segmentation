import SimpleITK as sitk
import scipy.ndimage as ndimage
import os
from parameter import Para
import skimage
import pickle
import nibabel as nib
import numpy as np
import cv2
from prepare.get_slices import deal_image
from seg.demo import load_network, test_volume


def data_preprocessing(origin_image_path, image_path):
    print("in data_preprocessing...")
    def change_direction(ct):
        spacing, origin = ct.GetSpacing(), ct.GetOrigin()
        new_ct = sitk.GetImageFromArray(np.flipud(sitk.GetArrayFromImage(ct).transpose((2, 1, 0))))
        new_ct.SetSpacing((spacing[1], spacing[2], spacing[0]))
        new_ct.SetOrigin((origin[1], origin[2], origin[0]))
        return new_ct

    for i in os.listdir(origin_image_path):
        ct = sitk.ReadImage(os.path.join(origin_image_path, i), sitk.sitkInt16)
        image_id = i.split('_')[1]

        change_label = False
        if ct.GetDirection() != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            ct = change_direction(ct)
            change_label = True

        ct_array = sitk.GetArrayFromImage(ct)

        if change_label:
            ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / 1, ct.GetSpacing()[1] / 1, ct.GetSpacing()[0] / 1), order=3)
        else:
            # 重采样，调整z轴spacing
            ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / 1, 1, 1), order=3)

        # 将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], 1))

        sitk.WriteImage(new_ct, os.path.join(image_path, f"case_{image_id}_0000.nii"))
        print("finish image", image_id)
    print("Finish data_preprocessing.\n\n")


def get_slices(args, image_path, slices_path):
    for i in os.listdir(image_path):
        image = np.array(nib.load(os.path.join(image_path, i)).dataobj)
        image = deal_image(args, image)
        x, y, z = image.shape

        idxs = list(range(1, z-1))[::2]

        for idx in idxs:
            filename = i.split("_")[1] + "_{:0>4d}".format(idx) + ".jpg"
            cv2.imwrite(os.path.join(slices_path, filename), image[:, :, idx - 1:idx + 2])
    print("Finish get slices.\n")


def make_dirs(args):
    def mk_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    root_path = args['root_path']
    slices_path = os.path.join(root_path, "slices")
    images_path = os.path.join(root_path, "images")
    volumes_path = os.path.join(root_path, "volumes")
    pred_path = os.path.join(root_path, "pred")
    mk_path(slices_path)
    mk_path(images_path)
    mk_path(volumes_path)
    mk_path(pred_path)
    return root_path, slices_path, images_path, volumes_path, pred_path


def yolox_demo(slice_path, yolox_net_path, yolox_output_path):
    os.system(f"python tools/demo.py image -f exps/default/yolox_s.py -c {yolox_net_path} --path {slice_path} "
              f"--conf 0.05 --nms 0.45 --tsize 640 --device gpu --output_dir {yolox_output_path} --save_result")


def get_volume_by_bbox(volumes_path, image_id, bbox, xy_step, z_step, z_depth, w, h, ct_array, ct, node_num):
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

    if ct_array_tmp.shape != (64, 64, 64):
        print("WRONG:", "image_{}_{:0>4d}.nii    xy_step:{},   z_step:{}".format(image_id, node_num, xy_step, z_step))
        return False

    new_ct = sitk.GetImageFromArray(ct_array_tmp)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], 1))

    sitk.WriteImage(new_ct, os.path.join(volumes_path, "image_{}_{:0>4d}.nii".format(image_id, node_num)))

    return x_min, y_min, z_min, x_max, y_max, z_max, xy_step, z_step


def deal_single_ct(volumes_path, image_path, image, bboxes):
    id = image.split("_")[1]

    ct = sitk.ReadImage(os.path.join(image_path, image), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    bbox_dict = {}
    z_depth, w, h = ct_array.shape
    node_num = 1
    for bbox in bboxes:
        bbox_dict["image_{}_{:0>4d}.nii".format(id, node_num)] = \
            get_volume_by_bbox(volumes_path, id, bbox, 1, 1, z_depth, w, h, ct_array, ct, node_num)
        bbox_dict["image_{}_{:0>4d}.nii".format(id, node_num+1)] = \
            get_volume_by_bbox(volumes_path, id, bbox, 2, 2, z_depth, w, h, ct_array, ct, node_num+1)
        bbox_dict["image_{}_{:0>4d}.nii".format(id, node_num+2)] = \
            get_volume_by_bbox(volumes_path, id, bbox, 3, 3, z_depth, w, h, ct_array, ct, node_num+2)
        node_num += 3
    print("finish", image)
    return bbox_dict


def get_volumes(image_path, bboxes_path, volumes_path):

    with open(bboxes_path, "rb") as f:
        bboxes = pickle.load(f)

    image_boxes = dict.fromkeys(list(set(x[:3] for x in bboxes.keys())))
    for key in image_boxes:
        image_boxes[key] = []
    for key in bboxes:
        for bbox in bboxes[key]:
            image_boxes[key[:3]].append([int((bbox[0]+bbox[2])//2), int((bbox[1]+bbox[3])//2), int(key[-4:])])

    volumes = {}
    image_list = os.listdir(image_path)
    for i in range(len(image_list)):
        id = image_list[i].split("_")[1]
        bbox_dict = deal_single_ct(volumes_path, image_path, image_list[i], image_boxes[id])
        volumes[f"image_{id}"] = bbox_dict

    with open(os.path.join(volumes_path, "volumes.pkl"), 'wb') as f:
        pickle.dump(volumes, f, pickle.HIGHEST_PROTOCOL)
    print("Finish get volumes.")


def seg_demo(image_path, volumes_image_path, pred_path, seg_net_path):
    with open(os.path.join(volumes_image_path, "volumes.pkl"), 'rb') as f:
        volumes = pickle.load(f)

    net = load_network(seg_net_path, "unet")

    for key in volumes:
        ct_id = key.split("_")[1]
        origin_image = sitk.ReadImage(os.path.join(image_path, f"case_{ct_id}_0000.nii"), sitk.sitkUInt8)
        origin_image_array = sitk.GetArrayFromImage(origin_image)
        probability_map = np.zeros_like(origin_image_array, dtype=np.float64)
        count_map = np.zeros_like(origin_image_array, dtype=np.float32)
        del origin_image_array
        for volume in volumes[key]:
            if not os.path.exists(os.path.join(volumes_image_path, volume)):
                continue

            x_min, y_min, z_min, x_max, y_max, z_max, xy_step, z_step = volumes[key][volume]

            if xy_step > 1 or z_step > 1:
                continue

            ct = sitk.ReadImage(os.path.join(volumes_image_path, volume), sitk.sitkInt16)
            ct = sitk.GetArrayFromImage(ct)

            out = test_volume(net, ct)
            out = ndimage.zoom(out, (xy_step, xy_step, z_step), order=1)

            probability_map[z_min:z_max, x_min:x_max, y_min:y_max] += out
            count_map[z_min:z_max, x_min:x_max, y_min:y_max] += 1

        np.divide(probability_map, count_map, np.zeros_like(probability_map), where=count_map != 0)
        pred_seg = np.zeros_like(probability_map)

        pred_seg[probability_map >= 0.5] = 1
        pred_seg = pred_seg.astype(np.uint8)

        # 保存预测结果
        pred_seg = sitk.GetImageFromArray(pred_seg)
        pred_seg.SetDirection(origin_image.GetDirection())
        pred_seg.SetOrigin(origin_image.GetOrigin())
        pred_seg.SetSpacing(origin_image.GetSpacing())

        sitk.WriteImage(pred_seg, os.path.join(pred_path, f"pred_{ct_id}.nii"))
        print(f"finish pred {ct_id}")



if __name__ == '__main__':
    args = Para().get_demo_para()
    yolox_net_path = "best_ckpt.pth"
    seg_net_path = "unet.pth"
    root_path, slices_path, images_path, volumes_path, pred_path = make_dirs(args)
    data_preprocessing(args['input_path'], images_path)
    get_slices(args, images_path, slices_path)
    yolox_demo(slices_path, yolox_net_path, os.path.join(root_path, "yolox_output"))
    get_volumes(images_path, os.path.join(root_path, "yolox_output", "bboxes.pkl"), volumes_path)
    seg_demo(images_path, volumes_path, pred_path, seg_net_path)









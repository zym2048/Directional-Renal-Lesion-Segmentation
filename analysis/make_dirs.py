import os
from parameter import Para

if __name__ == '__main__':

    args = Para().get_para()

    root_path = args['root_path']

    os.makedirs(os.path.join(root_path, "Kits_coco"))
    os.makedirs(os.path.join(root_path, "Kits_coco", "annotations"))
    os.makedirs(os.path.join(root_path, "Kits_coco", "train2017"))
    os.makedirs(os.path.join(root_path, "Kits_coco", "test2017"))
    os.makedirs(os.path.join(root_path, "Kits_coco", "val2017"))

    os.makedirs(os.path.join(root_path, "Kits_data"))
    os.makedirs(os.path.join(root_path, "Kits_data", "image"))
    os.makedirs(os.path.join(root_path, "Kits_data", "label"))

    os.makedirs(os.path.join(root_path, "Kits_origin_data"))
    os.makedirs(os.path.join(root_path, "Kits_origin_data", "image"))
    os.makedirs(os.path.join(root_path, "Kits_origin_data", "label"))

    os.makedirs(os.path.join(root_path, "Kits_pics"))

    os.makedirs(os.path.join(root_path, "Kits_seg"))
    for mode_type in ["train", "test", "val"]:
        os.makedirs(os.path.join(root_path, "Kits_seg", mode_type))
        os.makedirs(os.path.join(root_path, "Kits_seg", mode_type, "image"))
        os.makedirs(os.path.join(root_path, "Kits_seg", mode_type, "label"))
        os.makedirs(os.path.join(root_path, "Kits_seg", mode_type, "pred"))
        os.makedirs(os.path.join(root_path, "Kits_seg", mode_type, "pred", "maps"))

    os.makedirs(os.path.join(root_path, "Kits_yolox_output"))
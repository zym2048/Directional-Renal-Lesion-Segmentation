import os
import pickle

class Para():
    def __init__(self):
        para_dir = {}
        demo_dir = {}
        # 测试用参数
        demo_dir['input_path'] = "..."
        demo_dir['root_path'] = "..."
        demo_dir['HU_HIGH'] = 400
        demo_dir['HU_LOW'] = -200

        # 目录
        para_dir['root_path'] = "..."
        para_dir['coco_path'] = os.path.join(para_dir['root_path'], "Kits_coco")
        para_dir['origin_data_path'] = os.path.join(para_dir['root_path'], "Kits_origin_data")
        para_dir['data_path'] = os.path.join(para_dir['root_path'], "Kits_data")
        para_dir['yolox_output_path'] = os.path.join(para_dir['root_path'], "Kits_yolox_output")
        para_dir['seg_path'] = os.path.join(para_dir['root_path'], "Kits_seg")

        # HU窗口
        para_dir['HU_HIGH'] = 400
        para_dir['HU_LOW'] = -200

        # 随机数
        para_dir['random_seed'] = 57

        # 划分的训练测试集
        if os.path.exists(os.path.join(para_dir['root_path'], "partition.pkl")):
            with open(os.path.join(para_dir['root_path'], "partition.pkl"), "rb") as f:
                partition = pickle.load(f)
                para_dir["nidus_train"] = partition["nidus_train"]
                para_dir["nidus_val"] = partition["nidus_val"]
                para_dir["nidus_test"] = partition["nidus_test"]

        self.para_dir = para_dir
        self.demo_dir = demo_dir

    def get_para(self):
        return self.para_dir

    def get_demo_para(self):
        return self.demo_dir
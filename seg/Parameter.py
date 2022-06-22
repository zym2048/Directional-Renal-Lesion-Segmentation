
train_ct_path = '...'

train_seg_path = '...'

test_ct_path = '...'

test_seg_path = '...'
# test_ct_path = './data/train/ct'
#
# test_seg_path = './data/train/seg'


training_set_path = '...'
pred_path = './result/old_dataset/unet_mini'

crf_path = './crf'

module_path = './module/UNet_min/net200-0.059-0.162-0.496.pth'

size = 48  # 使用48张连续切片作为网络的输入

down_scale = 0.5  # 横断面降采样因子

expand_slice = 20  # 仅使用包含肝脏以及肝脏上下20张切片作为训练样本

slice_thickness = 1  # 将所有数据在z轴的spacing归一化到1mm

upper, lower = 200, -200  # CT数据灰度截断窗口

# ---------------------训练数据获取相关参数-----------------------------------


# -----------------------网络结构相关参数------------------------------------

drop_rate = 0.3  # dropout随机丢弃概率

# -----------------------网络结构相关参数------------------------------------


# ---------------------网络训练相关参数--------------------------------------

gpu = '0'  # 使用的显卡序号

Epoch = 400

learning_rate = 1e-4

learning_rate_decay = [500, 750]

# learning_rate = 5e-4
#
# learning_rate_decay = [400, 650]

alpha = 0.33  # 深度监督衰减系数

batch_size = 1

num_workers = 0

pin_memory = True

cudnn_benchmark = True

# ---------------------网络训练相关参数--------------------------------------


# ----------------------模型测试相关参数-------------------------------------

threshold = 0.5  # 阈值度阈值


stride = 12  # 滑动取样步长

maximum_hole = 5e4  # 最大的空洞面积

# ----------------------模型测试相关参数-------------------------------------


# ---------------------CRF后处理优化相关参数----------------------------------

z_expand, x_expand, y_expand = 10, 30, 30  # 根据预测结果在三个方向上的扩展数量

max_iter = 20  # CRF迭代次数

s1, s2, s3 = 1, 10, 10  # CRF高斯核参数

# ---------------------CRF后处理优化相关参数----------------------------------
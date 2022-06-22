# 肾部病灶定向分割算法
## 参考代码
本文代码参考如下仓库代码：

**YOLOX**: https://github.com/Megvii-BaseDetection/YOLOX

**KiU-Net**: https://github.com/jeya-maria-jose/KiU-Net-pytorch



## 数据预处理

首先在parameter.py代码中进行参数调整。



运行如下代码进行文件夹创建：

```python
python analysis\make_dirs.py
```



运行如下代码进行数据集预处理：

```
python analysis\dataset_analysis.py
```



运行如下代码进行数据集划分：

```
python analysis\dataset_partition.py
```



运行如下代码制作用于目标检测的COCO格式数据集：

```
python prepare\get_slices.py
```



运行如下代码制作用于局部语义分割的数据集：

```
python prepare\get_volumes.py
```



## 预训练模型

目标检测阶段yolox_s网络预训练模型可从如下方式获取：

链接：https://pan.baidu.com/s/19JmAy0KRccUnpg5cNlivpg 
提取码：rai1



语义分割阶段unet3D网络预训练模型可从如下方式获取：

链接：https://pan.baidu.com/s/12Jtdsi_f12gy9tFuuEhczA 
提取码：w6mj



## 测试

运行如下代码进行算法测试:

```
python demo.py
```



## 训练

在制作各部分对应数据集后，可通过运行如下代码进行模型训练：

目标检测阶段：

```
python -m yolox.tools.train -n yolox-s -d 1 -b 8
```

语义分割阶段（需要首先调整seg/Parameter.py中的参数）：

```
python seg/train.py
```


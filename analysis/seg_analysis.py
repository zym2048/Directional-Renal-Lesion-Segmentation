import pickle
import pandas as pd
import os
from parameter import Para


with open(f"..\\dice_0.926265237634037.pkl", 'rb') as f:
    dice_for_file = pickle.load(f)

args = Para().get_para()
with open(os.path.join(os.path.join(args['seg_path'], "trainset", "volumes.pkl")), 'rb') as f:
    volumes_info = pickle.load(f)

info = []
for key in dice_for_file:
    v = volumes_info[key]
    info.append([key, v["image"], v["nudis_areas"], dice_for_file[key]] + list(v["offset"]))

df = pd.DataFrame(info, columns=["name", "id", "areas", "pred", "x_offset", "y_offset", "z_offset", "xy_step", "z_step"])

# 相关性分析
df.corr(method='spearman')
#              areas      pred  x_offset  y_offset  z_offset   xy_step    z_step
# areas     1.000000  0.537997   0.00000  0.000000  0.000000 -0.006852 -0.008933
# pred      0.537997  1.000000  -0.00236  0.027223 -0.019871 -0.310742 -0.318441
# x_offset  0.000000 -0.002360   1.00000  0.000000  0.000000  0.000000  0.000000
# y_offset  0.000000  0.027223   0.00000  1.000000  0.000000  0.000000  0.000000
# z_offset  0.000000 -0.019871   0.00000  0.000000  1.000000  0.000000  0.000000
# xy_step  -0.006852 -0.310742   0.00000  0.000000  0.000000  1.000000  0.745916
# z_step   -0.008933 -0.318441   0.00000  0.000000  0.000000  0.745916  1.000000
# 分析：与预测pred相比，区域大小正相关，说明大区域检测效果较好，但如何相关需进一步讨论
# xy_step，z_step负相关

df_step_0 = df[(df['xy_step'] == 1) & (df['z_step'] == 1)]
# df_step_0[df_step_0['pred']<0.5]
#                     name   id  areas      pred  x_offset  y_offset  z_offset  \
# 55    image_008_0016.nii  008    200  0.456033         0         0         0
# 117   image_012_0018.nii  012     19  0.000000        15        15       -15
# 119   image_012_0020.nii  012     19  0.000000        15       -15       -15
# 121   image_012_0022.nii  012     19  0.000000       -15        15       -15
# 123   image_012_0024.nii  012     19  0.000000       -15       -15       -15
# 175   image_022_0016.nii  022    183  0.482659         0         0         0
# 176   image_022_0017.nii  022    183  0.361797        15        15        15
# 256   image_028_0022.nii  028    126  0.477842       -15        15       -15
# 565   image_071_0136.nii  071      6  0.000000         0         0         0
# 566   image_071_0137.nii  071      6  0.000000        15        15        15
# 567   image_071_0138.nii  071      6  0.000000        15        15       -15
# 568   image_071_0139.nii  071      6  0.000000        15       -15        15
# 569   image_071_0140.nii  071      6  0.000000        15       -15       -15
# 570   image_071_0141.nii  071      6  0.000000       -15        15        15
# 571   image_071_0142.nii  071      6  0.177858       -15        15       -15
# 572   image_071_0143.nii  071      6  0.000000       -15       -15        15
# 573   image_071_0144.nii  071      6  0.000000       -15       -15       -15
# 729   image_096_0062.nii  096     31  0.418919        15        15        15
# 733   image_096_0066.nii  096     31  0.000000       -15        15        15
# 734   image_096_0067.nii  096     31  0.000000       -15        15       -15
# 735   image_096_0068.nii  096     31  0.000000       -15       -15        15
# 780   image_096_0113.nii  096    260  0.243514       -15       -15        15
# 881   image_099_0049.nii  099    244  0.354447        15       -15        15
# 882   image_099_0050.nii  099    244  0.152019        15       -15       -15
# 896   image_099_0064.nii  099     82  0.442722        15       -15        15
# 897   image_099_0065.nii  099     82  0.316699        15       -15       -15
# 1088  image_107_0020.nii  107     32  0.405136        15       -15       -15
# 1174  image_107_0106.nii  107    289  0.414040         0         0         0
# 1193  image_107_0125.nii  107    343  0.410421        15       -15       -15
# 1809  image_189_0001.nii  189    107  0.000000         0         0         0
# 1810  image_189_0002.nii  189    107  0.000000        15        15        15
# 1811  image_189_0003.nii  189    107  0.000000        15        15       -15
# 1812  image_189_0004.nii  189    107  0.000000        15       -15        15
# 1813  image_189_0005.nii  189    107  0.000000        15       -15       -15
# 1814  image_189_0006.nii  189    107  0.000000       -15        15        15
# 1815  image_189_0007.nii  189    107  0.000000       -15        15       -15
# 1816  image_189_0008.nii  189    107  0.000000       -15       -15        15
# 1817  image_189_0009.nii  189    107  0.000000       -15       -15       -15
# 1824  image_189_0016.nii  189      1  0.000000         0         0         0
# 1825  image_189_0017.nii  189      1  0.000000        15        15        15
# 1826  image_189_0018.nii  189      1  0.000000        15        15       -15
# 1827  image_189_0019.nii  189      1  0.000000        15       -15        15
# 1828  image_189_0020.nii  189      1  0.000000        15       -15       -15
# 1829  image_189_0021.nii  189      1  0.000000       -15        15        15
# 1830  image_189_0022.nii  189      1  0.000000       -15        15       -15
# 1831  image_189_0023.nii  189      1  0.000000       -15       -15        15
# 1832  image_189_0024.nii  189      1  0.000000       -15       -15       -15
# 1978  image_220_0035.nii  220    192  0.438040        15       -15       -15
# 分析：从训练集准确率上看 对于step == 1的情况，分辨率不好的情况基本都是小样本

# areas.describe()
# count  2.430000e+02
# mean   7.236635e+04
# std    1.938134e+05
# min    1.000000e+00
# 25%    2.470000e+02
# 50%    3.264000e+03
# 75%    3.749250e+04
# max    1.297687e+06

def getAreasclass(df):
    if df['areas'] < 250:
        return 1
    elif df['areas'] < 3200:
        return 2
    elif df['areas'] < 37500:
        return 3
    elif df['areas'] < 70000:
        return 4
    else:
        return 5

df_step_0.loc[:, "areas_class"] = df.apply(getAreasclass, axis=1)

# df_step_0[df_step_0['areas_class']==1]['pred'].describe()
# count    243.000000
# mean       0.684953
# std        0.305580
# min        0.000000
# 25%        0.637832
# 50%        0.820836
# 75%        0.887314
# max        0.981188

# df_step_0[df_step_0['areas_class']==2]['pred'].describe()
# Out[77]:
# count    333.000000
# mean       0.781759
# std        0.101421
# min        0.243514
# 25%        0.739650
# 50%        0.797337
# 75%        0.852509
# max        0.953217

# df_step_0[df_step_0['areas_class']==3]['pred'].describe()
# count    315.000000
# mean       0.867185
# std        0.066804
# min        0.512974
# 25%        0.844038
# 50%        0.880544
# 75%        0.917744
# max        0.960733

# df_step_0[df_step_0['areas_class']==4]['pred'].describe()
# count    63.000000
# mean      0.907614
# std       0.032585
# min       0.830348
# 25%       0.885846
# 50%       0.910828
# 75%       0.931562
# max       0.954701

# df_step_0[df_step_0['areas_class']==5]['pred'].describe()
# count    252.000000
# mean       0.946121
# std        0.031930
# min        0.818492
# 25%        0.931758
# 50%        0.950522
# 75%        0.967815
# max        0.999901

df_step_not_0 = df[(df['xy_step'] > 1) | (df['z_step'] > 1)]
# df_step_not_0['pred'].describe()
# count    783.000000
# mean       0.711064
# std        0.239071
# min        0.000000
# 25%        0.650365
# 50%        0.794564
# 75%        0.875383
# max        0.985692

df_step_not_0[(df_step_not_0['xy_step'] + df_step_not_0['z_step']) == 3]['pred'].describe()
# count    267.000000
# mean       0.764283
# std        0.198071
# min        0.000000
# 25%        0.731330
# 50%        0.817505
# 75%        0.892331
# max        0.985692

df_step_not_0[(df_step_not_0['xy_step'] == 2) & (df_step_not_0['z_step'] == 2)]['pred'].describe()
# count    133.000000
# mean       0.736622
# std        0.214804
# min        0.000000
# 25%        0.686344
# 50%        0.810395
# 75%        0.880766
# max        0.965947

df_step_not_0[(df_step_not_0['xy_step'] + df_step_not_0['z_step']) == 5]['pred'].describe()
# count    258.000000
# mean       0.678188
# std        0.257826
# min        0.000000
# 25%        0.625783
# 50%        0.772033
# 75%        0.856362
# max        0.963209

df.loc[:, "areas_class"] = df.apply(getAreasclass, axis=1)
df['step'] = df['xy_step']+df['z_step']
# 再分析step, areas, pred三者之间的相关性
df[df['step'] == 2][['areas', 'pred']].corr(method='spearman')  # 0.654065
df[df['step'] == 3][['areas', 'pred']].corr(method='spearman')  # 0.502901
df[df['step'] == 4][['areas', 'pred']].corr(method='spearman')  # 0.433428
df[df['step'] == 5][['areas', 'pred']].corr(method='spearman')  # 0.411508
df[df['step'] == 6][['areas', 'pred']].corr(method='spearman')  # 0.420584

df[df['areas_class'] == 1][['step', 'pred']].corr(method='spearman')  # -0.107744
df[df['areas_class'] == 2][['step', 'pred']].corr(method='spearman')  # -0.451624
df[df['areas_class'] == 3][['step', 'pred']].corr(method='spearman')  # -0.538286
df[df['areas_class'] == 4][['step', 'pred']].corr(method='spearman')  # -0.744728
df[df['areas_class'] == 5][['step', 'pred']].corr(method='spearman')  # -0.606621


# 训练集分析：
# 1. 效果与大小正相关
# 2. 效果与步长负相关
# 3. 结节越大，step与预测准确度的负相关性越强？
# 下一步的工作是，根据目标检测结构（目前是目标检测GT），计算三维卷积区域
# 首先，先去看 3D卷积 数据集是什么样的
# 数据预处理可以参考： LiTS/data_prepare 进行修改，注意里面关于Spacing等内容的修改
# 有时间可以看一下KiU-Net的代码结构
# 最后得到的结果是对各个病灶，保存一个 64*64*64的CT影像nii文件，与一个标签nii文件

import os


def getIOU(x1min, x1max, y1min, y1max, x2min, x2max, y2min, y2max):
    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    return (ymax - ymin) * (xmax - xmin)

SaveROOT = "..\\dataset"

XMLROOT = os.path.join(SaveROOT, "KITS")
XMLPaths = os.listdir(XMLROOT)
ids = list(set([id[0:3] for id in XMLPaths]))
ids.sort()
node_num = 1
for id in ids:
    CT_XML = [x for x in XMLPaths if f"{id}_" in x]
    CT_XML.sort()

    imgRoot = os.path.join("..\\Kits_data", "image")
    labelRoot = os.path.join("..\\Kits_data", "label")

    # 获取所有影像及标签路径，两者在列表中一一对应
    # ..\\case_xxx_0000.nii    use [-12:-9] to get xxx
    if os.path.isfile(os.path.join(labelRoot, f"label_{id}.nii")):
        labelpath = os.path.join(labelRoot, f"label_{id}.nii")
    else:
        labelpath = os.path.join(labelRoot, f"case_{id}_0000.nii")

    import SimpleITK as sitk

    ct = sitk.ReadImage(os.path.join(imgRoot, f"case_{id}_0000.nii"), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    print(labelpath)
    seg = sitk.ReadImage(labelpath, sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    from xml.dom.minidom import parse

    z_depth, w, h = ct_array.shape
    nodes = []
    # nodes =[    [  node1:[z1, xmin, xmax, ymin, ymax],...,[zn, xmin, xmax, ymin, ymax]  ]
    #             [  node2:[z1, xmin, xmax, ymin, ymax],...,[zn, xmin, xmax, ymin, ymax]  ]
    #       ]
    for slice in CT_XML:

        # xxx_xxxx.xml[-8:-4] for xxxx
        z = int(slice[-8:-4])
        root = parse(os.path.join(XMLROOT, slice)).documentElement

        # w = int(root.getElementsByTagName("size")[0].getElementsByTagName("width")[0].childNodes[0].data)
        # h = int(root.getElementsByTagName("size")[0].getElementsByTagName("height")[0].childNodes[0].data)
        # z_depth = int(root.getElementsByTagName("size")[0].getElementsByTagName("z_depth")[0].childNodes[0].data)

        objects = root.getElementsByTagName("object")
        for obj in objects:
            boundingbox = obj.getElementsByTagName("bndbox")[0]
            xmin = int(boundingbox.getElementsByTagName("xmin")[0].childNodes[0].data)
            xmax = int(boundingbox.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymin = int(boundingbox.getElementsByTagName("ymin")[0].childNodes[0].data)
            ymax = int(boundingbox.getElementsByTagName("ymax")[0].childNodes[0].data)

            # print(z, " :  ", xmin, xmax, ymin, ymax)
            if len(nodes) == 0:
                nodes.append([[z, xmin, xmax, ymin, ymax]])
                continue
            L = []
            for idx in range(len(nodes)):
                node = nodes[idx][-1]
                iou = getIOU(xmin, xmax, ymin, ymax,
                             node[1], node[2], node[3], node[4])
                if abs(z - node[0]) < 10 and iou > 0:
                    L.append([idx, iou])

            if len(L) > 0:
                sorted(L, key=lambda x: x[1])
                nodes[L[-1][0]].append([z, xmin, xmax, ymin, ymax])
            else:
                nodes.append([[z, xmin, xmax, ymin, ymax]])

    print(nodes)
    print(len(nodes))
    for node in nodes:
        node_xmin = min(node, key=lambda x: x[1])[1]
        node_xmax = max(node, key=lambda x: x[2])[2]
        node_ymin = min(node, key=lambda x: x[3])[3]
        node_ymax = max(node, key=lambda x: x[4])[4]
        node_zmin, node_zmax = node[0][0], node[-1][0]

        node_width, node_height = node_xmax - node_xmin, node_ymax - node_ymin
        xy_step = max(node_width, node_height) // 64 + 1
        z_step = (node_zmax - node_zmin) // 64 + 1

        x_min = (node_xmin + node_xmax) // 2 - xy_step * 32
        x_max = (node_xmin + node_xmax) // 2 + xy_step * 32
        y_min = (node_ymin + node_ymax) // 2 - xy_step * 32
        y_max = (node_ymin + node_ymax) // 2 + xy_step * 32
        z_min = (node_zmin + node_zmax) // 2 - z_step * 32
        z_max = (node_zmin + node_zmax) // 2 + z_step * 32

        # 假如体积框超过了边界，则让其向周围平移
        if x_min < 0:
            x_max, x_min = x_max + (-x_min), 0
        if x_max >= w:
            x_min, x_max = x_min - (w - x_max - 1), w - 1
        if y_min < 0:
            y_max, y_min = y_max + (-y_min), 0
        if y_max >= h:
            y_min, y_max = y_min - (h - y_max - 1), h - 1
        if z_min < 0:
            z_max, z_min = z_max + (-z_min), 0
        if z_max >= z_depth:
            z_min, z_max = z_min - (z_depth - z_max - 1), z_depth - 1

        x_min, y_min, z_min = max(x_min, 0), max(y_min, 0), max(z_min, 0)
        x_max, y_max, z_max = min(x_max, w - 1), min(y_max, h - 1), min(z_max, z_depth - 1)
        # print("ct_array shape : ", ct_array.shape)
        # print("seg_array shape : ", seg_array.shape)

        # print(x_min, y_min, z_min, x_max, y_max, z_max)
        # print(z_step, xy_step)
        ct_array_tmp = ct_array[z_min:z_max:z_step, x_min:x_max:xy_step, y_min:y_max:xy_step]
        seg_array_tmp = seg_array[z_min:z_max:z_step, x_min:x_max:xy_step, y_min:y_max:xy_step]
        # print("ct_array shape : ", ct_array_tmp.shape)
        # print("seg_array shape : ", seg_array_tmp.shape)
        # 最终将数据保存为nii
        new_ct = sitk.GetImageFromArray(ct_array_tmp)

        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing(ct.GetSpacing())

        new_seg = sitk.GetImageFromArray(seg_array_tmp)

        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(new_ct, os.path.join(SaveROOT, "new", "image", "image_{}_{:0>4d}.nii".format(id, node_num)))
        sitk.WriteImage(new_seg, os.path.join(SaveROOT, "new", "label", "label_{}_{:0>4d}.nii".format(id, node_num)))
        print(f"finish id:{id}, node_num:{node_num}")
        node_num += 1

    print(f"finish id:{id}\n")

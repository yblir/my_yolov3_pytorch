import numpy as np
import torch


def get_anchors(anchors_path):
    '''
    获得所有先验框
    :return:
    '''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]

    return np.array(anchors).reshape(-1, 2)


def get_classes(classes_path):
    '''
    获取所有类别名
    :return:
    '''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def box_iou(box1, box2):
    '''
    计算两个矩形框iou交并比
    :param box1: tensor,wh,[x,y,w,h]
    :param box2: tensor,anchor,[x,y,w,h]
    :return:
    '''
    # box1 (-1,3,13,13,25)
    b1 = box1.unsqueeze(-2)  # (-1,3,13,13,1,25)
    b1_xy, b1_wh = b1[..., :2], b1[..., 2:4]
    b1_min = b1_xy - b1_wh / 2
    b1_max = b1_xy + b1_wh / 2

    # box2 (3,4)
    b2 = box2.unsqueeze(0)  # (1,3,4)
    b2_xy, b2_wh = b2[..., :2], b2[..., 2:4]
    b2_min = b2_xy - b2_wh / 2
    b2_max = b2_xy + b2_wh / 2

    # 获得相交部分的坐标
    inter_min = torch.maximum(b1_min, b2_min)
    inter_max = torch.minimum(b1_max, b2_max)
    inter_wh = torch.clamp(inter_max - inter_min, min=0)

    # 获得各部分面积
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    # (n,9) n为真实框数量
    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def decode_pred_boxes(feat, anchors, num_classes, input_shape, training=False):
    '''
    解码预测框
    :param feat: 模型预测结果,(b,75,13,13)
    :param anchors: 416尺度下的先验框，(3,2)
    :param num_classes:类别数量
    :param input_shape:(416,416)
    :param training:预测模式时为False
    :return:
    '''
    # 当前输出特征对应的anchors
    num_anchors = len(anchors)
    anchors_reshape = np.reshape(anchors, [num_anchors, 1, 1, 2])
    grid_shape = feat.shape[2:4]  # w,h

    # 构建13x13网格,x-w,y-h
    x = torch.reshape(torch.arange(0, end=grid_shape[1]), [1, 1, -1, 1])
    y = torch.reshape(torch.arange(0, end=grid_shape[0]), [1, -1, 1, 1])
    grid_x = torch.tile(x, [num_anchors, grid_shape[0], 1, 1])
    grid_y = torch.tile(y, [num_anchors, 1, grid_shape[1], 1])
    grid_xy = torch.cat([grid_x, grid_y], dim=-1)  # (3,13,13,2)

    # b,75,13,13 => b,3,13,13,25
    y_pred = feat.reshape(-1, num_anchors, 5 + num_classes,
                          grid_shape[0], grid_shape[1]).permute(0, 1, 3, 4, 2).contiguous()
    # 模型预测的是先验框到真实的的变化量, 因此要获得预测框中心,要加上所在网格位置, 要获得宽高,要乘上原有anchor宽高
    # 预测框xy=o(预测的xy变化量)+grid_xy
    # todo cuda ?
    box_xy = (torch.sigmoid(y_pred[..., :2]) + grid_xy) / torch.tensor(grid_shape[::-1])

    # 预测框wh=先验框*e^预测的wh比例变化量
    box_wh = torch.exp(y_pred[..., 2:4]) * torch.from_numpy(anchors_reshape / input_shape[::-1])
    box_wh = box_wh.type(torch.float32)  # box_xy,box_wh统一类型
    pred_box = torch.cat([box_xy, box_wh], dim=-1)  # (b,3,13,13,4)

    pred_xy = torch.sigmoid(y_pred[..., 0:2])
    pred_wh = y_pred[..., 2:4]  # wh预测不需要sigmoid, 是因为与真实值是比例关系,不怕越界
    pred_conf = torch.sigmoid(y_pred[..., 4:5])
    pred_class = torch.sigmoid(y_pred[..., 5:])

    if training:
        return grid_xy, y_pred, pred_box, pred_xy, pred_wh, pred_conf, pred_class

    return box_xy, box_wh, pred_conf, pred_class


def encode_true_boxes(true_boxes, anchors, input_shape, num_classes):
    '''
    编码真实框
    :param true_boxes: list,长度等于batch_size,每个元素是矩阵,(n,5),(x,y,w,h,c) n为每张图片标注框数量,已归一化
    :param anchors: (9,2), 相对于416尺寸
    :param num_classes: voc=20
    :param input_shape: np.array, (416,416),h,w
    :return:
    '''
    num_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # 计算一个batch有多少图片
    batch_size = len(true_boxes)
    # 构建(13,13),(26,26),(52,52) 尺寸的真实值全0空壳,用于填入真实值
    grid_shape = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]  # h,w
    # b,3,13,13,85
    y_true = [
        np.zeros(
            (batch_size, len(anchor_mask[i]), grid_shape[i][0], grid_shape[i][1], 5 + num_classes)
        ) for i in range(num_layers)
    ]

    # 处理anchor宽高,适应box_iou函数 32,14->(0,0,32,14)
    anchor_tensor = torch.from_numpy(anchors)
    anchor_boxes = torch.cat([torch.zeros_like(anchor_tensor), anchor_tensor], dim=-1)

    for b in range(batch_size):
        box_wh = true_boxes[b][..., 2:4] * input_shape[::-1]  # hw统一到416尺度, 之后再与416尺度下的anchor计算iou
        # valid_mask = box_wh[:, 0] > 0
        wh = box_wh[box_wh[:, 0] > 0]  # 提取真实框宽高, 去除真实框所有的填充项0,,box_wh shape=(n,2)
        if len(wh) == 0:
            continue  # 过滤一个标注框都没有真实值,真有这种情况的!

        wh = torch.from_numpy(wh)
        wh_boxes = torch.cat([torch.zeros_like(wh), wh], dim=-1)
        # 找出与每个真实框iou交并比最大的anchor框编号,(n,9)
        iou = box_iou(wh_boxes, anchor_boxes)
        best_anchor_index = np.argmax(iou, axis=-1)  # (n,)

        # i指这张图上的第几个标注框,idx指与该标注框iou最大的anchor对应的编号
        for i, idx in enumerate(best_anchor_index):
            for j in range(num_layers):
                if idx in anchor_mask[j]:
                    # x指标注框水平方向,y指竖直方向,grid_shape:h,w
                    x = np.floor(true_boxes[b][i, 0] * grid_shape[j][1]).astype('int32')
                    y = np.floor(true_boxes[b][i, 1] * grid_shape[j][0]).astype('int32')
                    k = anchor_mask[j].index(idx)  # k指该层中的第几个先验框,如13x13中的第2层
                    c = true_boxes[b][i, 4].astype('int32')

                    y_true[j][b, k, y, x, 0:4] = true_boxes[b][i, 0:4]
                    y_true[j][b, k, y, x, 4] = 1  # 是否有物体的置信度
                    y_true[j][b, k, y, x, 5 + c] = 1

                    break  # 一个标注框只会与一个anchor有最大iou,找到后跳出内层循环,计算下一个标注框

    y_true = [torch.from_numpy(item).to(torch.float32) for item in y_true]

    return y_true


def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''
    校准预测框
    :param box_xy:
    :param box_wh:
    :param input_shape:
    :param image_shape:
    :return:
    '''
    # 把y轴放前面是因为方便预测框和图像的宽高进行相乘
    box_yx = box_xy.cpu().numpy()[..., ::-1]
    box_hw = box_wh.cpu().numpy()[..., ::-1]

    # new_shape指的是宽高缩放情况
    new_shape = np.round(image_shape * np.min(input_shape / image_shape))
    # 这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = torch.from_numpy(box_yx - (box_hw / 2.))
    box_maxes = torch.from_numpy(box_yx + (box_hw / 2.))

    boxes = torch.cat([box_mins[..., 0:1], box_mins[..., 1:2],  # y_min x_min
                       box_maxes[..., 0:1], box_maxes[..., 1:2]  # y_max x_max
                       ], dim=-1)

    image_shape = torch.from_numpy(image_shape)
    boxes *= torch.cat([image_shape, image_shape], dim=-1)
    boxes = torch.reshape(boxes, [-1, 4]).type(torch.float32)

    return boxes

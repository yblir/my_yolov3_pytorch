import numpy as np
import torch
from torch import nn

from codes.util_func import box_iou, decode_pred_boxes

np.set_printoptions(threshold=np.inf)


class YoloLoss(nn.Module):
    def __init__(self, anchors, input_shape, num_classes):
        super(YoloLoss, self).__init__()
        self.anchors = anchors
        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_layers = len(anchors) // 3

        self.normalize = False

    @staticmethod
    def clip_by_tensor(t, t_min, t_max):
        t = t.float()

        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    @staticmethod
    def MSELoss(pred, target):
        return (pred - target) ** 2

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def negative_mask(self, pred_box, target, object_mask):
        '''
        挑选出iou小于阈值的,作为负样本
        如某张图片13x13的所有输出,与该图片所有真实框做iou,确定负样本
        :param pred_box:解码后的预测框,tensor,(b,3,13,13,4) 已归一化
        :param target:真实框,np.array,(b,n,5),n:为每张图片标注框数量
        :param object_mask:
        :return:
        '''
        batch_sz = pred_box.shape[0]
        pred_box_h, pred_box_w = pred_box.shape[2:4]
        neg_list = [None] * batch_sz

        for b in range(batch_sz):
            true_box = target[b][:, :4]

            if true_box.any():
                iou = box_iou(pred_box[b], torch.from_numpy(true_box))
                best_iou, _ = torch.max(iou, dim=-1)
                neg_mask = best_iou < 0.5
            else:
                neg_mask = torch.zeros(self.num_layers, pred_box_h, pred_box_w)

            neg_list[b] = neg_mask
        # =>b,3,13,13
        neg_tensor = torch.stack(neg_list, dim=0)
        # object_mask是该层的正样本,那么1-object_mask就是去除正样本后的所有负样本
        # 在这些可选负样本中再挑选iou小于0.5的选项作为最终负样本
        neg_tensor = (1 - object_mask) * torch.unsqueeze(neg_tensor, dim=-1)
        # =>b,3,13,13,1
        return neg_tensor

    def forward(self, y_pred, true_data_boxes):
        '''
        yolov3损失函数
        :param y_pred: 模型输出特征[tensor([b,255,13,13]),...]
        :param true_data_boxes: 真实值+已编码的真实值[tensor([b,3,13,13,85],...)]
        :return:
        '''

        target, y_true = true_data_boxes
        grid_shape = [torch.tensor(y_pred[i].shape[2:4]) for i in range(self.num_layers)]
        batch_size = y_pred[0].shape[0]
        losses = []
        all_pos = 0  # 正样本数量

        for i in range(self.num_layers):
            object_mask = y_true[i][..., 4:5]  # (b,3,13,13,1)
            cur_anchors = self.anchors[self.anchor_mask[i]]  # 如13x13特征层对应的三个anchors,shape=(3,2)
            # 谁让通道维度在前呢! 只能改变shape了. 如13x13特征层共有三个通道,每个通道对应1个anchor框
            anchor_reshape = torch.from_numpy(cur_anchors.reshape(3, 1, 1, 2))

            # pred_box不是pred_xy,wh等模型输出的直接堆叠,而是经过公式计算后得到的预测框,直接指示预测位置
            grid_xy, raw_pred, pred_box, pred_xy, pred_wh, \
            pred_conf, pred_class = decode_pred_boxes(y_pred[i], cur_anchors,
                                                      self.num_classes, self.input_shape, training=True)

            # 根据计算公式,反推出真实框xy,wh应该对应的预测值是什么, 对应的预测值就是pred_xy,pred_wh
            true_xy = y_true[i][..., :2] * grid_shape[i] - grid_xy
            true_wh = torch.log(y_true[i][..., 2:4] *
                                torch.from_numpy(self.input_shape[::-1].copy()) / anchor_reshape)

            # 去除所有的log(0)=-inf
            true_wh = torch.where(object_mask > 0, true_wh, torch.zeros_like(true_wh))

            # (b,3,13,13,1)
            loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]
            neg_mask = self.negative_mask(pred_box, target, object_mask)

            loss_xy = torch.sum(object_mask * loss_scale * self.BCELoss(pred_xy, true_xy))
            loss_wh = torch.sum(object_mask * loss_scale * 0.5 * self.MSELoss(pred_wh, true_wh))  # 这是MSE

            # 正样本损失+负样本损失
            loss_conf = torch.sum(object_mask * self.BCELoss(pred_conf, object_mask)) + \
                        torch.sum(neg_mask * self.BCELoss(pred_conf, object_mask))
            loss_cls = torch.sum(object_mask * self.BCELoss(pred_class, y_true[i][..., 5:]))

            losses.append(loss_xy + loss_wh + loss_conf + loss_cls)

            # 计算正样本数量
            # if self.normalize:
            #     num_pos = torch.sum(object_mask)
            #     num_pos = torch.max(num_pos, torch.ones_like(num_pos))
            # else:
            #     num_pos = batch_size / 3
            #
            # all_pos += num_pos
        last_loss = sum(losses)
        # last_loss = sum(losses) / all_pos

        return last_loss

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from codes.darknet import darknet53


def conv_bn_leaky(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out,
                           kernel_size=kernel_size, stride=(1, 1), padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    '''
    后处理,5次卷积+2次输出结果处理
    :param x:
    :param filters_list:
    :param init_filters: 模块最初的卷积输入通道
    :param ch_out:
    :return:
    '''
    m = nn.Sequential(
        conv_bn_leaky(in_filters, filters_list[0], 1),
        conv_bn_leaky(filters_list[0], filters_list[1], 3),
        conv_bn_leaky(filters_list[1], filters_list[0], 1),
        conv_bn_leaky(filters_list[0], filters_list[1], 3),
        conv_bn_leaky(filters_list[1], filters_list[0], 1))
    n = nn.Sequential(
        conv_bn_leaky(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter,
                  kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))
    return m, n


class YoloBody(nn.Module):
    def __init__(self, anchors, num_classes):
        super(YoloBody, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.backbone = darknet53()

        # out_filters : [64, 128, 256, 512, 1024]

        ch0, ch1, ch2 = [len(self.anchors) // 3 * (5 + self.num_classes)] * 3
        out_filters = self.backbone.layers_out_filters

        self.layer_x0, self.layer_y0 = make_last_layers([512, 1024], out_filters[-1], ch0)

        self.layer_conv1 = conv_bn_leaky(512, 256, 1)
        self.up_sample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_x1, self.layer_y1 = make_last_layers([256, 512], out_filters[-2] + 256, ch1)

        self.layer_conv2 = conv_bn_leaky(256, 128, 1)
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_x2, self.layer_y2 = make_last_layers([128, 256], out_filters[-3] + 128, ch2)

    # todo 从今以后,模型类只写__init__和forward两个方法,其他都写成外面的函数
    def forward(self, inputs):
        '''
        :param inputs: 模型输入
        :param anchors: np.array,(9,2)
        :param num_classes: coco=80
        :return:
        '''
        # 52,26,13
        feat2, feat1, feat0 = self.backbone(inputs)

        # 第一个特征
        out0_branch = self.layer_x0(feat0)
        out0 = self.layer_y0(out0_branch)

        # 第二个特征层
        x1_in = self.layer_conv1(out0_branch)
        x1_in = self.up_sample1(x1_in)

        x1_in = torch.cat([x1_in, feat1], dim=1)
        out1_branch = self.layer_x1(x1_in)
        out1 = self.layer_y1(out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.layer_conv2(out1_branch)
        x2_in = self.up_sample2(x2_in)
        x2_in = torch.cat([x2_in, feat2], dim=1)

        out2_branch = self.layer_x2(x2_in)
        out2 = self.layer_y2(out2_branch)

        return out0, out1, out2

    # def forward(self, inputs):
    #     '''
    #
    #     :param inputs: 模型输入
    #     :param anchors: np.array,(9,2)
    #     :param num_classes: coco=80
    #     :return:
    #     '''
    #     # 52,26,13
    #     feat3, feat2, feat1 = self.backbone(inputs)
    #     out_filters = self.backbone.layers_out_filters
    #     # 计算每个分支的输出通道数,共三个特征层,每个特征层anchors适量一样,都是3个
    #     ch1, ch2, ch3 = [len(self.anchors) // 3 * (5 + self.num_classes)] * 3
    #     x1, y1 = make_last_layers(feat1, [512, 1024], out_filters[-1], ch1)
    #
    #     x2_in = conv_bn_leaky(512, 256, 1)(x1)
    #     x2_in = self.up_sample(x2_in)
    #     x2_in = torch.cat([x2_in, feat2], dim=1)  # 在通道维度合并
    #     x2, y2 = make_last_layers(x2_in, [256, 512], out_filters[-2] + 256, ch2)
    #
    #     # 在52尺度上处理
    #     x3_in = conv_bn_leaky(256, 128, 1)(x2)
    #     x3_in = self.up_sample(x3_in)
    #     x3_in = torch.cat([x3_in, feat3], dim=1)
    #     x3, y3 = make_last_layers(x3_in, [128, 256], out_filters[-3] + 128, ch3)
    #
    #     del x1, x2, x2_in, x3, x3_in, feat1, feat2, feat3
    #
    #     return y1, y2, y3

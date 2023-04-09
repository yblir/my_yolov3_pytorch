import os
import time
import colorsys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

from codes.yolo3_model_raw import YoloBody
from codes.util_func import decode_pred_boxes, correct_boxes, get_anchors, get_classes

root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class YOLO(object):
    _defaults = {
        "model_path": os.path.join(root, 'setting', 'yolo_weights.pth'),
        "anchors_path": os.path.join(root, 'setting', 'yolo_anchors.txt'),
        "classes_path": os.path.join(root, 'setting', 'coco_classes.txt'),
        "input_shape": np.array((416, 416)),  # h,w
        "confidence": 0.5,
        "iou": 0.3,
        "cuda": False,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": False,
    }

    def __init__(self):
        self.__dict__.update(self._defaults)
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.color = self._set_color()
        self.model = self._load_model()

    def _load_model(self):
        model = YoloBody()
        print('loading weights into state dict ...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            model.load_state_dict(torch.load(self.model_path, map_location=device))
            # model = model.cuda()
            model = model.eval()
        except Exception as e:
            raise ValueError('loading weights failure')

        return model

    def _set_color(self):
        '''
        设置绘制的边框颜色
        :return:
        '''
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        # *x: 解包(10,1.,1,.)这样的结构
        color = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # [(12,233,9),(...),(...)]  # 每个小元组就是一个rgb色彩值
        color = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color))
        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(color)
        return color

    def boxes_and_scores(self, feats, num_classes, input_shape, image_shape):
        '''
        解码输出结果,获得预测框和对应得分
        :return:
        '''
        num_layers = len(feats)
        all_boxes, all_scores = [], []
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        for i in range(num_layers):
            box_xy, box_wh, box_conf, box_class = decode_pred_boxes(feats[i], self.anchors[anchor_mask[i]],
                                                                    len(self.class_names), input_shape, training=False)
            boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
            box_scores = torch.reshape(box_conf * box_class, [-1, num_classes])

            all_boxes.append(boxes)
            all_scores.append(box_scores)
        all_boxes = torch.cat(all_boxes, dim=0)  # [n,4], 图中所有预测框坐标,大概有1w+
        all_scores = torch.cat(all_scores, dim=0)  # [n,80],图中所有预测框得分,数量与坐标一一对应

        mask = all_scores >= 0.5  # 挑选出得分大于0.5的所有框

        # 取出所有符合得分阈值的框,再进行非极大抑制得到最终的预测框
        res_boxes, res_scores, res_classes = [], [], []
        for c in range(num_classes):
            new_boxes = torch.masked_select(all_boxes, mask[:, c:c + 1])
            new_boxes = torch.reshape(new_boxes, [-1, 4])
            new_scores = torch.masked_select(all_scores[:, c], mask[:, c])

            # 非极大抑制,丢弃iou大于0.5的部分
            nms_index = nms(new_boxes, new_scores, 0.5)
            nms_boxes = new_boxes[nms_index]
            nms_scores = new_scores[nms_index]
            nms_classes = torch.ones_like(nms_scores) * c

            res_boxes.append(nms_boxes)
            res_scores.append(nms_scores)
            res_classes.append(nms_classes)

        res_boxes = torch.cat(res_boxes, dim=0)
        res_scores = torch.cat(res_scores, dim=0)
        res_classes = torch.cat(res_classes, dim=0)

        return res_boxes, res_scores, res_classes

    def predict(self, image):
        '''
        yolo3预测过程
        :param image:
        :return:
        '''
        # 统一转成rgb格式,方式png和灰度出预测出错
        image_rgb = image.convert('RGB')
        image_shape = np.array(np.shape(image_rgb)[0:2])

        # 直接resize识别,不进行不失真resize. resize传入w,h
        img_resize = image_rgb.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)
        # 图片归一化后,再变换通道位置,之后再添加batch_size维度
        input_img = np.transpose(np.array(img_resize, dtype=np.float32) / 255., (2, 0, 1))
        input_img = np.expand_dims(input_img, 0)

        with torch.no_grad():
            img_tensor = torch.from_numpy(input_img)
            outputs = self.model(img_tensor)
            boxes, scores, classes = self.boxes_and_scores(outputs, len(self.class_names),
                                                           self.input_shape, image_shape)

        # 设置字体,pillow 绘图环节
        font = ImageFont.truetype(font='../setting/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 多次画框的次数,根据图片尺寸不同,把框画粗
        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        for i, c in enumerate(classes):
            c = int(c.numpy())
            predicted_class = self.class_names[c]
            score = scores[i]

            top, left, bottom, right = [item.numpy() for item in boxes[i]]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.color[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.color[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

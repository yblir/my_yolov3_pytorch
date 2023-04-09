import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from codes.yolo3_model import YoloBody
from codes.util_func import encode_true_boxes, get_anchors, get_classes
from codes.yolo3_loss import YoloLoss
from codes.data_gen import YoloData, yolo_dataset_collate


def get_lr(optimzer):
    for param in optimzer.param_groups:
        return param['lr']


def split_datasets(batch_s, data_path):
    '''
    划分训练集与测试集
    :param batch_s:
    :param data_path: 训练数据路径
    :return:
    '''
    val_split = 0.1
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    train_data = YoloData(lines[:num_train], input_shape, is_train=True)
    val_data = YoloData(lines[num_train:], input_shape, is_train=False)

    train_gen = DataLoader(train_data, shuffle=True, batch_size=batch_s, num_workers=4,
                           pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
    val_gen = DataLoader(val_data, shuffle=False, batch_size=batch_s, num_workers=4,
                         pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

    return train_gen, val_gen, num_train, num_val


def clc_loss(images, targets, model, yolo3_loss):
    '''
    计算损失函数
    :param images:
    :param targets:
    :param model:
    :param yolo3_loss:
    :return:
    '''
    with torch.no_grad():
        if is_cuda:
            image_tensor = torch.from_numpy(images).cuda()
        else:
            image_tensor = torch.from_numpy(images)
        y_true = encode_true_boxes(targets, anchors, input_shape, num_classes)
    logits = model(image_tensor)
    loss = yolo3_loss(logits, y_true)

    return loss


def fit_one_epoch(model, yolo3_loss, epoch, total_epoch, train_size, val_size, train_gen, val_gen, is_cuda):
    print('start train')
    total_train_loss = 0
    total_val_loss = 0
    model.train()
    with tqdm(total=train_size, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3) as bar:
        for i, (images, box_data) in enumerate(train_gen):
            with torch.no_grad():
                image_tensor = torch.from_numpy(images).cuda() if is_cuda else torch.from_numpy(images)
                true_boxes = encode_true_boxes(box_data, anchors, input_shape, num_classes)
            true_data_boxes = [box_data, true_boxes]  # 已归一化的真实框+已归一化且与输出数据shape相同的真实框
            # 清零梯度,并反向传播
            optimizer.zero_grad()
            logits = model(image_tensor)
            train_loss = yolo3_loss(logits, true_data_boxes)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()

            bar.set_postfix(**{'total_train_loss': total_train_loss / (i + 1), 'learning_rate': get_lr(optimizer)})
            bar.update(1)

    model.eval()
    print('Start Validation')
    with tqdm(total=val_size, desc=f'Epoch {epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3) as bar:
        for i, (images, box_data) in enumerate(val_gen):
            with torch.no_grad():
                image_tensor = torch.from_numpy(images).cuda() if is_cuda else torch.from_numpy(images)
                true_boxes = encode_true_boxes(box_data, anchors, input_shape, num_classes)
            true_data_boxes = [box_data, true_boxes]  # 已归一化的真实框+已归一化且与输出数据shape相同的真实框
            # optimizer.zero_grad()
            val_logits = model(image_tensor)
            val_loss = yolo3_loss(val_logits, true_data_boxes)
            total_val_loss += val_loss.item()

            bar.set_postfix(**{'total_val_loss': total_val_loss / (i + 1)})
            bar.update(1)

    # loss_history.append_loss(total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(total_epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' %
          (total_train_loss / (train_size + 1), total_val_loss / (val_size + 1)))

    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' %
               ((epoch + 1), total_train_loss / (train_size + 1), total_val_loss / (val_size + 1)))


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    is_cuda = False
    normalize = False  # 是否对损失归一化
    input_shape = np.array([416, 416])
    batch_size = 8
    anchors_path = os.path.join(root_path, 'setting', 'yolo_anchors.txt')
    classes_path = os.path.join(root_path, 'setting', 'voc_classes.txt')
    annotation_path = os.path.join(root_path, 'setting', '2007_train.txt')

    anchors = get_anchors(anchors_path)
    classes = get_classes(classes_path)
    num_classes = len(classes)

    train_gen, val_gen, num_train, num_val = split_datasets(batch_size, annotation_path)

    # 创建模型
    model = YoloBody(anchors, num_classes)
    model = model.train()
    if is_cuda:
        # model = torch.nn.DataParallel(model)
        model = model.cuda()
    yolo3_loss = YoloLoss(anchors, input_shape, num_classes)

    # 训练2轮,冻结权重训练+解冻训练
    for i in range(2):
        if i == 0:  # 冻结训练
            lr, param_mode = 1e-3, False
            epoch1, epoch2 = 0, 3  # 左闭右开 [ )
        else:
            lr, param_mode = 1e-4, True
            epoch1, epoch2 = 3, 6
        optimizer = optim.Adam(model.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        # 设置是否解冻
        for param in model.backbone.parameters():
            param.requires_grad = param_mode

        # 设置每个batch的图片数量
        train_size, val_size = num_train // batch_size, num_val // batch_size

        # 排除数量太少的异常情况,若图片数量连一个batch都没有,不训练了
        if train_size == 0:
            raise ValueError('数据集太小,无法训练')

        # 模型训练
        for epoch in range(epoch1, epoch2):
            fit_one_epoch(model, yolo3_loss, epoch, epoch2,
                          train_size, val_size, train_gen, val_gen, is_cuda)
            lr_scheduler.step()

#  HSANet: A HYBRID SELF-CROSS ATTENTION NETWORK FOR REMOTE SENSING CHANGE DETECTION
#  IGARSS 2025
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import optim
import utils.visualization as visual
from utils import data_loader
from torch.optim import lr_scheduler
from tqdm import tqdm
import random
from utils.utils import clip_gradient, adjust_lr
from utils.metrics import Evaluator
from network.HSANet import HSANet


import time

start = time.time()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, net, criterion, optimizer, num_epoches):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    epoch_loss = 0
    net.train(True)

    length = 0
    st = time.time()
    for i, (A, B, mask) in enumerate(tqdm(train_loader)):
        A = A.cuda()
        B = B.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()
        preds = net(A, B)
        loss = criterion(preds[0], Y) + criterion(preds[1], Y)
        # ---- loss function ----
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()

        output = F.sigmoid(preds[1])
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        pred = output.data.cpu().numpy().astype(int)
        target = Y.cpu().numpy().astype(int)

        Eva_train.add_batch(target, pred)

        length += 1
    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (epoch, num_epoches, train_loss, IoU, Pre, Recall, F1))
    print("Strat validing!")

    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A, B)[1]
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_val.add_batch(target, pred)

            length += 1
    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    new_iou = IoU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        torch.save(best_net, save_path + '_best_iou.pth')
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    vis.close_summary()


if __name__ == '__main__':
    seed_everything(42) # 设定随机种子


    save_path = './output/'
    data_name = 'LEVIR'
    model_name = 'HSANet'
    batchsize = 8
    trainsize = 256
    lr = 5e-4
    epoch = 50

    save_path = save_path + data_name + '/' + model_name
    if data_name == 'LEVIR':
        train_root = 'D:\\study\\datasets\\CD_datasets\\LEVIR\\train\\'
        val_root = 'D:\\study\\datasets\\CD_datasets\\LEVIR\\val\\'
    elif data_name == 'WHU':
        train_root = '/data/users/xuyichu/Code-RS/Data_HAN/WHU/train/'
        val_root = '/data/users/xuyichu/Code-RS/Data_HAN/WHU/val/'
    elif data_name == 'SYSU':
        train_root = '/data/users/xuyichu/Code-RS/Data_HAN/SYSU/train/'
        val_root = '/data/users/xuyichu/Code-RS/Data_HAN/SYSU/val/'
    elif data_name == 'S2Looking':
        train_root = '/data/users/xuyichu/Code-RS/Data_HAN/S2Looking/train/'
        val_root = '/data/users/xuyichu/Code-RS/Data_HAN/S2Looking/val/'

    # 使用自定义的 data_loader 模块加载训练/验证数据，返回 DataLoader 实例，供训练使用。
    train_loader = data_loader.get_loader(train_root, batchsize, trainsize, num_workers=2, shuffle=True,
                                          pin_memory=True)
    val_loader = data_loader.get_test_loader(val_root, batchsize, trainsize, num_workers=2, shuffle=False,
                                             pin_memory=True)
    # 初始化两个评估器，用于训练与验证阶段评估模型性能（比如 IoU、Precision、Recall 等）。
    Eva_train = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)

    # 构建模型，创建 HSANet 模型并移动到 GPU 上。
    if model_name == 'HSANet':
        model = HSANet().cuda()

    # 使用 二分类的带 Logits 的交叉熵损失，适合输出未经过 Sigmoid 的二分类输出。
    criterion = nn.BCEWithLogitsLoss().cuda()

    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    # base_optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    # 加入了 weight decay 的 Adam 优化器。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0025)
    # CosineAnnealingWarmRestarts: 余弦退火策略 + 周期性重启，有利于模型更好收敛。
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = data_name
    best_iou = 0.0

    print("Start train...")
    # args = parser.parse_args()

    for epoch in range(1, epoch):
        print(f"Epoch {epoch} started")  # 添加调试输出
        start_time = time.time()
        # 打印当前学习率（因为学习率会变化）。
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch} learning rate : {param_group['lr']}")
        # cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # 每一轮训练前重置评估器状态。
        Eva_train.reset()
        Eva_val.reset()
        # 调用训练函数 train(...) 执行一次完整的训练与验证过程（这个函数应在其他文件中定义）。
        train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, model, criterion, optimizer,
              epoch)
        # 更新学习率（执行调度器）
        lr_scheduler.step()

        # 当前epoch执行所使用的时间
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} 耗时: {epoch_time:.2f}秒")
        # print('现在的数据是：', args.data_name)


end = time.time()
print('程序训练train的时间为:', end - start)
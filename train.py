import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from net.MyCOD import Network
from utils.dataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from net.IB_Loss import IBLoss
from torch.optim.lr_scheduler import LambdaLR

file = open("log/CODIB.txt", "a")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

mae = nn.L1Loss(reduction='mean')
IB_Loss = IBLoss(temperature=1, reduction='mean')

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch):
    model.train()

    loss_record0, loss_record1, loss_record2, loss_record3, loss_record4, IB_loss_record,mae_loss_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, wavelet_gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        wavelet_gts = Variable(wavelet_gts).cuda()
        # ---- forward ----
        out, out1, out2, out3, out4, z1, z2, wavelet = model(images)
        # ---- loss function ----
        loss4 = structure_loss(out4, gts)
        loss3 = structure_loss(out3, gts)
        loss2 = structure_loss(out2, gts)
        loss1 = structure_loss(out1, gts)
        loss0 = structure_loss(out, gts)
        mae_loss = mae(wavelet, wavelet_gts)
        IB_loss1 = IB_Loss(z1, out4, gts)
        IB_loss2 = IB_Loss(z2, out1, gts)
        IB_loss = 0.05*(IB_loss1 + IB_loss2)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + IB_loss + mae_loss
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----
        loss_record0.update(loss0.data, opt.batchsize)
        loss_record1.update(loss1.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record4.update(loss4.data, opt.batchsize)
        IB_loss_record.update(IB_loss.data, opt.batchsize)
        mae_loss_record.update(mae_loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-0: {:.4f}], [lateral-1: {:.4f}], [lateral-2: {:.4f}], [lateral-3: {:.4f}], [lateral-4: {:.4f}], [IB: {:,.4f}], [mae: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record0.avg, loss_record1.avg, loss_record2.avg, loss_record3.avg, loss_record4.avg, IB_loss_record.avg, mae_loss_record.avg))
            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-0: {:.4f}], [lateral-1: {:.4f}], [lateral-2: {:.4f}], [lateral-3: {:.4f}], [lateral-4: {:.4f}], [IB: {:,.4f}], [mae: {:,.4f}]\n'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record0.avg, loss_record1.avg, loss_record2.avg, loss_record3.avg, loss_record4.avg, IB_loss_record.avg, mae_loss_record.avg))
    scheduler.step()
    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    # if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.epoch:
    torch.save(model.state_dict(), save_path + 'CODIB-%d.pth' % epoch)
    print('[Saving Snapshot:]', save_path + 'CODIB-%d.pth' % epoch)
    file.write('[Saving Snapshot:]' + save_path + 'CODIB-%d.pth' % epoch + '\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=384, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='CODIB')
    opt = parser.parse_args()

    # ---- build models ----
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = Network(64)
    model = nn.DataParallel(model.cuda(), device_ids=[0])


    params = model.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: (1 -ep / (opt.epoch)) ** 0.9)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    wavelet_gt_root = '{}/Wavelet_gt/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, wavelet_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)

    file.close()

import torch
import numpy as np
# from thop import profile
# from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay


def poly_lr(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    lr = init_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def warmup_poly(optimizer, init_lr, curr_iter, max_iter):
    warm_start_lr = 1e-7
    warm_steps = 1000

    if curr_iter<= warm_steps:
        warm_factor = (init_lr / warm_start_lr) ** (1 / warm_steps)
        warm_lr = warm_start_lr * warm_factor ** curr_iter
        for param_group in optimizer.param_groups:
            param_group['lr'] = warm_lr
    else:
        lr = init_lr * (1 - (curr_iter - warm_steps) / (max_iter - warm_steps)) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))



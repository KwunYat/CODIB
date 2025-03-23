import torch
import torch.nn as nn

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()

class VGG(nn.Module):
    # pooling layer at the front of block
    def __init__(self,):
        super(VGG, self).__init__()

        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 32, 3, 1, 1))
        conv1.add_module('bn1_1', nn.BatchNorm2d(32))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(32, 32, 3, 2, 1))
        conv1.add_module('bn1_2', nn.BatchNorm2d(32))
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))

        self.conv1 = conv1
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.MaxPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(32, 64, 3, 1, 1))
        conv2.add_module('bn2_1', nn.BatchNorm2d(64))
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv2.add_module('bn2_2', nn.BatchNorm2d(64))
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv3.add_module('bn3_1', nn.BatchNorm2d(128))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv3.add_module('bn3_2', nn.BatchNorm2d(128))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(128, 128, 3, 1, 1))
        conv3.add_module('bn3_3', nn.BatchNorm2d(128))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3_1', nn.MaxPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv4.add_module('bn4_1', nn.BatchNorm2d(256))
        conv4.add_module('relu4_1', nn.ReLU())
        conv4.add_module('conv4_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv4.add_module('bn4_2', nn.BatchNorm2d(256))
        conv4.add_module('relu4_2', nn.ReLU())
        conv4.add_module('conv4_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv4.add_module('bn4_3', nn.BatchNorm2d(256))
        conv4.add_module('relu4_3', nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4', nn.MaxPool2d(2, stride=2))
        conv5.add_module('conv5_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv5.add_module('bn5_1', nn.BatchNorm2d(512))
        conv5.add_module('relu5_1', nn.ReLU())
        conv5.add_module('conv5_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('bn5_2', nn.BatchNorm2d(512))
        conv5.add_module('relu5_2', nn.ReLU())
        conv5.add_module('conv5_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('bn5_2', nn.BatchNorm2d(512))
        conv5.add_module('relu5_3', nn.ReLU())
        self.conv5 = conv5
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

    def initialize(self):
        weight_init(self)
 
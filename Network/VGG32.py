import torch
import torch.nn as nn
from Network.ABN import MultiBatchNorm, weights_init_ABN, weights_init


class VGG32(nn.Module):
    def __init__(self, num_classes=10):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)

    def forward(self, x, return_feature=False):
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        if return_feature:
            return x, y
        else:
            return y


class VGG32ABN(nn.Module):
    def __init__(self, num_classes=10, num_abn=2):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = MultiBatchNorm(64, num_abn)
        self.bn2 = MultiBatchNorm(64, num_abn)
        self.bn3 = MultiBatchNorm(128, num_abn)

        self.bn4 = MultiBatchNorm(128, num_abn)
        self.bn5 = MultiBatchNorm(128, num_abn)
        self.bn6 = MultiBatchNorm(128, num_abn)

        self.bn7 = MultiBatchNorm(128, num_abn)
        self.bn8 = MultiBatchNorm(128, num_abn)
        self.bn9 = MultiBatchNorm(128, num_abn)
        self.bn10 = MultiBatchNorm(128, num_abn)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init_ABN)
        # self.cuda()

    def forward(self, x, return_feature=False, bn_label=None):
        if bn_label is None:
           # bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long).cuda()
            bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long)
        x = self.dr1(x)
        x = self.conv1(x)
        x, _ = self.bn1(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x, _ = self.bn2(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x, _ = self.bn3(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x, _ = self.bn4(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x, _ = self.bn5(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x, _ = self.bn6(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x, _ = self.bn7(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x, _ = self.bn8(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x, _ = self.bn9(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        if return_feature:
            return x, y
        else:
            return y

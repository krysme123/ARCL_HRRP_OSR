import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, apply_complex
from complexPyTorch.complexLayers import NaiveComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu, complex_avg_pool2d
from torch.nn import Conv1d
from torch.nn import functional as f
import argparse


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_):
        return apply_complex(self.conv_r, self.conv_i, input_)


def complex_avg_pool1d(input_, *args, **kwargs):        # Perform complex average pooling.
    absolute_value_real = f.adaptive_avg_pool1d(input_.real, *args, **kwargs)
    absolute_value_imag = f.adaptive_avg_pool1d(input_.imag, *args, **kwargs)
    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(torch.complex64)


class ComplexCNN1D(nn.Module):
    def __init__(self, **options):
        super(self.__class__, self).__init__()
        self.num_classes = options['num_classes']
        self.feature_dim = options['feature_dim']

        self.conv1 = ComplexConv1d(1, 8, (9,), (1,), 4)
        # self.bn1 = ComplexBatchNorm1d(8)          # 这个原代码写的有问题
        self.bn1 = NaiveComplexBatchNorm1d(8)
        self.conv2 = ComplexConv1d(8, 16, (7,), (1,), 3)
        self.bn2 = NaiveComplexBatchNorm1d(16)

        self.conv3 = ComplexConv1d(16, 32, (5,), (1,), 2)
        self.bn3 = NaiveComplexBatchNorm1d(32)

        self.lin1 = ComplexLinear(32 * 32, self.feature_dim)
        self.lin2 = ComplexLinear(self.feature_dim, self.num_classes)

    def forward(self, x, return_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        print(x.shape)
        x = complex_avg_pool1d(x, 128)
        x = complex_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = complex_avg_pool1d(x, 64)
        x = complex_relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = complex_avg_pool1d(x, 32)
        x = complex_relu(x)

        x = torch.flatten(x, 1)
        x = self.lin1(x)
        y = self.lin2(x)
        # ############### 虽然是复数网络，但是最后给的输出，还是得取模值给出一个实数输出才行
        y = torch.sqrt(torch.pow(y.real, 2) + torch.pow(y.imag, 2))
        x = torch.sqrt(torch.pow(x.real, 2) + torch.pow(x.imag, 2))
        if return_feature:
            return x, y
        else:
            return y


class ComplexVGG32(nn.Module):
    def __init__(self, **options):
        super(self.__class__, self).__init__()
        self.num_classes = options['num_classes']
        self.feature_dim = options['feature_dim']

        self.conv1 = ComplexConv2d(1,       64,     3, 1, 1, bias=False)
        self.conv2 = ComplexConv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = ComplexConv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = ComplexConv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = ComplexConv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = ComplexConv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = ComplexConv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = ComplexConv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = ComplexConv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = ComplexBatchNorm2d(64)
        self.bn2 = ComplexBatchNorm2d(64)
        self.bn3 = ComplexBatchNorm2d(128)

        self.bn4 = ComplexBatchNorm2d(128)
        self.bn5 = ComplexBatchNorm2d(128)
        self.bn6 = ComplexBatchNorm2d(128)

        self.bn7 = ComplexBatchNorm2d(128)
        self.bn8 = ComplexBatchNorm2d(128)
        self.bn9 = ComplexBatchNorm2d(128)
        self.bn10 = ComplexBatchNorm2d(128)

        self.fc = ComplexLinear(self.feature_dim, self.num_classes)
        # self.dr1 = ComplexDropout2d(0.2).to(self.device)          # 这个库里自带的代码有问题
        # self.dr2 = ComplexDropout2d(0.2).to(self.device)
        # self.dr3 = ComplexDropout2d(0.2).to(self.device)

    def forward(self, x, return_feature=False):
        # x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = complex_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = complex_relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = complex_relu(x)

        # x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = complex_relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = complex_relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = complex_relu(x)

        # x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = complex_relu(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = complex_relu(x)

        x = self.conv9(x)
        x = self.bn9(x)
        x = complex_relu(x)

        x = complex_avg_pool2d(x, 32, 32)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        # ############### 虽然是复数网络，但是最后给的输出，还是得取模值给出一个实数输出才行
        y = torch.sqrt(torch.pow(y.real, 2) + torch.pow(y.imag, 2))
        if return_feature:
            return x, y
        else:
            return y



parser = argparse.ArgumentParser("Open Set Recognition")

if __name__ == '__main__':
    net = ComplexCNN1D()
    print(net)

    args = parser.parse_args()
    options_ = vars(args)
    options_['num_classes'] = 3
    options_['feature_dim'] = 128
    if torch.cuda.is_available():
        options_['device'] = torch.device('cuda')
        options_['use_gpu'] = True
        print("Currently using Cuda.\n")
    elif torch.backends.mps.is_available():
        options_['device'] = torch.device('mps')
        options_['use_gpu'] = True
        print("Currently using MPS.\n")
    else:
        options_['device'] = torch.device('cpu')
        options_['use_gpu'] = False
        print("Currently using CPU!")
    net = ComplexVGG32(**options_)
    print(net)

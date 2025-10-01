import torch
import torch.nn as nn
from Network.ABN import weights_init


class _netD32(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD32, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(ndf * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output).flatten()

        return output


class _netG32(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG32, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


def Generator32(n_gpu, nz, ngf, nc):
    model = _netG32(n_gpu, nz, ngf, nc)
    model.apply(weights_init)
    return model


def Discriminator32(n_gpu, nc, ndf):
    model = _netD32(n_gpu, nc, ndf)
    model.apply(weights_init)
    return model


class _netD64(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_netD64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(ndf * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output).flatten()

        return output


class _netG64(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(_netG64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


def Generator64(n_gpu, nz, ngf, nc):
    model = _netG64(n_gpu, nz, ngf, nc)
    model.apply(weights_init)
    return model


def Discriminator64(n_gpu, nc, ndf):
    model = _netD64(n_gpu, nc, ndf)
    model.apply(weights_init)
    return model


class _Discriminator1D(nn.Module):
    def __init__(self, n_channels):
        super(_Discriminator1D, self).__init__()
        self.main = nn.Sequential(
            # input size. [N, 1, 256]
            nn.Conv1d(in_channels=1, out_channels=2*n_channels, kernel_size=(9,), stride=(1,), padding=4),
            nn.BatchNorm1d(2*n_channels),
            # nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(128),
            # nn.ReLU()
            nn.LeakyReLU(0.2, inplace=True),
            # state size. [N, 2, 128]

            nn.Conv1d(in_channels=2*n_channels, out_channels=4*n_channels, kernel_size=(7,), stride=(1,), padding=3),
            nn.BatchNorm1d(4*n_channels),
            # nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. [N, 4, 64]

            nn.Conv1d(in_channels=4*n_channels, out_channels=4*n_channels, kernel_size=(5,), stride=(1,), padding=2),
            nn.BatchNorm1d(4*n_channels),
            # nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. [N, 4, 32]

            nn.Conv1d(in_channels=4*n_channels, out_channels=4*n_channels, kernel_size=(5,), stride=(1,), padding=2),
            nn.Sigmoid()
            # state size. [N, 4, 32]
        )
        self.avgpool = nn.AdaptiveAvgPool1d(32)
        self.classifier = nn.Sequential(
            nn.Linear(4*n_channels * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, input_):
        output = self.main(input_)
        # output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output).flatten()
        return output


class _Generator1D(nn.Module):
    def __init__(self, n_channel, output_channel=1):
        super(_Generator1D, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution, z: [N, n_channel, L]
            nn.ConvTranspose1d(n_channel, n_channel * 4, (4,), (2,), (1,), bias=False),
            nn.BatchNorm1d(n_channel * 4),
            nn.ReLU(True),
            # state size. [N, 4*n_channel, 2*L]
            nn.ConvTranspose1d(n_channel * 4, n_channel * 2, (4,), (2,), (1,), bias=False),
            nn.BatchNorm1d(n_channel * 2),
            nn.ReLU(True),
            # state size. [N, 2*n_channel, 4*L]
            nn.ConvTranspose1d(n_channel * 2, n_channel, (4,), (2,), (1,), bias=False),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(True),
            # state size. [N, n_channel, 8*L]
            nn.ConvTranspose1d(n_channel, 8, (4,), (2,), (1,), bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            # state size. [N, 8, 16*L]
            nn.ConvTranspose1d(8, output_channel, (3,), (1,), (1,), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_):
        return self.main(input_)


def Generator1D(n_c, output_channels=1):
    model = _Generator1D(n_c, output_channels)
    model.apply(weights_init)
    return model


def Discriminator1D(n_c):
    model = _Discriminator1D(n_c)
    model.apply(weights_init)
    return model


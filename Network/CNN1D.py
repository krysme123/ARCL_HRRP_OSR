import torch
import torch.nn as nn
from Network.ABN import MultiBatchNorm1D


class CNN1D(nn.Module):
    def __init__(self, num_classes=6, init_channels=1, feature_dim=128):  # 网络结构试了这么多次，还就这个效果最好，服了。。。
        super(CNN1D, self).__init__()
        self.cnn = torch.nn.Sequential(
            nn.Conv1d(in_channels=init_channels, out_channels=8, kernel_size=(9,), stride=(1,), padding=4),
            nn.BatchNorm1d(8),
            # nn.MaxPool1d(2),
            # nn.AdaptiveMaxPool1d(128),
            nn.AdaptiveAvgPool1d(128),
            # nn.LeakyReLU(0.2),
            nn.ReLU(),
            # nn.PReLU(),

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=(7,), stride=(1,), padding=3),
            nn.BatchNorm1d(16),
            # nn.MaxPool1d(kernel_size=2),
            # nn.AdaptiveMaxPool1d(64),
            nn.AdaptiveAvgPool1d(64),
            # nn.LeakyReLU(0.2),
            nn.ReLU(),
            # nn.PReLU(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(5,), stride=(1,), padding=2),
            nn.BatchNorm1d(32),
            # nn.MaxPool1d(2),
            # nn.AdaptiveMaxPool1d(32),
            nn.AdaptiveAvgPool1d(32),
            # nn.LeakyReLU(0.2),
            nn.ReLU()
            # nn.PReLU(),
        )
        self.linear1 = nn.Linear(32 * 32, feature_dim)
        self.linear2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_feature=False):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # 这个view的作用就是拉直x，成为一列
        x = self.linear1(x)
        feature = nn.ReLU()(x)
        x = self.linear2(feature)
        if return_feature:
            return feature, x
        else:
            return x


class CNN1DABN(nn.Module):
    def __init__(self, num_classes, num_abn=2, init_channels=1):
        super(CNN1DABN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=init_channels, out_channels=8, kernel_size=(9,), stride=(1,), padding=4)
        self.bn1 = MultiBatchNorm1D(8, num_abn)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=(7,), stride=(1,), padding=3)
        self.bn2 = MultiBatchNorm1D(16, num_abn)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(5,), stride=(1,), padding=2)
        self.bn3 = MultiBatchNorm1D(32, num_abn)
        self.linear1 = nn.Linear(32 * 32, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x, return_feature=False, bn_label=None):
        if bn_label is None:
            bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long).cuda()
        x = self.conv1(x)
        x, _ = self.bn1(x, bn_label)
        x = nn.AdaptiveAvgPool1d(128)(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x, _ = self.bn2(x, bn_label)
        x = nn.AdaptiveAvgPool1d(64)(x)
        x = nn.ReLU()(x)

        x = self.conv3(x)
        x, _ = self.bn3(x, bn_label)
        x = nn.AdaptiveAvgPool1d(32)(x)
        x = nn.ReLU()(x)

        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        feature = nn.ReLU()(x)
        x = self.linear2(feature)
        if return_feature:
            return feature, x
        else:
            return x

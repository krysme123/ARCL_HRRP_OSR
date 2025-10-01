import torch.nn as nn
import torch.nn.functional as f
from Loss.LabelSmoothing import smooth_cross_entropy_loss
import torch
from Loss.Dist import Dist


class AdapRingLoss(nn.Module):
    def __init__(self, **options):
        super(AdapRingLoss, self).__init__()
        self.label_smoothing = options['label_smoothing']
        self.weight_pl = float(options['lambda'])
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.device = options['device']
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feature_dim'])
        self.points = self.Dist.centers

    def forward(self, x, y, labels=None):
        # logits = f.softmax(y, dim=1)
        # if labels is None:
        #     return logits, 0

        # if not self.label_smoothing:
        #     loss = f.cross_entropy(y, labels)
        # else:
        #     loss = smooth_cross_entropy_loss(y, labels=labels, smoothing=self.label_smoothing, dim=-1)

        # ##################################### 第一种思路LeNet验证了不太行
        # o_center = x.mean(0).repeat(x.shape[0], 1)
        # loss_r = f.mse_loss(self.radius, (x-o_center).pow(2).sum(dim=1).mean(0, keepdim=True))
        # ######################################################################################

        # ########################## 第二种思路，LeNet验证了也不是很行。。。
        # dist = self.Dist(x)
        # logits = f.softmax(-dist, dim=1)
        dist_l2_p = self.Dist(x, center=self.points)  # formula (6):d_e

        o_center = torch.mean(self.Dist.centers, dim=0) #计算所有类别原型的均值，作为全局中心
        # x - o_center.repeat(x.size(0), 1)：将输入特征 x 的每个样本减去全局中心 o_center，相当于将样本特征平移到以 o_center 为原点的坐标系。
        # center=self.points - o_center.repeat(self.Dist.num_classes, 1)：将类别原型 self.points 也减去全局中心 o_center，相当于将类别原型平移到以 o_center 为原点的坐标系。
        dist_dot_p = self.Dist(x-o_center.repeat(x.size(0), 1),
                               center=self.points-o_center.repeat(self.Dist.num_classes, 1),
                               metric='dot')  # formula (6):d_d
        logits = dist_l2_p - dist_dot_p  # [batch_size,num_classes]
        if labels is None:
            return logits, 0

        # if not self.label_smoothing:
        #     loss = f.cross_entropy(-dist, labels)
        # else:
        #     loss = smooth_cross_entropy_loss(-dist, labels=labels, smoothing=self.label_smoothing, dim=-1)

        if not self.label_smoothing:
            loss = f.cross_entropy(logits, labels)
        else:
            loss = smooth_cross_entropy_loss(logits, labels=labels, smoothing=self.label_smoothing, dim=-1)

        # o_center = self.Dist.centers.mean(0).repeat(x.shape[0], 1)
        # loss_r = f.mse_loss(self.radius, (x - o_center).pow(2).sum(dim=1).mean(0, keepdim=True))
        # #################################################################################################

        # ########################## 第三种思路，
        # o_center = torch.mean(self.Dist.centers, dim=0)
        _dis_known = (x - o_center.repeat(x.size(0), 1)).pow(2).sum(1)

        target = torch.ones(_dis_known.size()).to(self.device)
        loss_r1 = self.margin_loss(-self.radius, -_dis_known, target )
        loss_r2 = self.margin_loss(-_dis_known, -self.radius, target)
        loss_r = loss_r1 + loss_r2

        loss = loss + self.weight_pl * loss_r
        return logits, loss

    def normalized_logits(self, x):
        # dist = self.Dist(x)
        # logits = f.softmax(-dist, dim=1)
        dist_l2_p = self.Dist(x, center=self.points)  # formula (6):d_e

        o_center = torch.mean(self.Dist.centers, dim=0)
        dist_dot_p = self.Dist(x-o_center.repeat(x.size(0), 1),
                               center=self.points-o_center.repeat(self.Dist.num_classes, 1),
                               metric='dot')  # formula (6):d_d
        logits = dist_l2_p - dist_dot_p  # [batch_size,num_classes]
        # return logits
        return f.softmax(logits, dim=1)

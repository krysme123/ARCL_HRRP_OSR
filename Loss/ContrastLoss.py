import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_contrast=0.1, temperature=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature
        self.register_buffer('class_centers', torch.zeros(num_classes, feature_dim))  # 非可学习参数

    def forward(self, logits, features, labels):
        # 交叉熵损失
        ce_loss = F.cross_entropy(logits, labels)

        # 更新特征中心（动量更新）
        with torch.no_grad():
            for idx in torch.unique(labels):
                mask = (labels == idx)
                if mask.sum() > 0:
                    curr_center = features[mask].mean(dim=0)
                    self.class_centers[idx] = 0.9 * self.class_centers[idx] + 0.1 * curr_center

        # 计算归一化相似度矩阵
        centers_norm = F.normalize(self.class_centers, p=2, dim=1)
        sim_matrix = torch.mm(centers_norm, centers_norm.T)  # [num_classes, num_classes]
        sim_matrix = (sim_matrix + 1) / 2  # 归一化到[0,1]

        # 对比损失项（温度缩放）
        probs = F.softmax(logits / self.temperature, dim=1)
        batch_sim = sim_matrix[labels]  # [batch_size, num_classes]
        contrast_loss = -torch.log(1 - probs * batch_sim + 1e-6).mean()

        return ce_loss + self.lambda_contrast * contrast_loss
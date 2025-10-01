import numpy as np
import torch
from typing import List
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import FinerWeightedTarget

# Finer-CAM: https://arxiv.org/pdf/2501.11309
# FinerCAM类的初始化部分接收模型、目标层等参数，并创建了一个基础CAM实例。
# forward方法是核心，处理输入张量，计算激活和梯度，然后生成CAM。

class FinerCAM:
    def __init__(self, model, target_layers, reshape_transform=None, base_method=GradCAM):
        self.base_cam = base_method(model, target_layers, reshape_transform)
        self.compute_input_gradient = self.base_cam.compute_input_gradient
        self.uses_gradients = self.base_cam.uses_gradients

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                target_size=None,
                eigen_smooth: bool = False,
                alpha: float = 1,
                comparison_categories: List[int] = [1, 2, 3],
                target_idx: int = None, # konwn的索引（0-4）
                H: int = None,
                W: int = None
                ) -> np.ndarray:

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        # 1.：通过base_cam（如Grad-CAM）的activations_and_grads方法，提取目标层的激活值（activations）和对应梯度（gradients）。
        outputs = self.base_cam.activations_and_grads(input_tensor, H, W)

        main_categories = []
        comparisons = []

        if targets is None:
            if isinstance(outputs, (list, tuple)):
                output_data = outputs[0].detach().cpu().numpy()
            else:
                output_data = outputs.detach().cpu().numpy()

            # 权重相似性排序：是在训练好的线性分类器上计算各个类别权重之间的余弦相似度，从而找到每个目标类别最相似的几个类别作为参考
            sorted_indices = np.empty_like(output_data, dtype=int)
            # Sort indices based on similarity to the target logit,
            # with more similar values (smaller differences) appearing first.
            for i in range(output_data.shape[0]):
                # target_logit为主类别（概率最大的哪个） ,当target_idx为None时，选择当前样本预测的最高分类别；否则，直接使用target_idx指定的类别。
                target_logit = output_data[i][np.argmax(output_data[i])] if target_idx is None else output_data[i][target_idx]
                # 排序依据：通过计算输出logit与目标logit的绝对差异，按差异从小到大排序：
                differences = np.abs(output_data[i] - target_logit)
                sorted_indices[i] = np.argsort(differences)

            targets = []
            for i in range(sorted_indices.shape[0]):
                # 差异越小，表示与目标logit越相似。sorted_indices的第0位（sorted_indices[i, 0]）是最相似的主类别（即预测结果）
                main_category = int(sorted_indices[i, 0])
                # 这里遍历comparison_categories中的索引值，从已排序的类别列表sorted_indices中提取对应的类别编号。
                # 例如，默认参数会选取第2、3、4相似的类别作为对比类。
                current_comparison = [int(sorted_indices[i, idx]) for idx in comparison_categories]
                main_categories.append(main_category)
                comparisons.append(current_comparison)
                # 生成的target将主类别与对比类组合，并通过alpha参数调节对比类的权重：
                target = FinerWeightedTarget(main_category, current_comparison, alpha)
                targets.append(target)

        if self.uses_gradients:
            # 梯度反向传播
            self.base_cam.model.zero_grad()
            if isinstance(outputs, (list, tuple)):
                loss = sum([target(output) for target, output in zip(targets, outputs)])
            else:
                loss = sum([target(output) for target, output in zip(targets, [outputs])])
            loss.backward(retain_graph=True)

        # cam_per_layer应该是一个列表，每个元素对应一个目标层的CAM结果
        cam_per_layer = self.base_cam.compute_cam_per_layer(
            input_tensor, targets, target_size, eigen_smooth
        )

        return self.base_cam.aggregate_multi_layers(cam_per_layer), outputs, main_categories, comparisons
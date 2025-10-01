
'''1. Score-CAM 的工作原理，Score-CAM 是一种 基于类激活映射（CAM）的可视化方法，其核心流程如下：
    前向传播：输入图像通过模型，得到预测结果和目标层的激活图（feature maps）。
    计算权重：通过遮挡（masking）输入图像的局部区域，观察模型输出的变化，计算每个激活图对预测结果的贡献（即“得分”）。
    生成热力图：将激活图按得分加权叠加，归一化后生成热力图。
    整个过程仅涉及模型的前向计算，不涉及反向传播或参数更新。
'''
import argparse
import os
import glob
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from Dataset.HRRP_datasets import HRRPOSRDataImage
from Network.VGG32 import VGG32ABN

from cam.scorecam import ScoreCAM
from skimage import io
from PIL import Image
import torch.nn.functional as f

parser = argparse.ArgumentParser("Open Set Recognition")
def preprocess_image(image_path):
    """预处理函数：返回模型输入张量 + 原始尺寸图像"""
    # 读取原始图像（保持原尺寸）
    raw_img = io.imread(image_path)  # 单通道灰度图，shape=(H, W)

    # 生成模型输入（调整到64x64）
    img_pil = Image.fromarray(raw_img).convert("RGB")
    transform_heat = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    input_tensor = transform_heat(img_pil)  # [3,64,64]

    return input_tensor.unsqueeze(0), raw_img  # [1,3,64,64], [H,W]

def generate_explainable_heatmap(options,model, input_tensor, raw_img, idx_to_class):
    """
    生成热力图
    参数:
        model: 加载的VGG16模型
        input_tensor: 预处理后的张量[1,1,224,224]
        raw_img: 原始单通道图像[224,224]
        class_to_idx: 类别映射字典(已删除）
    返回:
        heatmap: 热力图矩阵
        pred_idx: 预测数字标签
        confidence: 置信度(0-1)
    """
    model.eval()
    # 1. 模型预测
    with torch.no_grad():
        features, logits = model(input_tensor, return_feature=True)
        # probs = torch.nn.functional.softmax(logits, dim=1)
        temperature = 2.0  # 实验调整此值（>1.0 软化分布，<1.0 锐化分布）
        probs = f.softmax(logits / temperature, dim=1)
        print("输出概率2:", probs)
        # 预测标签转换
        pred_idx = logits.argmax().item()
        print('预测标签是：',pred_idx)
        pred_raw_class = idx_to_class[pred_idx]  # 转换为原始类编号（如 3）
        class_name_str = str(pred_raw_class)  # 将整数转换为字符串
        class_name = options['class_name'].get(class_name_str, 'Unknown')
        confidence = probs[0, pred_idx].item()
    # 2. 热力图生成（保持原有ScoreCAM调用）
    cam_extractor = ScoreCAM({
        'type': 'vgg32abn',
        'arch': model,
        'layer_name': 'conv9',  # 最后一层卷积层
        # 'target_layer': target_layer,  # 直接传入层对象2
        'input_size': (input_tensor.shape[2:])   # 自动获取输入尺寸
    })
    heatmap = cam_extractor(input_tensor).squeeze().cpu().numpy()

    # 关键修改：将热力图插值到原图尺寸
    orig_h, orig_w = raw_img.shape[:2]  # 获取原图尺寸
    heatmap = cv2.resize(
        heatmap,
        (orig_w, orig_h),  # OpenCV尺寸顺序为(width, height)
        interpolation=cv2.INTER_LINEAR  # 双线性插值
    )
    print("热力图范围:", heatmap.min(), heatmap.max())  # 应为 [0, 1]
    # 3. 可视化增强
    plt.figure(figsize=(15, 5))

    # 原始图像
    plt.subplot(131)
    plt.imshow(raw_img, cmap='gray')
    plt.title("Original Image", fontsize=9)
    plt.axis('off')

    # 纯热力图
    plt.subplot(132)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Activation Heatmap", fontsize=9)
    plt.colorbar()
    plt.axis('off')

    # 叠加效果
    plt.subplot(133)
    # 转换单通道热力图到伪彩色
    heatmap_uint8 = (heatmap * 255).astype('uint8')
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 转换原图为RGB三通道（适配彩色叠加）
    if len(raw_img.shape) == 2:  # 灰度图转RGB
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
    else:
        raw_img_rgb = raw_img.copy()

    # 确保数据类型一致
    raw_img_rgb = raw_img_rgb.astype('uint8')

    # 叠加
    overlay = cv2.addWeighted(
        raw_img_rgb, 0.7,
        heatmap_colored, 0.3, 0
    )
    plt.imshow(overlay, cmap='jet')
    plt.title(f"Predicted: {class_name}\nConfidence: {confidence:.2%}", fontsize=9)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"Heatmap/vgg32ABN_scorecam_{class_name}_{confidence:.2%}_score.png", bbox_inches='tight', dpi=150)
    plt.close()

    return heatmap, class_name, confidence

# 使用示例
if __name__ == "__main__":

    args = parser.parse_args()
    options = vars(args)
    options['class_name'] = {'0': 'A319', '1': 'A320', '2': 'A321', '3': 'A330-2', '4': 'A330-3',
                  '5': 'A350-941', '6': 'B737-7', '7': 'B737-8', '8': 'B747-89L', '9': 'CRJ-900',
                  '10': 'An-26', '11': 'Cessna', '12': 'Yak-42'}
    options['data_dir'] = 'F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/'
    options['known'] = [3, 0, 7, 6, 10]

    # 1. 加载模型和类别
    checkpoint = torch.load('F:/研究生/开集识别/abc/Results_HRRP2D/VGG32ABN_Softmax/2025-05-22-23-40-36/network/VGG32ABN_Softmax_best_auc.pth', map_location='cpu')
    model = VGG32ABN(num_classes=len(options['known']))
    print([name for name, _ in model.named_modules()])
    model.load_state_dict(checkpoint)  # checkpoint 本身就是 state_dict
    idx_to_class = {i: cls for i, cls in enumerate(options['known'])}

    # 2. 加载测试集图片
    class_name = options['class_name']
    data_dir = options["data_dir"]
    # 一. 使用已知类别进行测试热力图
    # options['data_root'] = [data_dir + class_name[str(j)] + '/2D_test_real' for j in options['known']]
    # transform_ = transforms.Compose([transforms.Resize((64, 64)),
    #                                  transforms.ToTensor()])
    # dataset = HRRPOSRDataImage(options['data_root'], transform_)
    #
    # index_ = 2100
    # image_index = dataset.images[index_]  # 根据索引index获取该图片
    # img_path = os.path.join(dataset.all_dir[index_], image_index)  # 获取索引为index的图片的路径名
    # print("img_path:", img_path)

    # 二. 使用未知类别进行测试 热力图
    options['unknown'] = list(set(range(len(class_name))) - set(options['known']))
    options['data_root'] = [data_dir + class_name[str(j)] + '/2D_test_real' for j in options['unknown']]

    transform_ = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])
    dataset = HRRPOSRDataImage(options['data_root'], transform_)
    index_ = 9
    image_index = dataset.images[index_]  # 根据索引index获取该图片
    img_path = os.path.join(dataset.all_dir[index_], image_index)  # 获取索引为index的图片的路径名
    print("img_path:", img_path)

    # 3. 预处理图像
    input_tensor, raw_img = preprocess_image(img_path)

    model.eval()  # 切换到评估模式
    # 4. 生成可解释热力图
    heatmap, class_name, conf = generate_explainable_heatmap(options, model, input_tensor, raw_img, idx_to_class)
    print(f"生成完成！类别: {class_name}, 置信度: {conf:.2%}")

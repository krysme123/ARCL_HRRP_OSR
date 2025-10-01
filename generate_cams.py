import os
import difflib
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from Dataset.HRRP_datasets import HRRPOSRDataImage
from Network.VGG32 import VGG32ABN
from pytorch_grad_cam import FinerCAM, GradCAM
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from skimage import io

class ModifiedDINO(nn.Module):
    """
    A wrapper for the original DINO model that adds a classifier layer.
    """
    def __init__(self, original_model, classifier_path, num_classes, feature_dim=768):
        super(ModifiedDINO, self).__init__()
        self.original_model = original_model
        self.classifier = nn.Linear(feature_dim, num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=device)
        )
        self.blocks = self.original_model.blocks
        print("ModifiedDINO initialized")

    def forward(self, x):
        features = self.original_model.forward_features(x)["x_norm_patchtokens"]
        features = features.mean(dim=1)
        logits = self.classifier(features)
        return logits


class ModifiedVGGABN(nn.Module):
    def __init__(self, original_model, classifier_path, num_classes):
        super().__init__()
        self.feature_extractor = original_model  # 原始VGG32ABN
        self.classifier = self._build_classifier(classifier_path, num_classes)

        # 冻结特征提取器参数
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)

        print("ModifiedVGGABN Initialized")

    def _build_classifier(self, path, num_classes):
        """验证分类器维度一致性"""
        assert self.feature_extractor.fc.out_features == num_classes, \
            f"分类器输出维度{self.feature_extractor.fc.out_features}与num_classes不匹配"

        # 直接使用原始分类器
        classifier = self.feature_extractor.fc
        classifier.load_state_dict(torch.load(path))
        return classifier

    def forward(self, x):
        """改造前向传播以获取中间特征"""
        # 获取最后一个卷积层的输出
        feature_maps = self._get_last_conv_features(x)

        # 全局平均池化
        features = nn.functional.adaptive_avg_pool2d(feature_maps, (1, 1))
        features = torch.flatten(features, 1)

        return self.classifier(features)

    def _get_last_conv_features(self, x):
        """提取目标层特征（conv9的输出）"""
        # 通过hook获取中间层输出
        feature_maps = None

        def hook(module, input, output):
            nonlocal feature_maps
            feature_maps = output.detach()

        handle = self.feature_extractor.conv9.register_forward_hook(hook)
        self.feature_extractor(x)  # 执行完整前向传播
        handle.remove()

        return feature_maps

def get_image_paths_from_folder(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths


# 替换reshape_transform
def vgg_reshape_transform(tensor, *args):
    """将conv9的输出转换为标准特征图格式"""
    # 输入形状: [batch, 128, height, width]
    return tensor  # 无需额外变换


def get_true_label_idx(class_name, class_names):
    #用于处理类别名称可能存在拼写差异或大小写不一致的情况，确保正确获取目标类别的索引。
    closest_match = difflib.get_close_matches(class_name, class_names, n=1, cutoff=0.8)
    if closest_match:
        return class_names.index(closest_match[0])
    return None

def get_key_from_value(dictionary, value):
    # 遍历字典，找到值对应的键
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def get_index_in_known(known_list, target_key):
    # 遍历已知列表，找到目标键的索引
    if target_key in known_list:
        return known_list.index(target_key)
    return None

def preprocess(image, patch_size=14, max_size=1000):
    '''
    对输入的图像进行预处理。首先将图像转换为RGB模式，然后根据max_size调整图像大小，确保最长边不超过该值。
    接着计算新的高度和宽度，使其为patch_size的整数倍，这可能是因为模型（如ViT）需要将图像分割为固定大小的patch。
    然后应用一系列的变换，包括调整大小、转换为Tensor和归一化。
    返回处理后的图像张量以及网格的高度和宽度，这些网格参数可能与后续的特征图生成有关。
    '''
    image = image.convert("RGB")
    width, height = image.size

    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        width = int(width * scale)
        height = int(height * scale)
        image = image.resize((width, height), Image.BICUBIC)

    new_height_pixels = int(np.ceil(height / patch_size) * patch_size)
    new_width_pixels = int(np.ceil(width / patch_size) * patch_size)

    transform = Compose([
        Resize((new_height_pixels, new_width_pixels), interpolation=Image.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                  std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    image_tensor = transform(image).to(torch.float32)
    
    grid_height = new_height_pixels // patch_size
    grid_width = new_width_pixels // patch_size
    
    return image_tensor, grid_height, grid_width


def run_finer_cam_on_dataset(opts,image_list, cam, preprocess, save_path, device):
    """
    Run FinerCAM on a dataset of images.
    这个函数是主流程的执行部分。首先根据dataset_path获取所有图像的路径，然后遍历每张图像。
    对于每张图像，获取其类别名称，通过get_true_label_idx找到对应的目标索引，然后进行预处理。
    接着，对三种模式（Baseline、Finer-Default、Finer-Compare）分别调用CAM生成方法，得到不同模式下的热力图。
    最后将结果保存为.npy文件，包含高分辨率的热力图、主类别和对比类别信息。
    """
    os.makedirs(save_path, exist_ok=True)
    known = opts['known']
    class_names =opts['class_names']
    # Baseline是没有对比目标类与相似类来抑制共享特征的传统方法，Finer-Default是默认设置，可能对比多个相似类，而Finer-Compare可能只对比最相似的类。
    modes = ["Baseline", "Finer-Default", "Finer-Compare"]

    for img_path in tqdm(image_list): # 例如 img_path =  F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/An-26/2D_test_real\\2D_test_An-26_999.jpg
        # 1. 路径解析
        image_filename = os.path.basename(img_path) # 获取图片名 2D_test_An-26_997.jpg
        class_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))  # 获取文件名（类名） An-26
        base_name = os.path.splitext(image_filename)[0] # 截取图片名称 2D_test_An-26_997
        new_filename = f"{class_name}_{base_name}.npy"  # 拼接 新文件名 An-26_2D_test_An-26_997.npy

        # 2. 图像预处理
        raw_img = io.imread(img_path)  # 单通道灰度图，shape=(H, W)
        # 生成模型输入（调整到64x64）
        img_pil = Image.fromarray(raw_img).convert("RGB")
        original_width, original_height = img_pil.size

        transform_heat = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image_tensor = transform_heat(img_pil)  # [3,64,64]

        key = get_key_from_value(class_names, class_name)
        if key is not None:
            key = int(key)  # 将键从字符串转换为整数
            # 获取键在 known 列表中的索引(0-4)
            target_idx = get_index_in_known(known, key)
            # if target_idx is not None:
            #     print(f"键 {key} 在 known 列表中的索引是 {target_idx}")
            # else:
            #     print(f"键 {key} 不在 known 列表中")
        else:
            print(f"类别名称 {class_name} 不在 class_names 字典中")

        image_tensor = image_tensor.unsqueeze(0).to(device)

        results_by_mode = {}
        for mode in modes:
            # When alpha = 0, FinerCAM degrades to Baseline
            if mode == "Baseline":
                grayscale_cam, model_outputs, main_category, comparison_categories = cam(
                    input_tensor=image_tensor,
                    targets = None,
                    target_idx=target_idx,
                    H=8, # 输入64x64的图像，池化前的特征图尺寸压缩到 8x8
                    W=8,
                    alpha=0
                )
            elif mode == "Finer-Default":
            # Our default setting: compare with the three most similar categories
                grayscale_cam, model_outputs, main_category, comparison_categories = cam(
                    input_tensor=image_tensor,
                    targets = None,
                    target_idx=target_idx,
                    H=8,
                    W=8,
                    comparison_categories=[1,2,3] #这些数字代表的是在已知类别列表 (known) 中的相对索引，索引从0开始
                )
            elif mode == "Finer-Compare":
            # Compare only with the most similar category
                grayscale_cam, model_outputs, main_category, comparison_categories = cam(
                    input_tensor=image_tensor,
                    targets = None,
                    target_idx=target_idx,
                    H=8,
                    W=8,
                    comparison_categories=[1]
                )

            grayscale_cam = grayscale_cam[0, :]
            # grayscale_cam_highres = cv2.resize(grayscale_cam, (original_width, original_height))
            grayscale_cam_highres = cv2.resize(grayscale_cam, (224, 224))
            # 计算置信度
            logits = model_outputs.detach().cpu()
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
            predicted_idx = logits.argmax(dim=1).item()  # 预测类别在已知列表中的索引
            original_class_id = opts['known'][predicted_idx]  # 转换为原始类别ID
            predicted_class = opts['class_names'][str(original_class_id)]  # 获取类别名称
    
            results_by_mode[mode] = {
                "highres": np.array([grayscale_cam_highres], dtype=np.float16),
                "main_category": main_category,
                "comparison_categories": comparison_categories,
                "confidence": float(probs[predicted_idx] * 100),  # 置信度
                "predicted_class": predicted_class  # 新增预测类别
            }

        np.save(os.path.join(save_path, new_filename), results_by_mode)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Finer-CAM on a dataset')
    parser.add_argument('--classifier_path', type=str, required=False,
                        help='Path to the classifier model')
    parser.add_argument('--dataset_path', type=str, required=False,
                        help='Path to the validation set')
    parser.add_argument('--save_path', type=str, required=False,
                        help='Directory to save FinerCAM results')
    args = parser.parse_args()

    options = vars(args)

    options['class_names'] = {'0': 'A319', '1': 'A320', '2': 'A321', '3': 'A330-2', '4': 'A330-3',
                             '5': 'A350-941', '6': 'B737-7', '7': 'B737-8', '8': 'B747-89L', '9': 'CRJ-900',
                             '10': 'An-26', '11': 'Cessna', '12': 'Yak-42'}
    options['data_dir'] = 'HRRP_13_pre_results/'
    options['known'] = [3, 0, 7, 6, 10]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型和类别
    checkpoint = torch.load(
        './Results_HRRP2D/VGG32ABN_Softmax/2025-05-22-23-40-36/network/VGG32ABN_Softmax_best_auc.pth', map_location='cpu')
    model = VGG32ABN(num_classes=len(options['known']),return_feature=True)
    print([name for name, _ in model.named_modules()])
    model.load_state_dict(checkpoint)
    idx_to_class = {i: cls for i, cls in enumerate(options['known'])}

    model = model.to(device)

    # target_layers = [model.blocks[-1].norm1]
    # 选择最后一个卷积层（conv9）作为CAM目标层
    target_layers = [model.conv9]
    cam = FinerCAM(model=model, target_layers=target_layers,
                   reshape_transform=vgg_reshape_transform, base_method= GradCAM)

    # 2. 加载测试集图片
    class_names = options['class_names']
    data_dir = options["data_dir"]
    ###################################### 已知类 热力图 ###############################
    # options['save_path'] = 'results/known_cams'
    # options['data_root'] = [data_dir + class_names[str(j)] + '/2D_test_real' for j in options['known']]
    # transform_ = transforms.Compose([transforms.Resize((64, 64)),
    #                                  transforms.ToTensor()])
    # dataset = HRRPOSRDataImage(options['data_root'], transform_)
    ##################################################################################

    options['save_path'] = 'results/unknown_cams'
    options['unknown'] = list(set(range(len(class_names))) - set(options['known']))
    options['data_root'] = [data_dir + class_names[str(j)] + '/2D_test_real' for j in options['unknown']]
    
    transform_ = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor()])
    dataset = HRRPOSRDataImage(options['data_root'], transform_)

    img_path_list = []
    for i in range(len(dataset.images)):
        # print(os.path.join(dataset.all_dir[i], dataset.images[i]))
        img_path_list.append(os.path.join(dataset.all_dir[i], dataset.images[i])) # 获取已知类别 所以测试图片路径

    # print("img_path:", img_path_list)

    run_finer_cam_on_dataset(options,img_path_list, cam, preprocess, options['save_path'], device)
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse

from Dataset.HRRP_datasets import HRRPOSRDataImage



def extract_core_name_from_image_path(img_path):
    # img_path（图像文件路径，如'F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/A330-2/2D_test_real\\2D_test_A330-2_0.jpg）。
    # 输出：字符串（如A330-2_2D_test_A330-2_0）
    dir_name = os.path.dirname(img_path)
    class_name = os.path.basename(os.path.dirname(dir_name))
    file_name = os.path.basename(img_path)
    index = os.path.splitext(file_name)[0]
    return f"{class_name}_{index}"

def load_and_preprocess_image(img_path):
    """ 与generate_cams预处理保持一致的图像加载逻辑 """
    # 1. 保持与CAM生成时相同的读取方式
    raw_img = Image.open(img_path).convert("RGB")

    # 2. 应用相同的尺寸调整 (64x64)
    transform_heat = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform_heat(raw_img)

    # 3. 转换为OpenCV处理需要的格式 (HWC, BGR, [0,255])
    image_np = image_tensor.numpy().transpose(1, 2, 0) * 255
    image_np = image_np[:, :, ::-1].astype(np.uint8)  # RGB->BGR

    # 4. 后续可视化需要float32类型和[0,1]范围
    return image_np.astype(np.float32) / 255.0


def calculate_top_ratio(cam, sigma=1.0):
    """基于标准差计算显著激活区域占比"""
    cam = cam.astype(np.float32)
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 计算动态阈值（均值+2倍标准差）
    mean = np.mean(cam_normalized)
    std = np.std(cam_normalized)
    threshold = mean + sigma * std

    # 计算显著区域占比
    significant_ratio = np.mean(cam_normalized >= threshold)
    # 在calculate_top_ratio函数内添加：
    print(f"Mean: {mean:.3f}, Std: {std:.3f}, Threshold: {threshold:.3f}, Ratio: {significant_ratio:.3f}")
    return significant_ratio
def visualize_cam_on_image(image_bgr, cam):
    cam = cam.astype(np.float32)
    # print(f"image_bgr shape: {image_bgr.shape}, cam shape: {cam.shape}")
    # 确保热力图尺寸与输入图像一致
    # image_bgr = cv2.resize(image_bgr,(cam.shape[1], cam.shape[0]))
    # cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
    # print(f"image_bgr shape: {image_bgr.shape}, cam shape: {cam.shape}")

    # 将原始图像和热力图叠加
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # 使用修正后的计算方法
    significant_ratio = calculate_top_ratio(cam, sigma=1.0)
    cam = np.squeeze(cam)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = (image_rgb * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    visualization_rgb = cv2.addWeighted(image_rgb, 0.5, heatmap_rgb, 0.5, 0)
    visualization_bgr = cv2.cvtColor(visualization_rgb, cv2.COLOR_RGB2BGR)
    return visualization_bgr,significant_ratio

def get_image_paths_from_folder(folder_path):
    # 递归遍历文件夹，收集所有图像文件路径。
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def add_label_to_image(image, label, padding_height=40):
    # 在图像顶部添加标签文本。在可视化结果中标注图像含义。
    label_img = np.full((padding_height, image.shape[1], 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = max((image.shape[1] - text_size[0]) // 2, 0)
    text_y = (padding_height + text_size[1]) // 2 -5
    cv2.putText(label_img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    return np.vstack((label_img, image))


def main(args):
    os.makedirs(args['save_path'], exist_ok=True)
    image_paths = args['img_path_list']

    # 1.按类别组织图像路径
    class_to_image_paths = {}
    for path in image_paths:
        class_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        # normalized_class_name = normalize_label(class_name)
        # class_to_image_paths.setdefault(normalized_class_name, []).append(path)
        class_to_image_paths.setdefault(class_name, []).append(path)
    # class_to_image_paths.setdefault('A330-2', []).append('F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/A330-2/2D_test_real\\2D_test_A330-2_0.jpg')
    # class_to_image_paths.setdefault('A319', []).append('F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/A319/2D_test_real\\2D_test_A319_0.jpg')
    # class_to_image_paths.setdefault('B737-8', []).append('F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/B737-8/2D_test_real\\2D_test_B737-8_0.jpg')
    # class_to_image_paths.setdefault('B737-7', []).append('F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/B737-7/2D_test_real\\2D_test_B737-7_0.jpg')
    # class_to_image_paths.setdefault('An-26', []).append('F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/An-26/2D_test_real\\2D_test_An-26_0.jpg')
    # 2. 遍历每张图像
    for img_path in tqdm(image_paths, desc="Processing images"):
        # 提取核心名称（core_name）
        core_name = extract_core_name_from_image_path(img_path)
        # 加载并预处理原始图像。
        original_image = load_and_preprocess_image(img_path)
        original_image_class_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))

        vis_list = []
        labels = []
        vis_list.append((original_image * 255).astype(np.uint8))
        # labels.append(f"Original Image:{original_image_class_name}")
        labels.append(f"{original_image_class_name}")
        # 加载对应的CAM数据（.npy文件）：cam_path = 'results/cams\\A330-2_2D_test_A330-2_0.npy'
        # cam_dict是通过加载cam_path路径的.npy文件得到的字典，其中包含不同CAM方法（如Baseline、Finer-Default、Finer-Compare）的结果
        cam_path = os.path.join(args['cams_path'], f"{core_name}.npy")
        cam_dict = np.load(cam_path, allow_pickle=True).item()

        # 3.处理对比类别
        closest_comparison_idx = None
        for key in ["Baseline", "Finer-Default", "Finer-Compare"]:
            if key in cam_dict:
                # 从CAM数据中提取comparison_categories。
                outputs = cam_dict[key]
                # 找到最接近的对比类别索引。
                comparison_categories = outputs.get("comparison_categories",None)
                closest_comparison_idx = comparison_categories[0][0]
                if closest_comparison_idx is not None:
                    break
        # 4. 根据索引获取对比类别的图像路径，加载并预处理对比图像。
        #检查索引是否在合法范围内，若是则从（类别名称列表）中获取对应名称，否则标记为Unknown。
        if closest_comparison_idx is not None and 0 <= closest_comparison_idx < len(args['known']):
            # Step 1: 获取 known 列表中索引3对应的原始类别ID
            original_class_id = args['known'][closest_comparison_idx]  # 这里得到整数6

            # Step 2: 将原始类别ID转换为字符串，查询class_names
            class_k_label = class_names[str(original_class_id)]  # 查询 class_names['6']

        else:
            class_k_label = "Unknown"

        # normalized_class_k_label = normalize_label(class_k_label)
        second_img_path = None
        if class_k_label in class_to_image_paths:
            # 这是挑选 相似类
            candidate_paths = class_to_image_paths[class_k_label]
            if len(candidate_paths) > 1:
                # 选的 相似类列表第一张图片
                second_img_path = next((p for p in candidate_paths if p != img_path), candidate_paths[0])
            elif len(candidate_paths) == 1:
                second_img_path = candidate_paths[0]

        # Load second image and resize it to match the original image dimensions
        second_image = load_and_preprocess_image(second_img_path)
        if second_image.shape[:2] != original_image.shape[:2]:
            second_image = cv2.resize(second_image, (original_image.shape[1], original_image.shape[0]))

        vis_list.append((second_image * 255).astype(np.uint8))
        # labels.append(f"comparison_categories: {class_k_label}")
        labels.append(f" {class_k_label}")

        for key in ["Baseline", "Finer-Default", "Finer-Compare"]:
            if key in cam_dict:
                outputs = cam_dict[key]
                confidence = outputs.get("confidence", 0.0) 
                predicted_class = outputs.get("predicted_class", "Unknown")
                # "highres"应该是指高分辨率的CAM数据。通常，CAM生成时可能会有不同的分辨率版本，例如有些方法生成的CAM可能分辨率较低，
                # 而"highres"可能指的是经过上采样或其他处理后的高分辨率版本，以便与原始图像尺寸匹配，方便可视化。
                cams = outputs.get("highres", None)
                if cams is None:
                    continue
                cam = cams[0].squeeze()
                # cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))  # 调整热力图尺寸与预处理后的图像一致(64x64)

                visualization = visualize_cam_on_image(original_image, cam)
                vis_list.append(visualization)
                labels.append(f"{key} {confidence:.1f}")  # 换行显示

        # Add labels to images
        images_with_labels = [add_label_to_image(img, label) for img, label in zip(vis_list, labels)]

        # Resize images to have the same height
        margin_size = 10
        margin_color = (255, 255, 255)
        target_height = images_with_labels[0].shape[0]
        for i, img in enumerate(images_with_labels):
            if img.shape[0] != target_height:
                images_with_labels[i] = cv2.resize(img, (img.shape[1], target_height))

        # Concatenate images horizontally with a margin between them
        concatenated_image = images_with_labels[0]
        for img in images_with_labels[1:]:
            margin = np.full((concatenated_image.shape[0], margin_size, 3), margin_color, dtype=np.uint8)
            concatenated_image = np.hstack((concatenated_image, margin, img))

        output_filename = f"{core_name}_concatenated.jpg"
        output_path = os.path.join(args['save_path'], output_filename)
        cv2.imwrite(output_path, concatenated_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the dataset directory")
    parser.add_argument("--cams_path", type=str, required=False, help="Path to the CAMs directory")
    parser.add_argument("--save_path", type=str, required=False, help="Path to save visualizations")
    args = parser.parse_args()

    options = vars(args)

    options['class_names'] = {'0': 'A319', '1': 'A320', '2': 'A321', '3': 'A330-2', '4': 'A330-3',
                              '5': 'A350-941', '6': 'B737-7', '7': 'B737-8', '8': 'B747-89L', '9': 'CRJ-900',
                              '10': 'An-26', '11': 'Cessna', '12': 'Yak-42'}
    options['data_dir'] = 'F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/'
    options['known'] = [3, 0, 7, 6, 10]

    # 2. 加载测试集图片
    class_names = options['class_names']
    data_dir = options["data_dir"]
    ###################################### 已知类 热力图 ###############################
    options['cams_path'] = 'results/known_cams'
    options['save_path'] = 'results/known_visualization2'
    options['data_root'] = [data_dir + class_names[str(j)] + '/2D_test_real' for j in options['known']]
    transform_ = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor()])
    dataset = HRRPOSRDataImage(options['data_root'], transform_)
    ##################################################################################

    # options['cams_path'] = 'results/unknown_cams'
    # options['save_path'] = 'results/unknown_visualization'
    # options['unknown'] = list(set(range(len(class_names))) - set(options['known']))
    # options['data_root'] = [data_dir + class_names[str(j)] + '/2D_test_real' for j in options['unknown']]
    #
    # transform_ = transforms.Compose([transforms.Resize((224,224)),
    #                                  transforms.ToTensor()])
    # dataset = HRRPOSRDataImage(options['data_root'], transform_)

    img_path_list = []
    for i in range(len(dataset.images)):
        # print(os.path.join(dataset.all_dir[i], dataset.images[i]))
        img_path_list.append(os.path.join(dataset.all_dir[i], dataset.images[i]))  # 获取已知类别 所以测试图片路径

    options['img_path_list'] = img_path_list

    main(options)
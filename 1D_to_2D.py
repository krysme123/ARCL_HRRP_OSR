# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 加载.npy文件
# data = np.load('F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/A319/test_data_real_A319.npy')  # 替换为你的.npy文件路径
#
# # 选择一个特定的样本（例如第一个样本）
# selected_sample = data[0]
#
# # 绘制折线图
# plt.figure(figsize=(6, 4))  # 设置图像大小
# plt.plot(selected_sample, color='#1184c8ff')  # 绘制折线图，蓝色线条
# plt.xlabel('Range cell')  # 设置x轴标签
# plt.ylabel('Amplitude')  # 设置y轴标签
# # plt.title('A319')  # 添加标题
# plt.grid(True)  # 添加网格线
# plt.axis([0, len(selected_sample), 0, 0.5])  # 设置坐标轴范围（根据数据调整）
#
# # 保存图像
# plt.savefig('1d_hrrp_image.png', dpi=300, bbox_inches='tight')
#
# # 显示图像
# plt.show()
#
#
# import cv2
# image = cv2.imread('F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/A319/2D_test_real/2D_test_A319_0.jpg', cv2.IMREAD_GRAYSCALE)
# # 生成网格坐标轴
# plt.figure(figsize=(6, 4))
# plt.imshow(image, cmap='gray', extent=[0, 250, 0, 250])  # 强制映射到0-250范围
# plt.xlabel('Range cell', fontsize=12)
# plt.ylabel('Frequency cell', fontsize=12)
# # 保存图像
# plt.savefig('2d_hrrp_image.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib.patches import Circle
# # import matplotlib
# # matplotlib.use('TkAgg')  # 强制使用Tkinter后端
# # import matplotlib.pyplot as plt
# #
# # # 模拟数据
# # np.random.seed(42)
# # class_center = np.array([3, 3])  # 类别原型 o^c
# # samples = class_center + np.random.randn(20, 2) * 0.8  # 样本点
# #
# # # 计算双距离
# # def calc_logits(x, center):
# #     d_l2 = np.linalg.norm(x - center)
# #     d_dot = np.dot(x, center) / (np.linalg.norm(x)*np.linalg.norm(center))
# #     return d_l2 - d_dot  # logits_c
# #
# # # 绘制
# # plt.figure(figsize=(10, 8))
# # ax = plt.gca()
# #
# # # 1. 绘制样本点与类中心
# # plt.scatter(class_center[0], class_center[1], s=200, c='red', marker='*', label='Class Prototype $o^c$')
# # plt.scatter(samples[:,0], samples[:,1], s=80, c='blue', alpha=0.7, label='Samples')
# #
# # # 2. 绘制距离向量（几何约束）
# # for sample in samples:
# #     plt.plot([sample[0], class_center[0]], [sample[1], class_center[1]],
# #              'gray', linestyle='--', alpha=0.4)
# #     # 标注L2距离
# #     mid_point = (sample + class_center) / 2
# #     plt.annotate(f'$d_{{L2}}={np.linalg.norm(sample-class_center):.2f}$',
# #                 xy=mid_point, xytext=(5,-5), textcoords='offset points')
# #
# # # 3. 绘制方向约束（点积距离）
# # angles = np.linspace(0, 2*np.pi, 100)
# # circle = np.array([np.cos(angles), np.sin(angles)]).T * 0.5 + class_center
# # plt.plot(circle[:,0], circle[:,1], 'green', label='Direction Constraint')
# #
# # # 4. 添加理论公式标注
# # plt.text(0.5, 0.95, r'$logits_c = d_{L2}(x,o^c) - d_{dot}(x,o^c)$',
# #          transform=ax.transAxes, fontsize=14, ha='center',
# #          bbox=dict(facecolor='yellow', alpha=0.3))
# #
# # plt.xlabel('Feature Dimension 1', fontsize=12)
# # plt.ylabel('Feature Dimension 2', fontsize=12)
# # plt.title('Dual-Distance Optimization Mechanism (Sec 4.1)', fontsize=14)
# # plt.legend()
# # plt.grid(alpha=0.2)
# # plt.show()
#
# # import numpy as np
# #
# #
# # def calculate_top5_percentage(heatmap):
# #     """
# #     计算热力图中前5%高激活区域的面积占比
# #     :param heatmap: 2D numpy数组，已归一化到[0,1]
# #     :return: 比例值(0~1)
# #     """
# #     # 展平热力图并排序
# #     flattened = np.sort(heatmap.flatten())[::-1]  # 降序排列
# #
# #     # 计算前5%的阈值
# #     k = int(len(flattened) * 0.05)
# #     threshold = flattened[k]
# #
# #     # 计算超过阈值的像素占比
# #     mask = (heatmap >= threshold)
# #     return np.sum(mask) / mask.size
# #
# # # 假设heatmap是Finer-CAM输出（示例随机生成）
# # heatmap = np.random.rand(100, 100)  # 100x100的热力图
# # top5_ratio = calculate_top5_percentage(heatmap)
# # print(f"前5%高激活区域占比: {top5_ratio:.2%}")

import matplotlib.pyplot as plt
import numpy as np

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))

# 模拟飞机轨迹数据 (经纬度坐标)
trajectory = np.array([
    [116.3, 39.9],  # 起点
    [116.4, 39.8],
    [116.5, 39.7],
    [116.6, 39.6],
    [116.7, 39.5],  # 终点
])

# 转换为相对坐标 (简化处理)
x = trajectory[:, 0] - trajectory[0, 0]
y = trajectory[:, 1] - trajectory[0, 1]

# 绘制轨迹
ax.plot(x, y, 'b-', linewidth=2, marker='o', markersize=6,
        markerfacecolor='red', markeredgecolor='black')

# 设置图形属性
ax.set_xlabel('经度差 (°)')
ax.set_ylabel('纬度差 (°)')
ax.set_title('飞机飞行轨迹地平面投影')
ax.grid(True, linestyle='--', alpha=0.6)

# 标记起终点
ax.text(x[0], y[0], '  起点', verticalalignment='center')
ax.text(x[-1], y[-1], '  终点', verticalalignment='center')

plt.tight_layout()
plt.savefig('flight_trajectory.png', dpi=300)
plt.close()



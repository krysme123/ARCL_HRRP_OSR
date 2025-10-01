################################################# 1
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 参数设置
# R = 3.0       # 环形半径
# gamma = 0.5   # 边界阈值
# O_center = np.array([0, 0])  # 特征空间中心
#
# # 生成样本点
# np.random.seed(42)
# known_samples = R + gamma * np.random.randn(100, 2)  # 已知类（环形区域）
# unknown_center = 0.8 * R * np.random.randn(20, 2)    # 未知类（中心区域）
# unknown_outer = (R + 1.5*gamma) * np.random.randn(20, 2) # 未知类（外围）
#
# # 绘制
# plt.figure(figsize=(8, 8))
# # 环形区域填充
# circle_inner = plt.Circle(O_center, R-gamma, color='lightgreen', alpha=0.3, label='Known Region')
# circle_outer = plt.Circle(O_center, R+gamma, color='lightgreen', alpha=0.3)
# plt.gca().add_patch(circle_inner)
# plt.gca().add_patch(circle_outer)
# # 样本点
# plt.scatter(known_samples[:,0], known_samples[:,1], c='green', label='Known Samples')
# plt.scatter(unknown_center[:,0], unknown_center[:,1], c='blue', label='Unknown (Center)')
# plt.scatter(unknown_outer[:,0], unknown_outer[:,1], c='red', label='Unknown (Outer)')
# # 环形边界线
# theta = np.linspace(0, 2*np.pi, 100)
# plt.plot(R*np.cos(theta), R*np.sin(theta), 'r-', linewidth=2, label='Radius R')
# plt.plot((R-gamma)*np.cos(theta), (R-gamma)*np.sin(theta), 'k--', alpha=0.5)
# plt.plot((R+gamma)*np.cos(theta), (R+gamma)*np.sin(theta), 'k--', alpha=0.5)
# # 标注
# plt.annotate(f'$R-\\gamma={R-gamma:.1f}$', xy=(0, R-gamma), xytext=(20, -20),
#              textcoords='offset points', arrowprops=dict(arrowstyle='->'))
# plt.annotate(f'$R+\\gamma={R+gamma:.1f}$', xy=(0, R+gamma), xytext=(20, 20),
#              textcoords='offset points', arrowprops=dict(arrowstyle='->'))
# plt.legend()
# plt.title('ARCL Feature Space Constraint')
# plt.savefig('ARCL_Feature_Space_Constraint.png', dpi=300, bbox_inches='tight')
# plt.show()


################################################## 2
# import numpy as np
# import matplotlib
#
#
# matplotlib.use('TkAgg')  # 或 'Qt5Agg'、'Agg'
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.patches import Circle
# import matplotlib.font_manager as fm
#
# # 1. 修复中文显示问题
# try:
#     # 尝试使用系统自带中文字体（Windows）
#     font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
#     font_prop = fm.FontProperties(fname=font_path)
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False
# except:
#     # 备用方案：使用Matplotlib默认字体
#     plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS/Linux
#
# # 2. 参数设置
# R_init = 2.0
# gamma = 0.5
# n_samples = 30
# n_steps = 50
# n_prototypes = 5
#
# # 3. 初始化原型中心
# angles = np.linspace(0, 2 * np.pi, n_prototypes, endpoint=False)
# prototypes_init = np.column_stack([R_init * np.cos(angles), R_init * np.sin(angles)])
#
# # 4. 历史记录
# history = {
#     'R': [],
#     'prototypes': [],
#     'features': [],
#     'inner_bound': [],
#     'outer_bound': []
# }
#
#
# # 5. 优化过程（保持不变）
# def optimize_step(step):
#     progress = min(step / n_steps, 1.0)
#     R_current = R_init * (1 + 0.5 * progress)
#     prototypes_current = prototypes_init * (1 + 0.5 * progress)
#
#     features = []
#     for _ in range(n_samples):
#         angle = np.random.rand() * 2 * np.pi
#         d = R_current + gamma * (2 * np.random.rand() - 1)
#         pos = np.array([d * np.cos(angle), d * np.sin(angle)])
#         nearest_idx = np.argmin(np.linalg.norm(prototypes_current - pos, axis=1))
#         pos += 0.1 * (prototypes_current[nearest_idx] - pos) * progress
#         features.append(pos)
#
#     history['R'].append(R_current)
#     history['prototypes'].append(prototypes_current.copy())
#     history['features'].append(np.array(features))
#     history['inner_bound'].append(R_current - gamma)
#     history['outer_bound'].append(R_current + gamma)
#
#
# # 6. 预计算所有帧
# for step in range(n_steps):
#     optimize_step(step)
#
# # 7. 创建画布
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
# ax1.set_xlim(-4, 4)
# ax1.set_ylim(-4, 4)
# ax1.set_aspect('equal')
# ax1.set_title("ARCL双度量优化（5原型中心）", fontsize=14, pad=20)
#
# # 8. 动态元素初始化
# proto_scatter = ax1.scatter([], [], c='gold', marker='*', s=250, edgecolors='black', label='原型中心')
# sample_scatter = ax1.scatter([], [], c='green', s=60, alpha=0.7, label='样本特征')
# inner_circle = ax1.add_patch(Circle((0, 0), 0, fill=False, ls='--', color='blue', alpha=0.5))
# outer_circle = ax1.add_patch(Circle((0, 0), 0, fill=False, ls='--', color='blue', alpha=0.5))
# R_circle = ax1.add_patch(Circle((0, 0), 0, fill=False, color='red', linewidth=2))
# info_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))
#
# # 9. 右侧R曲线
# ax2.set_xlim(0, n_steps)
# ax2.set_ylim(0, R_init * 1.5)
# ax2.set_xlabel('迭代步数')
# ax2.set_ylabel('半径 R')
# R_line, = ax2.plot([], [], 'r-', linewidth=2)
# gamma_band = ax2.fill_between([], [], [], color='blue', alpha=0.2)
#
#
# # 10. 更新函数
# def update(frame):
#     # 更新主图
#     proto_scatter.set_offsets(history['prototypes'][frame])
#     sample_scatter.set_offsets(history['features'][frame])
#     R_circle.set_radius(history['R'][frame])
#     inner_circle.set_radius(history['inner_bound'][frame])
#     outer_circle.set_radius(history['outer_bound'][frame])
#
#     # 更新R曲线
#     R_line.set_data(range(frame + 1), history['R'][:frame + 1])
#
#     # 替换原错误行
#     for coll in ax2.collections:
#         coll.remove()
#     ax2.fill_between(range(frame + 1),
#                      history['inner_bound'][:frame + 1],
#                      history['outer_bound'][:frame + 1],
#                      color='blue', alpha=0.2)
#
#     # 更新文本
#     info_text.set_text(f'Step: {frame}/{n_steps}\nR = {history["R"][frame]:.2f}\nγ = {gamma:.2f}')
#
#     return [proto_scatter, sample_scatter, R_circle, inner_circle, outer_circle, info_text, R_line]
#
#
# # 11. 保存关键帧图片
# def save_key_frames():
#     # 第一帧
#     update(0)
#     plt.savefig('ARCL_first_frame.png', dpi=300, bbox_inches='tight')
#     print("已保存第一帧: ARCL_first_frame.png")
#
#     # 最后一帧
#     update(n_steps - 1)
#     plt.savefig('ARCL_final_frame.png', dpi=300, bbox_inches='tight')
#     print("已保存最终帧: ARCL_final_frame.png")
#
#
# # 12. 生成动画并保存关键帧
# ani = FuncAnimation(fig, update, frames=n_steps, interval=400, blit=False)
# save_key_frames()  # 保存关键帧
#
# try:
#     plt.tight_layout()
#     plt.show()
# except Exception as e:
#     print(f"显示异常: {e}")
#     ani.save('ARCL_animation.gif', writer='pillow', fps=10)
#     print("已降级保存为ARCL_animation.gif")


import matplotlib.pyplot as plt
plt.scatter([0], [0], marker='*', s=500, c='gold', edgecolors='black', label='原型中心')
plt.scatter(x_samples, y_samples, s=100, c='green', alpha=0.7, label='样本点')
plt.gca().add_patch(plt.Circle((0,0), radius=2.0, fill=False, linestyle='--', color='blue'))
plt.legend()
plt.savefig('prototype.pdf')  # 矢量图导出
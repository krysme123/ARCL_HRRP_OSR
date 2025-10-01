import matplotlib.pyplot as plt
import numpy as np
import os


def plot_distance_distribution(known_distances, unknown_distances, path=None, **options):
    plt.figure(figsize=(8, 6))

    # 绘制已知类别的距离分布直方图
    plt.hist(known_distances, bins=50, alpha=0.5, label='已知类别', color='blue', edgecolor='black')

    # 绘制未知类别的距离分布直方图
    plt.hist(unknown_distances, bins=50, alpha=0.5, label='未知类别', color='orange', edgecolor='black')

    plt.xlabel('特征到空间中心的距离')
    plt.ylabel('频数')
    plt.legend(loc='upper left')
    plt.tight_layout()

    if path:
        dir_path = os.path.join(path, 'image')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(os.path.join(dir_path, f'{options["dataset"]}_{options["loss"]}_distance_distribution.png'))

    plt.show()


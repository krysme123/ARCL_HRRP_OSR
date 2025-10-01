import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os


def plot_feature_distribution(features, labels, criterion, path=None, **options):
    results, prototypes = None, None

    plt.figure()
    # ############ 先绘制特征分布
    if features.shape[1] == 2:        # 2 dimension features
        scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='rainbow', marker='.')
    else:                               # t-SNE features
        print("Using t-SNE!")
        tsne = TSNE(n_components=2, random_state=9)
        if 'Dist' in criterion.__dir__():
            prototypes = criterion.Dist.centers.detach().cpu().numpy()
            inputs = np.concatenate((features, prototypes), axis=0)
        else:
            inputs = features
        results = tsne.fit_transform(inputs)
        scatter = plt.scatter(results[0:features.shape[0], 0], results[0:features.shape[0], 1],
                              c=labels, cmap='rainbow', marker='.', edgecolors='k')

    class_label = list(set(labels))
    legend1 = plt.legend(handles=scatter.legend_elements()[0], labels=class_label, loc='upper right')
    plt.xlabel(r'$\alpha_1$')
    plt.ylabel(r'$\alpha_2$')
    plt.tight_layout()

    # ############ 再添加prototypes
    if 'Dist' in criterion.__dir__():  # 列出对象的所有属性
        if features.shape[1] == 2:        # 2 dimension features
            prototypes = criterion.Dist.centers.detach().cpu().numpy()
            l11 = plt.scatter(prototypes[:, 0], prototypes[:, 1], marker='*', c=class_label, cmap='rainbow',
                              linewidths=1, edgecolors='black', s=70)
            # o_center = prototype.mean(0)
            # plt.scatter(o_center[0], o_center[1], marker='p', c='yellow', linewidths=1, edgecolors='black', s=50)
        else:
            l11 = plt.scatter(results[-prototypes.shape[0]:, 0], results[-prototypes.shape[0]:, 1], marker='*',
                              c=class_label, cmap='rainbow', linewidths=1,
                              edgecolors='black', s=70)
        plt.legend([l11], ['Prototypes'], loc='lower right')
        plt.gca().add_artist(legend1)

    if path:
        dir_path = path + '/image'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(dir_path + '/' + options['dataset'] + '_' + options['loss'] + '_feature.png')

    plt.show()


def plot_diff_feature_distribution(features, labels, known_labels, unknown_labels, criterion, path=None, **options):
    plt.figure()
    # ############ 先绘制特征分布
    if features.shape[1] == 2:  # 2 dimension features
        scatter_known = plt.scatter(features[known_labels, 0], features[known_labels, 1], c='blue', marker='.',
                                    label='已知类别')
        scatter_unknown = plt.scatter(features[unknown_labels, 0], features[unknown_labels, 1], c='orange', marker='o',
                                      label='未知类别')
    else:  # 使用 t-SNE 降维
        print("Using t-SNE!")
        tsne = TSNE(n_components=2, random_state=9, perplexity=min(30, features.shape[0] // 2))  # 确保 perplexity 小于样本数量
        results = tsne.fit_transform(features)
        scatter_known = plt.scatter(results[known_labels, 0], results[known_labels, 1], c='blue', marker='.',
                                    label='Known classes',linewidths=1, edgecolors='black', s=70)
        scatter_unknown = plt.scatter(results[unknown_labels, 0], results[unknown_labels, 1], c='orange', marker='o',
                                      label='UnKnown classes',linewidths=1, edgecolors='black', s=70)

    plt.xlabel(r'$\alpha_1$')
    plt.ylabel(r'$\alpha_2$')
    plt.legend()
    plt.tight_layout()

    if path:
        dir_path = path + '/image'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(dir_path + '/' + options['dataset'] + '_' + options['loss'] + '_feature.png')

    plt.show()
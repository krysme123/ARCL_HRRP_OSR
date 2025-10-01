import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch.nn.functional as f


def plot_feature_boundary(net, features, labels, criterion, save_path=None, **options):
    """
    :param net:             这个一般情况下特指 LeNet 网络，有一个 fc2 全连接层输出 logits
    :param features:
    :param labels:
    :param criterion:
    :param save_path:
    :param options:
    :return:
    """
    plt.figure()

    if len(features.shape) != 2:
        raise ValueError('该函数仅支持2维特征的特征决策边界绘制！！！')

    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='rainbow', marker='.')
    class_label = list(set(labels))
    plt.legend(handles=scatter.legend_elements()[0], labels=class_label, loc='upper right')
    plt.xlabel(r'$\alpha_1$')
    plt.ylabel(r'$\alpha_2$')
    plt.tight_layout()

    x_min, x_max, y_min, y_max = plt.axis()[0], plt.axis()[1], plt.axis()[2], plt.axis()[3]
    size = 1000
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, size), np.linspace(y_min, y_max, size))
    input_ = torch.from_numpy(
        np.concatenate((xx.reshape([size, size, 1]), yy.reshape([size, size, 1])), axis=2)) \
        .reshape(size * size, 2).float().to(options['device'])

    if 'normalized_logits' in criterion.__dir__():
        logits = criterion.normalized_logits(input_)
    else:
        logits = f.softmax(net.fc2(input_), dim=1)

    zz = np.argmax(logits.cpu().detach().numpy(), axis=1).reshape(size, size)
    plt.contourf(xx, yy, zz, cmap='rainbow', alpha=0.3, levels=len(class_label)+1)

    if save_path:
        dir_path = save_path + '/image'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(dir_path + '/' + options['dataset'] + '_' + options['loss'] + '_feature_decision_boundary.png')

    # ##################### 上半部分绘制决策边界，下半部分绘制logtis的热力分布 #############################################
    plt.figure()
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='rainbow', marker='.')
    class_label = list(set(labels))
    plt.legend(handles=scatter.legend_elements()[0], labels=class_label, loc='upper right')
    plt.xlabel(r'$\alpha_1$')
    plt.ylabel(r'$\alpha_2$')
    plt.tight_layout()

    zz = np.max(logits.cpu().detach().numpy(), axis=1).reshape(size, size)
    plt.contourf(xx, yy, zz, cmap='rainbow', levels=np.linspace(0, 1, 11), alpha=0.3)
    plt.colorbar()

    if save_path:
        dir_path = save_path + '/image'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(dir_path + '/' + options['dataset'] + '_' + options['loss'] + '_feature_logits_boundary.png')

    plt.show()

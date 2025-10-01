"""
    这是一个通用的用于对已有模型训练结果的测试分析代码框架，后面有定制化的特殊使用需求的话，可以在这个的基础上进行更改，效率更高
"""
import argparse
import importlib
import numpy as np

import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from Analysis.plot_roc_curve import plot_multiple_roc_curves
from Analysis.plot_distance_distribution import plot_distance_distribution
from Auxiliary.create_dir_file import name_transport
from Dataset.HRRP_datasets import HRRPOSRDataImage,HRRPOSRData310

from Network.LeNet import LeNet
from Network.VGG32 import VGG32ABN
from Network.CNN1D import CNN1D

from Analysis.get_features_logits import get_features_logits
from Analysis.plot_feature_distribution import plot_feature_distribution, plot_diff_feature_distribution
from Analysis.plot_feature_boundary import plot_feature_boundary
from Analysis.plot_confusion_matrix import plot_confusion_matrix
from Analysis.boundary_prototype import boundary_prototype
from Train_Test import test_osr

parser = argparse.ArgumentParser("Open Set Recognition")

# Dataset
parser.add_argument('--dataset', type=str, default='U_must_give_it_first', help="U_must_give_it_first")
parser.add_argument('--image_size', type=int, default=32)       # 二维图片有 image size 这个参数

# optimization
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--scheduler', type=str, default='please choose scheduler!')

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--lambda', type=float, default=0.1, help="weight for center loss in GCPLoss, SLCPLoss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0, usually 0~1, 0.2")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--alpha', type=float, default=0.1, help="weight for generator_1")
parser.add_argument('--gamma', type=int, default=10, help="the enlarge number of radius")
parser.add_argument('--network', type=str, default='classifier32')

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)

parser.add_argument('--complex', action='store_true', help="data type", default=False)

# misc
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
parser.add_argument('--cs++', action='store_true', help="AKPF++", default=False)

# parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
#                     help='Which pretraining to use if --model=timm_resnet50_pretrained.'
#                          'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')

parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')





if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)

    # ########################################### 判断使用哪种加速平台 ##########################################
    if torch.cuda.is_available():
        options['device'] = torch.device('cuda')
        options['use_gpu'] = True
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
        print("Currently using Cuda.\n")
    else:
        options['device'] = torch.device('cpu')
        options['use_gpu'] = False
        print("Currently using CPU!")

    # # ###### 载入数据，这一部分针对不同种类的数据需要进行定制化操作，从 Dataset 文件夹引入函数进行载入数据 ##########################
    # options['dataset'] = 'MNIST_10_CSR'
    #
    # # options['data_root'] = '/Volumes/Work2024/CV_data/mnist'
    # options['data_root'] = 'F:/CV_data/mnist'
    # dataset = create('mnist', **options)
    # options['num_classes'] = dataset.num_classes
    # train_loader = dataset.train_loader
    # test_loader = dataset.test_loader

    # options['dataset'] = 'CIFAR_10_OSR_for_RATR_CBD'
    # options['known'] = [0, 1, 2, 3, 5, 7, 9]
    # options['data_root'] = 'C:/Users/42941/Documents/CV_data/cifar10'
    # dataset = CIFAR10_OSR(**options)
    # options['num_classes'] = dataset.num_classes
    # train_loader = dataset.train_loader
    # test_loader = dataset.test_loader
    # out_loader = dataset.out_loader

    class_name = {'0': 'A319', '1': 'A320', '2': 'A321', '3': 'A330-2', '4': 'A330-3',
                  '5': 'A350-941', '6': 'B737-7', '7': 'B737-8', '8': 'B747-89L', '9': 'CRJ-900',
                  '10': 'An-26', '11': 'Cessna', '12': 'Yak-42'}
    options['dataset'] = 'HRRP2D'
    # options['known'] = [3, 0, 7, 6, 10]
    # options['known'] = [5, 12, 1, 9, 7]
    options['known'] = [11, 6, 5, 1, 10] #效果最好的一组
    options['known_classes'] = [11, 6, 5, 1, 10]
    data_dir = 'F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/'
    options['data_root'] = [data_dir + class_name[str(j)] + '/2D_train_real' for j in options['known']]
    options['num_classes'] = len(options['known'])
    options['batch_size'] = 32

    transform_ = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()])
    dataset = HRRPOSRDataImage(options['data_root'], transform_)
    train_loader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=True, num_workers=2,
                              pin_memory=options['use_gpu'])

    options['data_root'] = [data_dir + class_name[str(j)] + '/2D_test_real' for j in options['known']]
    transform_ = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])

    dataset = HRRPOSRDataImage(options['data_root'], transform_)
    test_loader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=False, num_workers=2,
                             pin_memory=options['use_gpu'])
    options['unknown'] = list(set(range(len(class_name))) - set(options['known']))
    options['unknown_classes'] = list(set(range(len(class_name))) - set(options['known']))
    options['data_root'] = [data_dir + class_name[str(j)] + '/2D_test_real' for j in options['unknown']]

    transform_ = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])
    dataset = HRRPOSRDataImage(options['data_root'], transform_)
    out_loader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=False, num_workers=2,
                            pin_memory=options['use_gpu'])

    ################################    加载 1D HRRP数据集 ################################################

    # # 配置参数
    # options['dataset'] = 'HRRP1D'
    # class_name = {'0': 'A319', '1': 'A320', '2': 'A321', '3': 'A330-2', '4': 'A330-3',
    #               '5': 'A350-941', '6': 'B737-7', '7': 'B737-8', '8': 'B747-89L', '9': 'CRJ-900',
    #               '10': 'An-26', '11': 'Cessna', '12': 'Yak-42'}
    #
    # options['data_root'] = 'data_code/HRRP_13_pre_results/'  # 数据根目录
    # options['known'] = ['A330-2', 'A319', 'B737-8', 'B737-7', 'An-26']  # 已知类别名称列表 √
    # # options['known'] = ['A350-941', 'Yak-42', 'A320', 'CRJ-900', 'B737-7']  # 已知类别名称列表  √
    # # options['known'] = ['A321', 'B747-89L', 'A330-3', 'Cessna', 'A350-941']  # 已知类别名称列表 √
    # # options['known'] = ['A330-3', 'CRJ-900', 'A330-2', 'A319', 'B737-8'] ✓
    # # options['known'] = ['Cessna', 'B737-7', 'A350-941', 'A320', 'An-26'] √
    # # options['known'] = ['A320', 'A330-3', 'An-26', 'A330-2', 'B737-8']
    # options['train'] = True  # 加载训练数据
    # options['complex'] = False  # 加载实数数据（非复数）
    # # 若已知类别是名称（如 ['A319', 'A320']）,获取总类别数组
    # all_class_names = list(class_name.values())
    #
    # # 加载训练数据
    # train_dataset = HRRPOSRData310(**options)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2,
    #                           pin_memory=options['use_gpu'])
    # options['num_classes'] = train_dataset.num_classes
    # # 加载测试数据（已知类别）
    # options['train'] = False
    # test_dataset = HRRPOSRData310(**options)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    #
    # # 加载未知类别数据（例如：'A330-2', 'Cessna' 等）,因为数据集处理时，用的就是known
    # options['unknown'] = [x for x in all_class_names if x not in options['known']]
    # # options['unknown'] = options['known'] # 赋值一下，训练完后生成文件记录会用到 unknown
    # options['train'] = False
    # out_dataset = HRRPOSRData310(**options)
    # out_loader = DataLoader(out_dataset, batch_size=32, shuffle=False, num_workers=2,
    #                         pin_memory=options['use_gpu'])

    # ###############################   加载1D HRRP数据集完毕 ##############################################

    # ############################# 载入网络结构和训练准则 ######################################


    options['network'] = 'VGG32ABN'
    options['feature_dim'] = 128
    net = VGG32ABN(options['num_classes']).to(options['device'])
    name = name_transport(**options)

    load_path = './Results_HRRP2D/VGG32ABN_Softmax/2025-06-03-20-16-06'
    net_name = options['network'] + '_' + name + '_100(100).pth'
    net.load_state_dict(torch.load(load_path + '/network/' + net_name,map_location=options['device']), strict=False) # 用gpu加载 在gpu上训练的模型
    loss = importlib.import_module('Loss.' + options['loss'])
    criterion = getattr(loss, options['loss'])(**options).to(options['device'])
    criterion_name = options['network'] + '_' + name + '_100(100)_criterion.pth'
    criterion.load_state_dict(torch.load(load_path + '/network/' + criterion_name,map_location=options['device'])) # 用gpu加载 在gpu上训练的模型


    # ########################### 开始定制化的分析和测试 #########################

    # ######### 拿到测试集的相关数据
    train_features, train_labels, train_logits, train_pred_labels = get_features_logits(net, criterion, train_loader, **options)
    test_features, test_labels, test_logits, pred_labels = get_features_logits(net, criterion, test_loader, **options)
    out_features, out_labels, out_logits, out_pred_labels = get_features_logits(net, criterion, out_loader, **options)
    print(test_features.shape,out_features.shape)

    all_features = torch.cat([torch.from_numpy(test_features), torch.from_numpy(out_features)], dim=0)
    all_labels = torch.cat([torch.from_numpy(test_labels), torch.from_numpy(out_labels)], dim=0)

    # 创建布尔数组来标识已知和未知类别
    known_labels = np.array([label in options['known_classes'] for label in all_labels])
    unknown_labels = np.array([label not in options['known_classes'] for label in all_labels])

    ######### 绘制特征分布图
    # plot_feature_distribution(test_features, test_labels, criterion, path=load_path, **options)
    plot_diff_feature_distribution(all_features, all_labels, known_labels, unknown_labels, criterion, path=load_path,**options)

    ####### 检查 各指标
    # test_osr(net, criterion, test_loader, out_loader, **options)

    ##### 绘制 多种方法的 AUROC 图像
    paths = [
        './Results_HRRP2D/VGG32ABN_AdapRingLoss/2025-06-03-20-15-48',
        './Results_HRRP2D/VGG32ABN_ARPL/2025-06-03-20-15-57',
        # './Results_HRRP2D/VGG32ABN_RPL/2025-06-03-20-16-01',
        './Results_HRRP2D/VGG32ABN_Softmax/2025-06-03-20-16-06',
        './Results_HRRP2D/VGG32ABN_Center/2025-06-03-20-15-51',
        # './Results_HRRP2D/VGG32ABN_SLCPL/2025-06-03-20-16-04'
        # 添加更多路径
    ]
    models = [
        'VGG32ABN_AdapRingLoss',
        'VGG32ABN_ARPL',
        # 'VGG32ABN_RPL',
        'VGG32ABN_Softmax',
        'VGG32ABN_Center',
        # 'VGG32ABN_SLCPL'
        # 添加更多模型名称
    ]
    loss_names = [
        'AdapRingLoss',
        'ARPLoss',
        # 'RPLoss',
        'Softmax',
        'CenterLoss',
        # 'SLCPLoss'
        # 添加更多损失函数名称
    ]

    # plot_multiple_roc_curves(paths, models, test_loader, out_loader,loss_names, **options)


    # ######### 绘制决策边界 # ######### 绘制 logits 的热力分布图
    # plot_feature_boundary(net, features_2d, test_labels, criterion, save_path=load_path, **options)

    # ######### 绘制混淆矩阵
    # y_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # plot_confusion_matrix(test_labels, pred_labels, y_classes, y_classes, save_path=load_path, **options)


"""
    这是一个通用的用于 OSR 或者 CSR 的代码框架，后面有定制化的特殊使用需求的话，可以在这个的基础上进行更改，效率更高
    适用范围仅限于各个不同的 loss function ！
"""
import argparse
import importlib

import numpy as np
import torch.nn as nn
import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from Dataset.HRRP_datasets import HRRPOSRData310, HRRPOSRData2D, HRRPOSRDataImage
from Network.LeNet import LeNet
from Train_Test import *
from Auxiliary.create_dir_file import create_dir_path
from Network.VGG32 import VGG32, VGG32ABN
from Network.CNN1D import CNN1D, CNN1DABN
from Auxiliary.get_optimizer_lr import get_optimizer_lr

# from Network.ComplexNN import ComplexVGG32, ComplexCNN1D
from Network.GAN import Generator32, Generator64, Discriminator64, Discriminator32, Generator1D, Discriminator1D
from Auxiliary.save_network import save_network, save_gan
from Auxiliary.schedulers import get_scheduler
from Auxiliary.create_logfile import create_logfile



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
# 新增
parser.add_argument('--lambda_contrast', type=float, default=0.5, help='Weight for contrastive loss term')
parser.add_argument('--class_sim_path', type=str, default=None, help='Path to class similarity matrix .npy file')

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
# parser.add_argument('--cs', action='store_true', dest='cs', default=False, help="Confusing Sample")
# parser.add_argument('--cs++', action='store_true', dest='cs_plus', default=False, help="AKPF++")

# parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
#                     help='Which pretraining to use if --model=timm_resnet50_pretrained.'
#                          'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')

parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
parser.add_argument('--known_classes', type=str, default='None')
parser.add_argument('--unknown_classes', type=str, default='None')


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
    elif torch.backends.mps.is_available():
        options['device'] = torch.device('mps')
        options['use_gpu'] = True
        print("Currently using MPS.\n")
    else:
        options['device'] = torch.device('cpu')
        options['use_gpu'] = False
        print("Currently using CPU!")

    # ###### 载入数据，这一部分针对不同种类的数据需要进行定制化操作，从 Dataset 文件夹引入函数进行载入数据 ##########################
    out_loader = None
    """
        必须对载入数据的类型 options['dataset'] 先做一个解释声明，这一解释声明内容将作为训练结果的文件夹名称保存
    """
    # # # #################################### 加载 2D HRRP数据集 ##########################################################################

    # class_name = {'0': 'A319', '1': 'A320', '2': 'A321', '3': 'A330-2', '4': 'A330-3',
    #               '5': 'A350-941', '6': 'B737-7', '7': 'B737-8', '8': 'B747-89L', '9': 'CRJ-900',
    #               '10': 'An-26', '11': 'Cessna', '12': 'Yak-42'}
    # options['dataset'] = 'HRRP2D'
    # # options['known'] = [3, 0, 7, 6, 10]
    # # options['known'] = [4, 1, 6, 11, 9]
    # # options['known'] = [5, 12, 1, 9, 7]
    # # options['known'] = [2, 8, 4, 11, 5]
    # # options['known'] = [4, 9, 3, 0, 7]
    # # options['known'] = [11, 6, 5, 1, 10]
    # # options['known'] = [1, 4, 10, 3, 7]
    # options['known'] = [9, 2, 6, 11, 0]
    # data_dir = 'data_code/HRRP_13_pre_results/'
    # options['data_root'] = [data_dir + class_name[str(j)] + '/2D_train_real' for j in options['known']]
    # options['num_classes'] = len(options['known'])
    # options['batch_size'] = 32
    # options['use_gpu'] = True
    # options['device'] = torch.device('cuda')
    # options['cs'] = False
    # options['cs++'] = False
    #
    # transform_ = transforms.Compose([transforms.Resize((64, 64)),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor()])
    # dataset = HRRPOSRDataImage(options['data_root'], transform_)
    # print('train_dataset:', dataset.label)
    # train_loader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=True, num_workers=2,
    #                           pin_memory=options['use_gpu'])
    #
    # options['data_root'] = [data_dir + class_name[str(j)] + '/2D_test_real' for j in options['known']]
    # transform_ = transforms.Compose([transforms.Resize((64, 64)),
    #                                  transforms.ToTensor()])
    #
    # dataset = HRRPOSRDataImage(options['data_root'], transform_)
    # test_loader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=False, num_workers=2,
    #                          pin_memory=options['use_gpu'])
    # options['unknown'] = list(set(range(len(class_name))) - set(options['known']))
    # options['data_root'] = [data_dir + class_name[str(j)] + '/2D_test_real' for j in options['unknown']]
    #
    # transform_ = transforms.Compose([transforms.Resize((64, 64)),
    #                                  transforms.ToTensor()])
    # dataset = HRRPOSRDataImage(options['data_root'], transform_)
    # out_loader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=False, num_workers=2,
    #                         pin_memory=options['use_gpu'])

    ################################    加载 1D HRRP数据集 ################################################

    # 配置参数
    options['dataset'] = 'HRRP1D'
    class_name = {'0': 'A319', '1': 'A320', '2': 'A321', '3': 'A330-2', '4': 'A330-3',
                  '5': 'A350-941', '6': 'B737-7', '7': 'B737-8', '8': 'B747-89L', '9': 'CRJ-900',
                  '10': 'An-26', '11': 'Cessna', '12': 'Yak-42'}

    options['data_root'] = 'data_code/HRRP_13_pre_results' # 数据根目录
    # options['known'] = ['A330-2', 'A319', 'B737-8', 'B737-7', 'An-26']  # 已知类别名称列表 √
    # options['known'] = ['A350-941', 'Yak-42', 'A320', 'CRJ-900', 'B737-7']  # 已知类别名称列表  √
    # options['known'] = ['A321', 'B747-89L', 'A330-3', 'Cessna', 'A350-941']  # 已知类别名称列表 √
    # options['known'] = ['A330-3', 'CRJ-900', 'A330-2', 'A319', 'B737-8'] ✓
    # options['known'] = ['Cessna', 'B737-7', 'A350-941', 'A320', 'An-26'] √ 效果最好
    options['known'] = ['A320', 'A330-3', 'An-26', 'A330-2', 'B737-8']
    options['train'] = True # 加载训练数据
    options['complex'] = False   # 加载实数数据（非复数）
    # 若已知类别是名称（如 ['A319', 'A320']）,获取总类别数组
    all_class_names = list(class_name.values())

    # 加载训练数据
    train_dataset = HRRPOSRData310(**options)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=2,
                            pin_memory=options['use_gpu'])
    options['num_classes'] = train_dataset.num_classes
    # 加载测试数据（已知类别）
    options['train'] = False
    test_dataset = HRRPOSRData310(**options)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载未知类别数据（例如：'A330-2', 'Cessna' 等）,因为数据集处理时，用的就是known
    options['unknown'] = [x for x in all_class_names if x not in options['known']]
    # options['unknown'] = options['known'] # 赋值一下，训练完后生成文件记录会用到 unknown
    options['train'] = False
    out_dataset = HRRPOSRData310(**options)
    out_loader = DataLoader(out_dataset, batch_size=32, shuffle=False,num_workers=2,
                            pin_memory=options['use_gpu'])

    # ###############################   加载1D HRRP数据集完毕 ##############################################

    # ############################# 载入网络结构，载入时必须声明options['network'] ######################################
    options['network'] = 'VGG32ABN'
    options['feature_dim'] = 128
    # net = VGG32(options['num_classes']).to(options['device'])
    net = VGG32ABN(options['num_classes']).to(options['device'])

    # options['network'] = 'CNN1DABN'
    # options['feature_dim'] = 128
    # net = CNN1DABN(options['num_classes']).to(options['device'])

    # options['network'] = 'CNN1D'
    # options['feature_dim'] = 128
    # net = CNN1D(options['num_classes']).to(options['device'])

    # options['network'] = 'ComplexVGG32'
    # options['feature_dim'] = 128
    # net = ComplexVGG32(**options).to(options['device'])

    # options['network'] = 'ComplexCNN1D'
    # options['feature_dim'] = 128
    # net = ComplexCNN1D(**options).to(options['device'])

    # options['network'] = 'LeNet'
    # options['feature_dim'] = 2
    # net = LeNet(options['num_classes']).to(options['device'])

    net_generator, net_generator_2, net_discriminator, criterion_bce = None, None, None, None

    # ####################################### 设置损失函数和优化器 ##################################################
    loss = importlib.import_module('Loss.' + options['loss'])
    criterion = getattr(loss, options['loss'])(**options).to(options['device'])

    optimizer_discriminator, optimizer_generator, optimizer_generator_2 = None, None, None


    params_list = [{'params': net.parameters()}, {'params': criterion.parameters()}]
    # optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    # options['optimizer_'] = 'Adam'       # sdg
    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
    options['optimizer_'] = 'SGD'       # sdg

    # scheduler 的类型 : 'step', 'plateau', 'cosine', 'cosine_warm_restarts',
    # 'cosine_warm_restarts_warmup', 'warm_restarts_plateau', 'multi_step'
    # options['scheduler'] = 'multi_step'
    options['scheduler'] = 'cosine_warm_restarts_warmup'
    scheduler = get_scheduler(optimizer, **options)

    # ###################### 生成保存文件的文件夹，相关路径等内容 ##########################################
    options['save_path'] = create_dir_path(**options)   # save_path 表示保存路径
    writer = SummaryWriter(options['save_path'])
    options['save_number'] = 10                         # save_number 表示一共保存几个网络模型，

    # ################################ 开始训练，保存训练数据 ###################################################
    options['best_auc'], options['best_auc_epoch'], auc, oscr = 0, None, None, None
    options['auc'], options['oscr'] = 0, 0
    for epoch in range(options['max_epoch']):
        options['print_freq'] = int(len(train_loader) / 5)      # 每个 epoch 内打印 5 行训练结果
        options['epoch'] = epoch

        print("Training ===========> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        train_loss = train(net, criterion, optimizer, train_loader, **options)

        if options['cs'] and options['loss'] == 'ARPLoss':
            train_cs(net, net_discriminator, net_generator, criterion, criterion_bce,
                     optimizer, optimizer_discriminator, optimizer_generator,
                     train_loader, **options)

        if out_loader:
            test_acc, test_loss, auc, oscr = test_osr(net, criterion, test_loader, out_loader, **options)
        else:
            test_acc, test_loss = test(net, criterion, test_loader, **options)

        if auc and auc > options['best_auc']:
            options['best_auc'] = auc
            options['best_auc_epoch'] = epoch
            save_network(net, criterion, exist_best_auc=True, **options)

        # ###### 观察训练过程中，这些变量随 epoch 的变化
        writer.add_scalar('train_loss', train_loss, epoch)  # 画loss，横坐标为epoch
        writer.add_scalar('train_lr', get_optimizer_lr(optimizer), epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        if auc or oscr:
            writer.add_scalar('auc', auc, epoch)
            writer.add_scalar('oscr', oscr, epoch)

        if options['scheduler'] == 'plateau' or options['scheduler'] == 'warm_restarts_plateau':
            scheduler.step(test_acc, epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)

        if epoch in np.linspace(0, options['max_epoch'] - 1, options['save_number']).astype(int):
            save_network(net, criterion, **options)
            if options['cs']:
                save_gan(net_generator, net_discriminator, **options)
            if options['cs++']:
                save_gan(net_generator, net_discriminator, net_generator_2, **options)

        if epoch == options['max_epoch']-1:
            options['acc'] = test_acc
            if auc or oscr:
                options['auc'], options['oscr'] = auc, oscr

    # ##################### 特殊参数的保存 #######################
    if options['loss'] == 'AKPFLoss' and (options['cs'] or options['cs++']):
        np.save(options['save_path']+'/R_recording.npy', options['R_recording'])
        np.save(options['save_path']+'/kR_recording.npy', options['kR_recording'])

    writer.close()
    create_logfile(**options)
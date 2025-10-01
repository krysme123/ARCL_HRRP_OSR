import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torch.nn.functional as f
import argparse
import os
from skimage import io
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class_name_10 = {'1': 'A319', '2': 'A320', '3': 'A321', '4': 'A330-2', '5': 'A330-3',
                 '6': 'A350-941', '7': 'B737-7', '8': 'B737-8', '9': 'B747-89L', '10': 'CRJ-900'}

class_name = {'0': 'A319', '1': 'A320', '2': 'A321', '3': 'A330-2', '4': 'A330-3',
                  '5': 'A350-941', '6': 'B737-7', '7': 'B737-8', '8': 'B747-89L', '9': 'CRJ-900',
                  '10': 'An-26', '11': 'Cessna', '12': 'Yak-42'}


def normalization(data):  # 对numpy数组data（x,1,y）进行1,2范数的归一化，输出的仍然是numpy数据
    data = torch.from_numpy(data)
    data = f.normalize(data, p=2, dim=2)
    return data.numpy()


class HRRPOSRData15(Dataset):               # 载入 HRRP_15 对应的 HRRP 数据
    def __init__(self, **options):          # 通过使用 options['known'] 和 options['train'] 来控制数据类别
        label, data = None, None
        for i in range(len(options['known'])):          # 输入的 known 的元素必须在 1~15 之间
            t = sio.loadmat(options['data_root'] + '/plane' + str(options['known'][i]) + '.mat')
            hrrp_data = t['hrrp_'].astype('float32')
            if options['train']:
                hrrp_data = hrrp_data[0:1600, :]
            else:
                hrrp_data = hrrp_data[1600:, :]
            if i == 0:
                label = np.zeros(hrrp_data.shape[0])
                data = hrrp_data
            else:
                label = np.concatenate((label, i * np.ones(hrrp_data.shape[0])), axis=0)
                data = np.concatenate((data, hrrp_data), axis=0)
        self.data = data.reshape([data.shape[0], 1, data.shape[1]])
        self.label = label
        self.Num = self.data.shape[0]
        self.data = normalization(self.data)
        self.num_classes = len(options['known'])
        if options['train']:
            print('训练数据的尺寸是：', self.data.shape)
            print('训练标签的尺寸是：', self.label.shape)
        else:
            print('测试数据的尺寸是：', self.data.shape)
            print('测试标签的尺寸是：', self.label.shape)

    def __len__(self):
        return self.Num

    def __getitem__(self, idx):
        x = self.data[idx, :]
        y = self.label[idx]
        return x, y


class HRRPOSRData310(Dataset):      # 载入 HRRP_3 或者 HRRP_10 对应的 HRRP 数据， options['data_root'] 和 options['known'] 必须对应起来
    def __init__(self, **options):     # 通过使用 options['known'] 和 options['train'] 和 options['complex'] 来控制数据类别
        label, data = None, None
        if 'unknown' in options:  # 检查键是否存在
            for i in range(len(options['unknown'])):  # 输入的 known 的元素必须在 飞机类别名称 之间
                if options['train']:
                    if options['complex']:
                        t = np.load(options['data_root'] + '/' + options['unknown'][i] + '/train_data_complex_' +
                                    options['unknown'][i] + '.npy')
                    else:
                        t = np.load(options['data_root'] + '/' + options['unknown'][i] + '/train_data_real_' +
                                    options['unknown'][i] + '.npy')
                else:
                    if options['complex']:
                        t = np.load(options['data_root'] + '/' + options['unknown'][i] + '/test_data_complex_' +
                                    options['unknown'][i] + '.npy')
                    else:
                        t = np.load(options['data_root'] + '/' + options['unknown'][i] + '/test_data_real_' +
                                    options['unknown'][i] + '.npy')
                if i == 0:
                    label = np.zeros(t.shape[0])
                    data = t
                else:
                    label = np.concatenate((label, i * np.ones(t.shape[0])), axis=0)
                    data = np.concatenate((data, t), axis=0)
        else:
            for i in range(len(options['known'])):  # 输入的 known 的元素必须在 飞机类别名称 之间
                if options['train']:
                    if options['complex']:
                        t = np.load(options['data_root'] + '/' + options['known'][i] + '/train_data_complex_' +
                                    options['known'][i] + '.npy')
                    else:
                        t = np.load(options['data_root'] + '/' + options['known'][i] + '/train_data_real_' +
                                    options['known'][i] + '.npy')
                else:
                    if options['complex']:
                        t = np.load(options['data_root'] + '/' + options['known'][i] + '/test_data_complex_' +
                                    options['known'][i] + '.npy')
                    else:
                        t = np.load(options['data_root'] + '/' + options['known'][i] + '/test_data_real_' +
                                    options['known'][i] + '.npy')
                if i == 0:
                    label = np.zeros(t.shape[0])
                    data = t
                else:
                    label = np.concatenate((label, i * np.ones(t.shape[0])), axis=0)
                    data = np.concatenate((data, t), axis=0)

        if options['complex']:          # Windows 系统可用，MacOS不支持
            self.data = data.reshape([data.shape[0], 1, data.shape[1]]).astype(np.complex64, copy=False)
        else:
            self.data = data.reshape([data.shape[0], 1, data.shape[1]]).astype(np.float32, copy=False)
        self.label = label.astype(np.longlong, copy=False)
        self.Num = self.data.shape[0]
        # self.data = normalization(self.data)
        self.num_classes = len(options['known'])

        if options['train']:
            print('训练数据的尺寸是：', self.data.shape)
            print('训练标签的尺寸是：', self.label.shape, '\n')
        else:
            print('测试数据的尺寸是：', self.data.shape)
            print('测试标签的尺寸是：', self.label.shape, '\n')

    def __len__(self):
        return self.Num

    def __getitem__(self, idx):
        x = self.data[idx, :]
        y = self.label[idx]
        return x, y


class HRRPOSRData2D(Dataset):        # 载入 HRRP_3 和 HRRP_10 的 2D 数据，且是 complex 数据
    def __init__(self, **options):     # 通过使用 options['known'] 和 options['train'] 和 options['complex'] 来控制数据类别
        if not options['complex']:
            raise ValueError('载入HRRP的二维模数据用的不是这个函数！')
        label, data = None, None
        for i in tqdm(range(len(options['known']))):          # 输入的 known 的元素必须在 'An-26', 'Cessna', 'Yak-42' 之间
            if options['train']:
                t = np.load(options['data_root'] + '/' + options['known'][i] + '/train_data_complex_2D_' +
                            options['known'][i] + '.npy')
            else:
                t = np.load(options['data_root'] + '/' + options['known'][i] + '/test_data_complex_2D_' +
                            options['known'][i] + '.npy')
            if i == 0:
                label = np.zeros(t.shape[0])
                data = t
            else:
                label = np.concatenate((label, i * np.ones(t.shape[0])), axis=0)
                data = np.concatenate((data, t), axis=0)
        self.data = data.astype(np.complex64, copy=False)
        self.label = label.astype(np.longlong, copy=False)
        self.Num = self.data.shape[0]
        self.num_classes = len(options['known'])

        if options['train']:
            print('训练数据的尺寸是：', self.data.shape)
            print('训练标签的尺寸是：', self.label.shape)
        else:
            print('测试数据的尺寸是：', self.data.shape)
            print('测试标签的尺寸是：', self.label.shape)

    def __len__(self):
        return self.Num

    def __getitem__(self, idx):
        x = self.data[idx, :]
        y = self.label[idx]
        return x, y


class HRRPOSRDataImage(Dataset):            # 这个写出的 Dataset 不能显性地给出 self.data，但是也可以当 Dataset 用
    """
        载入 HRRP_3 和 HRRP_10 的二维图片数据，其 train/test ，以及飞机种类，依靠 root_dir 来控制！
    """
    def __init__(self, root_dir, transform=None, gray=None):
        self.transform, self.gray = transform, gray
        if type(root_dir).__name__ != 'list':
            raise ValueError("要求数据路径为一个 list ！")
        elif len(root_dir) == 1:
            self.images = os.listdir(root_dir[0])         # list object
            length = len(self.images)
            self.root_dir = root_dir
            self.all_dir = length * root_dir
        else:
            self.root_dir = root_dir
            images, all_dir = [], []
            for i in root_dir:
                length = len(os.listdir(i))
                images.extend(os.listdir(i))
                all_dir.extend(length*[i])
            self.images = images
            self.all_dir = all_dir
        label = [self.root_dir.index(self.all_dir[i]) for i in range(len(self.images))]
        self.label = np.array(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.all_dir[index], image_index)  # 获取索引为index的图片的路径名
        img = io.imread(img_path)  # 读取该图片，这样读入的灰度图只有 1 通道，但是通过 imageFolder 读入的就有 3 通道
        # label = self.root_dir[index].split('/')[-1][-1]
        if self.transform:
            img = Image.fromarray(img)
            img = img.convert("RGB")
            img = self.transform(img)
        if self.gray:
            img = img.convert("L")
        label = self.root_dir.index(self.all_dir[index])
        return img, label


parser = argparse.ArgumentParser("Open Set Recognition")


if __name__ == '__main__':
    args = parser.parse_args()
    options_ = vars(args)

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        options_['use_gpu'] = True
    else:
        options_['use_gpu'] = False

    options_['batch_size'] = 128
    options_['num_workers'] = 2

    # options_['data_root'] = r'C:\Users\42941\Documents\HRRP_data\HRRP_15\HRRP15'
    # options_['data_root'] = '/Volumes/Work2024/HRRP_data/HRRP_15/HRRP15'        # 数据在网盘上的地址
    # options_['train'] = False
    # options_['known'] = [1, 3, 5]
    # data_ = HRRPOSRData15(**options_)

    # options_['data_root'] = r'C:\Users\42941\Documents\HRRP_data\HRRP_3_pre_results'
    # options_['data_root'] = '/Volumes/Work2024/HRRP_data/HRRP_3_pre_results'        # 数据在网盘上的地址
    # options_['train'] = False
    # options_['complex'] = False
    # options_['known'] = ['An-26', 'Cessna', 'Yak-42']
    # data_ = HRRPOSRData310(**options_)

    # options_['data_root'] = '/Volumes/Work2024/HRRP_data/HRRP_10_pre_results'        # 数据在网盘上的地址
    # options_['train'] = False
    # options_['complex'] = False
    # options_['known'] = ['A319', 'A320', 'A321', 'CRJ-900']
    # data_ = HRRPOSRData310(**options_)

    # options_['data_root'] = '/Volumes/Work2024/HRRP_data/HRRP_3_pre_results'        # 数据在网盘上的地址 - Mac
    # options_['data_root'] = 'F:/HRRP_data/HRRP_3_pre_results'                       # 数据在网盘上的地址 - Windows
    # options_['train'] = False
    # options_['complex'] = True
    # options_['known'] = ['An-26', 'Cessna']
    # data_ = HRRPOSRData2D(**options_)

    # options_['data_root'] = ['F:/HRRP_data/HRRP_10_pre_results/A319/2D_test_real',
    #                          'F:/HRRP_data/HRRP_10_pre_results/A320/2D_test_real',
    #                          'F:/HRRP_data/HRRP_10_pre_results/A321/2D_test_real',]    # 数据在网盘上的地址 - Windows
    # transform_ = transforms.Compose([
    #     transforms.Resize((32, 32)), transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()])
    # a = HRRPOSRDataImage(options_['data_root'], transform_)
    # print(a.label.shape)
    # print(len(a.images), a.__getitem__(0)[0].shape, a.__getitem__(0)[1])
    # train_data_loader = DataLoader(a, batch_size=options_['batch_size'], shuffle=True, num_workers=4)
    # print(train_data_loader)

    ####################################################### 2D ######################################################
    # options = {}
    # options['dataset'] = 'HRRP2D'
    # options['known'] = [3, 0, 7, 6, 10]
    # data_dir = 'F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results/'
    # options['data_root'] = [data_dir + class_name[str(j)] + '/2D_train_real' for j in options['known']]
    # options['num_classes'] = len(options['known'])
    # options['batch_size'] = 32
    # options['use_gpu'] = False
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
    # dataset = HRRPOSRDataImage(options['data_root'], transform_)
    # test_loader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=False, num_workers=2,
    #                          pin_memory=options['use_gpu'])
    # print('test_dataset:', dataset.label)
    # options['unknown'] = list(set(range(len(class_name))) - set(options['known']))
    # options['data_root'] = [data_dir + class_name[str(j)] + '/2D_test_real' for j in options['unknown']]
    # transform_ = transforms.Compose([transforms.Resize((64, 64)),
    #                                  transforms.ToTensor()])
    # dataset = HRRPOSRDataImage(options['data_root'], transform_)
    # out_loader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=False, num_workers=2,
    #                         pin_memory=options['use_gpu'])
    # print('out_dataset:', dataset.label)
    ##########################################################################################################

    ############################################### 1D ######################################################
    # 配置参数
    options = {
        'data_root': 'F:/Project/Score-CAM-master/data_code/HRRP_13_pre_results',  # 数据根目录
        'known': ['A330-2', 'A319', 'B737-8', 'B737-7', 'An-26'],  # 已知类别名称列表
        'train': True,  # 加载训练数据
        'complex': False  # 加载实数数据（非复数）
    }
    # 若已知类别是名称（如 ['A319', 'A320']）,获取总类别数组
    all_class_names = list(class_name.values())

    # 加载训练数据
    train_dataset = HRRPOSRData310(**options)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    options['num_classes'] = train_dataset.num_classes
    # 加载测试数据（已知类别）
    options['train'] = False
    test_dataset = HRRPOSRData310(**options)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 加载未知类别数据（例如：'A330-2', 'Cessna' 等）
    options['known'] = list(set(all_class_names) - set(options['known']))
    options['train'] = False
    out_dataset = HRRPOSRData310(**options)
    out_loader = DataLoader(out_dataset, batch_size=32, shuffle=False)


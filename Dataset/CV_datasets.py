import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, EMNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, ImageFolder
from PIL import Image
import numpy as np
import argparse
import torchvision

""" 这个程序可以加载诸多类别的数据集：
        黑白图片：mnist, emnist, kmnist, fashion_mnist
        彩色图片：cifar10, cifar100, svhn, tiny_imagenet, imagenet-o
        以上所有种类(除了imagenet-o以外)均可根据输入的 known 类别 list，实现本数据集自身的 OSR 数据划分；
        除了 tiny_imagenet 以外，其余 8 个数据集均可通过 create 函数实现全类别的数据集提取，而 tiny 自身需要通过 list(0~199) 来实现；
        options_['data_root'] = 'C:/Users/42941/Documents/CV_data/mnist' 选取绝对路径
"""


class MNISTRGB(MNIST):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class MNIST_(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(options['image_size']),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = MNISTRGB(root=options['data_root'], train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        
        testset = MNISTRGB(root=options['data_root'], train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = 10


class MNIST_Filter(MNISTRGB):
    """MNIST Dataset.
    """
    def __Filter__(self, known):
        targets = self.targets.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)


class MNIST_OSR(object):
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 10))) - set(options['known']))

        print('Selected Labels: ', options['known'])

        train_transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = MNIST_Filter(root=options['data_root'], train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'],
            pin_memory=pin_memory,
        )

        testset = MNIST_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
            pin_memory=pin_memory,
        )

        outset = MNIST_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
            pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class KMNISTRGB(KMNIST):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class KMNIST_(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(options['image_size']),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = KMNISTRGB(root=options['data_root'], train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = KMNISTRGB(root=options['data_root'], train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = 10


class KMNIST_Filter(KMNISTRGB):
    """kMNIST Dataset.
    """
    def __Filter__(self, known):
        targets = self.targets.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)


class KMNIST_OSR(object):
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 10))) - set(options['known']))

        print('Selected Labels: ', options['known'])

        train_transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = KMNIST_Filter(root=options['data_root'], train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = KMNIST_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = KMNIST_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class CIFAR10_(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(options['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = CIFAR10(root=options['data_root'], train=True, download=True, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = CIFAR10(root=options['data_root'], train=False, download=True, transform=transform)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.train_loader = train_loader
        self.test_loader = test_loader


class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset."""
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR(object):
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 10))) - set(options['known']))

        print('Selected Labels: ', options['known'])

        train_transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.RandomCrop(options['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = CIFAR10_Filter(root=options['data_root'], train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = CIFAR10_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class CIFAR100_(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(options['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = CIFAR100(root=options['data_root'], train=True, download=True, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = CIFAR100(root=options['data_root'], train=False, download=True, transform=transform)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        self.num_classes = 100
        self.train_loader = train_loader
        self.test_loader = test_loader


class CIFAR100_Filter(CIFAR100):
    """CIFAR100 Dataset."""
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR100_OSR(object):
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 100))) - set(options['known']))

        print('Selected Labels: ', options['known'])

        train_transform = transforms.Compose([
            transforms.RandomCrop(options['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = CIFAR100_Filter(root=options['data_root'], train=True, download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = CIFAR100_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = CIFAR100_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class SVHN_(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(options['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = SVHN(root=options['data_root'], split='train', download=True, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = SVHN(root=options['data_root'], split='test', download=True, transform=transform)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.train_loader = train_loader
        self.test_loader = test_loader


class SVHN_Filter(SVHN):
    """SVHN Dataset."""
    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)


class SVHN_OSR(object):
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 10))) - set(options['known']))

        print('Selected Labels: ', options['known'])

        train_transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.RandomCrop(options['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = SVHN_Filter(root=options['data_root'], split='train', download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = SVHN_Filter(root=options['data_root'], split='test', download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=options['data_root'], split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


# class ImageNet(object):
#     def __init__(self, **options):
#         transform_train = transforms.Compose([
#             transforms.Resize((options['image_size'], options['image_size'])),
#             transforms.RandomCrop(options['image_size'], padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ])
#         transform = transforms.Compose([
#             transforms.Resize((options['image_size'], options['image_size'])),
#             transforms.ToTensor(),
#         ])
#
#         pin_memory = True if options['use_gpu'] else False
#
#         trainset = ImageNet(root=options['data_root'], split='train', download='True', transform=transform_train)
#
#         train_loader = torch.utils.data.DataLoader(
#             trainset, batch_size=options['batch_size'], shuffle=True,
#             num_workers=options['workers'], pin_memory=pin_memory,
#         )
#
#         testset = ImageNet(root=options['data_root'], split='test', download='True', transform=transform)
#
#         test_loader = torch.utils.data.DataLoader(
#             testset, batch_size=options['batch_size'], shuffle=False,
#             num_workers=options['workers'], pin_memory=pin_memory,
#         )
#
#         self.num_classes = 200
#         self.train_loader = train_loader
#         self.test_loader = test_loader


class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset."""
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class Tiny_ImageNet_OSR(object):
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 200))) - set(options['known']))

        print('Selected Labels: ', options['known'])

        train_transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.RandomCrop(options['image_size'], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = Tiny_ImageNet_Filter(os.path.join(options['data_root'], 'tiny-imagenet-200', 'train'),
                                        train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = Tiny_ImageNet_Filter(os.path.join(options['data_root'], 'tiny-imagenet-200', 'val'), transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = Tiny_ImageNet_Filter(os.path.join(options['data_root'], 'tiny-imagenet-200', 'val'), transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class ImageNet2012_Filter(ImageFolder):
    """ImageNet2012 Dataset."""
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class ImageNet2012_OSR(object):
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 1000))) - set(options['known']))

        print('Selected Labels: ', options['known'])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = ImageNet2012_Filter(os.path.join(options['data_root'], 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = ImageNet2012_Filter(os.path.join(options['data_root'], 'val'), transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = ImageNet2012_Filter(os.path.join(options['data_root'], 'val'), transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class FashionMNISTRGB(FashionMNIST):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FashionMNIST_(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(options['image_size']),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = FashionMNISTRGB(root=options['data_root'], train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = FashionMNISTRGB(root=options['data_root'], train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = 10


class FashionMNIST_Filter(FashionMNISTRGB):
    """FashionMNIST Dataset."""
    def __Filter__(self, known):
        targets = self.targets.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)


class FashionMNIST_OSR(object):
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 10))) - set(options['known']))

        print('Selected Labels: ', options['known'])

        train_transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = FashionMNIST_Filter(root=options['data_root'], train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'],
            pin_memory=pin_memory,
        )

        testset = FashionMNIST_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
            pin_memory=pin_memory,
        )

        outset = FashionMNIST_Filter(root=options['data_root'], train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
            pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class EMNISTRGB(EMNIST):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class EMNIST_(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(options['image_size']),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = EMNISTRGB(root=options['data_root'], train=True, download=True, transform=transform, split='letters')
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = EMNISTRGB(root=options['data_root'], train=False, download=True, transform=transform, split='letters')
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = 10


class EMNIST_Filter(EMNISTRGB):
    """EMNIST Dataset."""
    def __Filter__(self, known):
        targets = self.targets.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)


class EMNIST_OSR(object):
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 10))) - set(options['known']))

        print('Selected Labels: ', options['known'])

        train_transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = EMNIST_Filter(root=options['data_root'], train=True, download=True, transform=train_transform,
                                 split='letters')
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'],
            pin_memory=pin_memory,
        )

        testset = EMNIST_Filter(root=options['data_root'], train=False, download=True, transform=transform,
                                split='letters')
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
            pin_memory=pin_memory,
        )

        outset = EMNIST_Filter(root=options['data_root'], train=False, download=True, transform=transform,
                               split='letters')
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
            pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class ImageNet_O(object):
    def __init__(self, **options):
        # train_transform = transforms.Compose([
        #     transforms.Resize((options['image_size'], options['image_size'])),
        #     transforms.RandomCrop(options['image_size'], padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ])
        pin_memory = True if options['use_gpu'] else False
        transform = transforms.Compose([
            transforms.Resize((options['image_size'], options['image_size'])),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder(root=options['data_root'], transform=transform)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=options['batch_size'], shuffle=True,
            num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        self.data_loader = data_loader
        self.num_classes = 200


class Omniglot(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(options['image_size']),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize(options['image_size']),
            transforms.ToTensor(),
        ])

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.Omniglot(root=options['data_root'], download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'],
            pin_memory=pin_memory)

        testset = torchvision.datasets.Omniglot(root=options['data_root'], download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
            pin_memory=pin_memory)
        self.train_loader = train_loader
        self.test_loader = test_loader


__factory = {
    'mnist': MNIST_,
    'kmnist': KMNIST_,
    'cifar10': CIFAR10_,
    'cifar100': CIFAR100_,
    'svhn': SVHN_,
    'emnist': EMNIST_,
    'fashion_mnist': FashionMNIST_,
    'imagenet_o': ImageNet_O,
    'omniglot': Omniglot,
}


def create(name, **options):
    if name not in __factory.keys():
        raise KeyError("Unknown Dataset: {}".format(name))
    return __factory[name](**options)


parser = argparse.ArgumentParser("Open Set Recognition")


if __name__ == '__main__':
    args = parser.parse_args()
    options_ = vars(args)

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        options_['use_gpu'] = True
    else:
        options_['use_gpu'] = False

    options_['image_size'] = 32
    options_['batch_size'] = 128
    options_['num_workers'] = 2
    options_['known'] = [1, 3, 5, 7, 9]

    # options_['data_root'] = r'C:\Users\42941\Documents\CV_data\mnist'
    # data = MNIST_(**options_)
    # data = MNIST_OSR(**options_)
    # data = create('mnist', **options_)
    # print(data, data.train_loader)

    # options_['data_root'] = r'C:\Users\42941\Documents\CV_data\kmnist'
    # data = KMNIST_(**options_)
    # data = KMNIST_OSR(**options_)
    # data = create('kmnist', **options_)
    # print(data, data.train_loader)

    # options_['data_root'] = r'C:\Users\42941\Documents\CV_data\cifar10'
    # data = CIFAR10_(**options_)
    # data = CIFAR10_OSR(**options_)
    # data = create('cifar10', **options_)
    # print(data, data.train_loader)

    # options_['data_root'] = r'C:\Users\42941\Documents\CV_data\cifar100'
    # data = CIFAR100_(**options_)
    # data = CIFAR100_OSR(**options_)
    # data = create('cifar100', **options_)
    # print(data, data.train_loader)

    # options_['data_root'] = r'C:\Users\42941\Documents\CV_data\svhn'
    # data = SVHN_(**options_)
    # data = SVHN_OSR(**options_)
    # data = create('svhn', **options_)
    # print(data, data.train_loader)

    # options_['data_root'] = r'C:\Users\42941\Documents\CV_data\tiny_imagenet'
    # data = Tiny_ImageNet_OSR(**options_)
    # print(data, data.train_loader)

    # options_['data_root'] = r'C:\Users\42941\Documents\CV_data\fashion_mnist'
    # data = FashionMNIST_(**options_)
    # data = FashionMNIST_OSR(**options_)
    # data = create('fashion_mnist', **options_)
    # print(data, data.train_loader)

    # options_['data_root'] = r'C:\Users\42941\Documents\CV_data\emnist'
    # data = EMNIST_(**options_)
    # data = EMNIST_OSR(**options_)
    # data = create('emnist', **options_)
    # print(data, data.train_loader)

    # options_['data_root'] = r'C:\Users\42941\Documents\CV_data\imagenet-o'
    # data = ImageNet_O(**options_)
    # data = create('imagenet_o', **options_)
    # print(data, data.data_loader)

    options_['data_root'] = 'F:/CV_data/omniglot'
    data = Omniglot(**options_)
    # data = create('omniglot', **options_)
    print(data, data.train_loader)

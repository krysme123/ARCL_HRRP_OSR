from torch.utils.data import Dataset
import os
import numpy as np
import random
import math
import numbers
import cv2
import torch
import argparse


class Compose:
    """Composes several transforms together.
    Args:
        transforms(list of 'Transform' object): list of transforms to compose
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for trans in self.transforms:
            img = trans(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToCVImage:
    """Convert an Opencv image to a 3 channel uint8 image"""
    def __call__(self, image):
        """
        Args:
            image (numpy array): Image to be converted to 32-bit floating point
        Returns:
            image (numpy array): Converted Image
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.astype('uint8')
        return image


class RandomResizedCrop:
    """Randomly crop a rectangle region whose aspect ratio is randomly sampled
    in [3/4, 4/3] and area randomly sampled in [8%, 100%], then resize the cropped
    region into a 224-by-224 square image.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped (w / h)
        interpolation: Default: cv2.INTER_LINEAR:
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='linear'):
        self.methods = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }
        self.size = (size, size)
        self.interpolation = self.methods[interpolation]
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        h, w, _ = img.shape
        area = w * h
        output_w, output_h, topleft_x, topleft_y = 0, 0, 0, 0
        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            target_ratio = random.uniform(*self.ratio)
            output_h = int(round(math.sqrt(target_area * target_ratio)))
            output_w = int(round(math.sqrt(target_area / target_ratio)))
            if random.random() < 0.5:
                output_w, output_h = output_h, output_w
            if output_w <= w and output_h <= h:
                topleft_x = random.randint(0, w - output_w)
                topleft_y = random.randint(0, h - output_h)
                break
        if output_w > w or output_h > h:
            output_w = min(w, h)
            output_h = output_w
            topleft_x = random.randint(0, w - output_w)
            topleft_y = random.randint(0, h - output_w)
        cropped = img[topleft_y: topleft_y + output_h, topleft_x: topleft_x + output_w]
        resized = cv2.resize(cropped, self.size, interpolation=self.interpolation)
        return resized

    def __repr__(self):
        inter_name = None
        for name, inter in self.methods.items():
            if inter == self.interpolation:
                inter_name = name
        interpolate_str = inter_name
        format_str = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_str += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_str += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_str += ', interpolation={0})'.format(interpolate_str)

        return format_str


class RandomHorizontalFlip:
    """Horizontally flip the given opencv image with given probability p.

    Args:
        p: probability of the image being flipped
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() < self.p:
            img = cv2.flip(img, 1)

        return img


class ToTensor:
    """convert an opencv image (h, w, c) ndarray range from 0 to 255 to a pytorch
    float tensor (c, h, w) ranged from 0 to 1
    """

    def __call__(self, img):
        """
        Args:
            a numpy array (h, w, c) range from [0, 255]

        Returns:
            a pytorch tensor
        """
        # convert format H W C to C H W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float() / 255.0

        return img


class Normalize:
    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation
    for each channel in torch tensor:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img):
        """
        Args:
            (H W C) format numpy array range from [0, 255]
        Returns:
            (H W C) format numpy array in float32 range from [0, 1]
        """
        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'
        if not self.inplace:
            img = img.clone()
        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])
        return img


class Resize:
    def __init__(self, resized=256, interpolation='linear'):
        methods = {
            "area": cv2.INTER_AREA,
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4
        }
        self.interpolation = methods[interpolation]
        if isinstance(resized, numbers.Number):
            resized = (resized, resized)
        self.resized = resized

    def __call__(self, img):
        img = cv2.resize(img, self.resized, interpolation=self.interpolation)
        return img


class CUB(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, **options):
        self.root = options['data_root']
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}

        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path
        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id
        self.data_id = []
        if self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        self.data_id.append(image_id)
        if not self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, index):
        """
        Args:
            index: index of training Dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(os.path.join(self.root, 'images', path))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            class_id = self.target_transform(class_id)
        return image, class_id

    def _get_path_by_id(self, image_id):
        return self.images_path[image_id]

    def _get_class_by_id(self, image_id):
        return self.class_ids[image_id]


class CUB_Filter(CUB):
    def __Filter__(self, known):
        location = []
        for index in range(self.__len__()):
            image_id = self.data_id[index]
            class_id = int(self._get_class_by_id(image_id)) - 1
            if class_id in known:
                location.extend([index])
                self.class_ids[image_id] = known.index(class_id) + 1
        self.data_id = np.array(self.data_id)[location]


TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]


class CUB_OSR(object):          # CUB 的 transform 得单独搞，没有使用 torchvision 的 transform ！！！！！
    def __init__(self, **options):
        self.num_classes = len(options['known'])
        self.known = options['known']
        self.unknown = list(set(list(range(0, 200))) - set(options['known']))
        print('Selected Labels: ', options['known'])

        train_transform = Compose([
            ToCVImage(),
            RandomResizedCrop(options_['image_size']),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(TRAIN_MEAN, TRAIN_STD)
        ])
        transform = Compose([
            ToCVImage(),
            Resize(options_['image_size']),
            ToTensor(),
            Normalize(TEST_MEAN, TEST_STD)
        ])
        pin_memory = True if options['use_gpu'] else False

        trainset = CUB_Filter(train=True, transform=train_transform, **options)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=options['batch_size'], shuffle=True, num_workers=options['num_workers'],
            pin_memory=pin_memory)

        testset = CUB_Filter(train=False, transform=transform, **options)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
            pin_memory=pin_memory)

        outset = CUB_Filter(train=False, transform=transform, **options)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=options['batch_size'], shuffle=False, num_workers=options['num_workers'],
            pin_memory=pin_memory)

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


parser = argparse.ArgumentParser("Open Set Recognition")

if __name__ == '__main__':
    args = parser.parse_args()
    options_ = vars(args)

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        options_['use_gpu'] = True
    else:
        options_['use_gpu'] = False

    options_['image_size'] = 64
    options_['batch_size'] = 64
    options_['num_workers'] = 2
    options_['known'] = [1, 3, 5, 7, 9]
    options_['data_root'] = r'C:\Users\42941\Documents\CV_data\CUB\CUB_200_2011'

    # # ############ 获取完整类别的 CUB 数据集 Dataset，用以下代码 ###############################
    train_transform_ = Compose([
        ToCVImage(),
        RandomResizedCrop(options_['image_size']),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(TRAIN_MEAN, TRAIN_STD)
    ])
    data = CUB(train=True, transform=train_transform_, **options_)
    print(data, len(data))

    # ########## 挑选类别做 OSR 的 CUB 数据集 dataloader，用以下代码 ###############################
    data = CUB_OSR(**options_)
    print(data, data.out_loader)

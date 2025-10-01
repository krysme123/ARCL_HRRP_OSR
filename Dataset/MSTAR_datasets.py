
from torch.utils.data import Dataset
import os
from skimage import io
from PIL import Image


class OSRImage(Dataset):
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
        # label = [self.root_dir.index(self.all_dir[i]) for i in range(len(self.images))]
        # self.label = np.array(label)

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

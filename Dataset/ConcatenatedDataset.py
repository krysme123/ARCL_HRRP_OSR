from torch.utils.data import Dataset


class ConcatenatedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            sample = self.dataset1[idx]
        else:
            idx -= len(self.dataset1)
            sample = self.dataset2[idx]

        # 根据你的数据结构处理样本，例如返回元组 (data, label) 或者直接 data
        return sample

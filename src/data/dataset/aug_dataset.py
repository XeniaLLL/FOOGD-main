from src.data.data_utils import *


class AugPairDataset(Dataset):
    def __init__(self, dataset, transform, supervise=True):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.supervise = supervise

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        if self.supervise:
            return self.transform(x), self.transform(x), y
        else:
            return x, self.transform(x), y

    def __len__(self) -> int:
        return len(self.dataset)


class FourierDGDataset(Dataset):
    def __init__(self, dataset, alpha=1.0):
        self.dataset = dataset
        self.img_size = self.dataset[0][0].size[0]
        self.pre_transform = get_pre_transform(image_size=self.img_size)
        self.post_transform = get_post_transform()
        self.alpha = alpha
        self.targets = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        x_ori, y_ori = self.dataset[index]
        x_ori = self.pre_transform(x_ori)
        sample_idx = random.randint(0, len(self.dataset) - 1)
        x_sam, y_sam = self.dataset[sample_idx]
        x_sam = self.pre_transform(x_sam)
        x_s2o, x_o2s = colorful_spectrum_mix(x_ori, x_sam, alpha=self.alpha)
        x_ori, x_sam = self.post_transform(x_ori), self.post_transform(x_sam)
        x_s2o, x_o2s = self.post_transform(x_s2o), self.post_transform(x_o2s)

        x = [x_ori, x_s2o]
        y = [y_ori, y_ori]
        return x, y

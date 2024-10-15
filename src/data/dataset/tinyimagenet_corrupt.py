import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class TinyImageNet_CORRUPT(Dataset):
    def __init__(self, root: str, cortype: str, transform: transforms):
        if cortype not in ['brightness', 'fog', 'glass_blur', 'motion_blur', 'snow', 'contrast', 'frost',
                           'impulse_noise', 'pixelate', 'defocus_blur', 'jpeg_compression',
                           'elastic_transform', 'gaussian_noise', 'shot_noise',
                           'zoom_blur']:
            raise AttributeError('corrupt type is not included in TinyImageNet-C.')

        self.root_dir = os.path.join(root, 'Tiny-ImageNet-C')
        self._image_transformer = transform
        self.image_dir = os.path.join(self.root_dir, cortype)

        self._create_class_idx_dict_train()

        self._make_dataset()

    def _create_class_idx_dict_train(self):
        classes = [d.name for d in os.scandir(os.path.join(self.image_dir, '1')) if d.is_dir()]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, severity in os.walk(self.image_dir):
            for files in severity:
                for f in files:
                    if f.endswith(".JPEG"):
                        num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _make_dataset(self):
        self.images = []
        self.targets = []
        img_root_dir = self.image_dir
        list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]

        for severity in ['1', '2', '3', '4', '5']:
            tmp_dir = os.path.join(img_root_dir, severity)
            for tgt in list_of_dirs:
                dirs = os.path.join(tmp_dir, tgt)
                if not os.path.isdir(dirs):
                    continue

                for root, _, files in sorted(os.walk(dirs)):
                    for fname in sorted(files):
                        if (fname.endswith(".JPEG")):
                            path = os.path.join(root, fname)
                            item = (path, self.class_to_tgt_idx[tgt])
                            target = self.class_to_tgt_idx[tgt]
                            self.images.append(item)
                            self.targets.append(target)

    def __getitem__(self, idx: int):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self._image_transformer is not None:
            sample = self._image_transformer(sample)

        return sample, tgt

    def __len__(self) -> int:
        return self.len_dataset

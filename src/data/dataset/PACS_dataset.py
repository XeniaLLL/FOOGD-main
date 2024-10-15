import os

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class PACSDataset(Dataset):
    def __init__(self, root, dataset_name, transform=None):
        self.dataset_name = dataset_name
        dataset_path = os.path.join(root, 'PACS')
        self.dataset_path = os.path.join(dataset_path, f'{dataset_name}')
        image_folder_dataset = ImageFolder(root=self.dataset_path)
        self.dataset = image_folder_dataset

        self.num_class = 7
        self.transform = transform
        self._image_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform:
            img = self.transform(img)

        # img = self._image_transformer(img)

        return img, target

    def __len__(self):
        return len(self.dataset)

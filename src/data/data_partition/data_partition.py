import logging
import os
import random

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, ImageFolder, CIFAR100, Places365, SVHN
from torchvision.transforms import transforms

from src.data.dataset.PACS_dataset import PACSDataset
from src.data.dataset.aug_dataset import FourierDGDataset
from src.data.dataset.load_cifar10_corrupted import CIFAR10_CORRUPT, CIFAR100_CORRUPT
from src.data.dataset.tinyimagenet import TinyImageNet
from src.data.dataset.tinyimagenet_corrupt import TinyImageNet_CORRUPT


def create_dirichlet_distribution(alpha: float, num_client: int, num_class: int, seed: int):
    random_number_generator = np.random.default_rng(seed)
    distribution = random_number_generator.dirichlet(np.repeat(alpha, num_client), size=num_class).transpose()
    distribution /= distribution.sum()
    return distribution


def create_pathological_distribution(class_per_client: int, num_client: int, num_class: int, seed: int):
    random_number_generator = np.random.default_rng(seed)
    repeat_count = (num_client * class_per_client + num_class - 1) // num_class
    classes_sequence = []
    classes = np.array([i for i in range(num_class)])
    for _ in range(repeat_count):
        random_number_generator.shuffle(classes)
        classes_sequence.extend(classes.tolist())
    clients_classes = [
        classes_sequence[i:(i + class_per_client)]
        for i in range(0, num_client * class_per_client, class_per_client)]
    distribution = np.zeros((num_client, num_class))
    for cid in range(num_client):
        for class_idx in clients_classes[cid]:
            distribution[cid, class_idx] = 1
    for class_idx in range(num_class):
        distribution[:, class_idx] /= distribution[:, class_idx].sum()
    distribution /= distribution.sum()
    return distribution


def split_by_distribution(targets, distribution):
    num_client, num_class = distribution.shape[0], distribution.shape[1]
    sample_number = np.floor(distribution * len(targets))
    class_idx = {class_label: np.where(targets == class_label)[0] for class_label in range(num_class)}

    idx_start = np.zeros((num_client + 1, num_class), dtype=np.int32)
    for i in range(0, num_client):
        idx_start[i + 1] = idx_start[i] + sample_number[i]

    client_samples = {idx: {} for idx in range(num_client)}
    for client_idx in range(num_client):
        samples_idx = np.array([], dtype=np.int32)
        for class_label in range(num_class):
            start, end = idx_start[client_idx, class_label], idx_start[client_idx + 1, class_label]
            samples_idx = (np.concatenate((samples_idx, class_idx[class_label][start:end].tolist())).astype(np.int32))
        client_samples[client_idx] = samples_idx

    return client_samples


def load_ood_dataset(dataset_path: str, ood_dataset: str):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    if ood_dataset == 'LSUN_C':
        ood_data = ImageFolder(
            root=os.path.join(dataset_path, 'LSUN'),
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomCrop(32, padding=4)
            ])
        )
    elif ood_dataset == 'LSUN-R':
        ood_data = ImageFolder(
            root=os.path.join(dataset_path, 'LSUN_resize'),
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )
    elif ood_dataset == 'Texture':
        ood_data = ImageFolder(
            root=os.path.join(dataset_path, 'dtd/images'),
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        )
    elif ood_dataset == 'isun':
        ood_data = ImageFolder(
            root=os.path.join(dataset_path, 'iSUN'),
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        )
    elif ood_dataset == 'place365':
        ood_data = Places365(
            root=dataset_path,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        )
    elif ood_dataset == 'SVHN':
        ood_data = SVHN(
            root=dataset_path,
            split='test',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            download=True
        )
    else:
        raise NotImplementedError('out of distribution dataset should be LSUN_C, dtd, isun')

    return ood_data


def load_PACS_train(dataset_path, leave_out):
    domain_datasets = dict()
    dataset_names = ['art_painting', 'cartoon', 'photo', 'sketch']
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for name in dataset_names:
        train_data = PACSDataset(dataset_path, name, transform=None)
        # train_data = PACSDataset(dataset_path, name, transform=trans)
        domain_datasets[name] = FourierDGDataset(train_data, 1.0)
    num_class = 7
    train_datasets = []
    for name in dataset_names:
        if name == leave_out:
            continue
        train_datasets.append(domain_datasets[name])
    return train_datasets, num_class

def load_PACS_test(dataset_path, leave_out):
    domain_datasets = dict()
    dataset_names = ['art_painting', 'cartoon', 'photo', 'sketch']
    trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for name in dataset_names:
        domain_datasets[name] = PACSDataset(dataset_path, name, transform=trans)
    num_class = 7
    train_datasets = [domain_datasets[leave_out] for _ in range(3)]
    return train_datasets, None, num_class


def dirichlet_load_train(dataset_path, id_dataset, num_client, alpha, seed, fourier_mix_alpha=1.0):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    if id_dataset == 'cifar10':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = CIFAR10(root=dataset_path, download=True, train=True, transform=trans)
        num_class = 10
    elif id_dataset == 'cifar10_fourier_aug':
        train_data = CIFAR10(root=dataset_path, download=True, train=True)  # careful: no transform for original data
        train_data = FourierDGDataset(train_data, fourier_mix_alpha)  # careful: 先只导入数据,后续在fourier 增广中调整
        num_class = 10
    elif id_dataset == 'cifar100':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = CIFAR100(root=dataset_path, download=True, train=True, transform=trans)
        num_class = 100
    elif id_dataset == 'cifar100_fourier_aug':
        train_data = CIFAR100(root=dataset_path, download=True, train=True)  # careful: no transform for original data
        train_data = FourierDGDataset(train_data, fourier_mix_alpha)  # careful: 先只导入数据,后续在fourier 增广中调整
        num_class = 100
    elif id_dataset == 'tinyimagenet':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = TinyImageNet(root=dataset_path, train=True, transform=trans)
        num_class = 200
    elif id_dataset == 'tinyimagenet_fourier_aug':
        train_data = TinyImageNet(root=dataset_path, train=True)
        train_data = FourierDGDataset(train_data, fourier_mix_alpha)
        num_class = 200
    else:
        raise NotImplementedError('in distribution dataset should be CIFAR10 or CIFAR100.')

    distribution = create_dirichlet_distribution(alpha, num_client, num_class, seed)
    train_split = split_by_distribution(np.array(train_data.targets), distribution)
    train_datasets = [Subset(train_data, train_split[idx]) for idx in range(num_client)]

    logging.info(f'-------- dirichlet distribution with alpha = {alpha}, {num_client} clients --------')
    logging.info(f'in-distribution train datasets: {[len(dataset) for dataset in train_datasets]}')
    return train_datasets, num_class


def pathological_load_train(dataset_path, id_dataset, num_client, class_per_client, seed):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    if id_dataset == 'cifar10':  # todo add fourier aug if work
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = CIFAR10(root=dataset_path, download=True, train=True, transform=trans)
        num_class = 10
    elif id_dataset == 'cifar100':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = CIFAR100(root=dataset_path, download=True, train=True, transform=trans)
        num_class = 100
    elif id_dataset == 'tinyimagenet':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = TinyImageNet(root=dataset_path, train=True, transform=trans)
        num_class = 200
    else:
        raise NotImplementedError('in distribution dataset should be CIFAR10 or CIFAR100.')

    distribution = create_pathological_distribution(class_per_client, num_client, num_class, seed)
    id_train_split = split_by_distribution(np.array(train_data.targets), distribution)
    train_datasets = [Subset(train_data, id_train_split[idx]) for idx in range(num_client)]

    logging.info(
        f'-------- pathological distribution with {class_per_client} classes per client, {num_client} clients --------')
    logging.info(f'in-distribution train datasets: {[len(dataset) for dataset in train_datasets]}')

    return train_datasets, num_class


def dirichlet_load_test(dataset_path, id_dataset, num_client, alpha, corrupt_list, seed):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    if (id_dataset == 'cifar10') or (id_dataset == 'cifar10_fourier_aug'):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_data = CIFAR10(root=dataset_path, download=True, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(CIFAR10_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 10
    elif (id_dataset == 'cifar100') or (id_dataset == 'cifar100_fourier_aug'):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_data = CIFAR100(root=dataset_path, download=True, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(CIFAR100_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 100
    elif id_dataset == 'tinyimagenet' or id_dataset == "tinyimagenet_fourier_aug":
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_data = TinyImageNet(root=dataset_path, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(TinyImageNet_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 200
    else:
        raise NotImplementedError('in distribution dataset should be CIFAR10 or CIFAR100.')

    distribution = create_dirichlet_distribution(alpha, num_client, num_class, seed)
    id_split = split_by_distribution(np.array(test_data.targets), distribution)
    cor_split = split_by_distribution(np.array(cor_test[0].targets), distribution)
    id_datasets = [Subset(test_data, id_split[idx]) for idx in range(num_client)]
    cor_datasets = [
        {cor_type: Subset(cor_test[idx], cor_split[client_idx]) for idx, cor_type in enumerate(corrupt_list)}
        for client_idx in range(num_client)]

    logging.info(f'-------- dirichlet distribution with alpha = {alpha}, {num_client} clients --------')
    logging.info(f'in-distribution test datasets: {[len(dataset) for dataset in id_datasets]}')
    return id_datasets, cor_datasets, num_class


def pathological_load_test(dataset_path, id_dataset, num_client, class_per_client, corrupt_list, seed):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if id_dataset == 'cifar10':
        test_data = CIFAR10(root=dataset_path, download=True, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(CIFAR10_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 10
    elif id_dataset == 'cifar100':
        test_data = CIFAR100(root=dataset_path, download=True, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(CIFAR100_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 100
    elif id_dataset == 'tinyimagenet':
        test_data = TinyImageNet(root=dataset_path, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(TinyImageNet_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 200
    else:
        raise NotImplementedError('in distribution dataset should be CIFAR10 or CIFAR100.')

    distribution = create_pathological_distribution(class_per_client, num_client, num_class, seed)
    id_split = split_by_distribution(np.array(test_data.targets), distribution)
    cor_split = split_by_distribution(np.array(cor_test[0].targets), distribution)
    id_datasets = [Subset(test_data, id_split[idx]) for idx in range(num_client)]
    cor_datasets = [
        {cor_type: Subset(cor_test[idx], cor_split[client_idx]) for idx, cor_type in enumerate(corrupt_list)}
        for client_idx in range(num_client)]

    logging.info(
        f'-------- pathological distribution with {class_per_client} classes per client, {num_client} clients --------')
    logging.info(f'in-distribution test datasets: {[len(dataset) for dataset in id_datasets]}')
    return id_datasets, cor_datasets, num_class


def load_test_ood(dataset_path, ood_dataset, seed, partial):
    random_number_generator = np.random.default_rng(seed)
    ood_data = load_ood_dataset(dataset_path, ood_dataset)

    if partial:
        idx = random.sample([i for i in range(len(ood_data))], int(0.2 * len(ood_data)))
        ood_data = Subset(ood_data, idx)
        logging.info(f'out of distribution test dataset\'s length: {len(ood_data)}')
        return ood_data
    else:
        logging.info(f'out of distribution test dataset\'s length: {len(ood_data)}')
        return ood_data

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.data.data_partition.data_partition import (
    pathological_load_test,
    dirichlet_load_test,
    load_test_ood,
    dirichlet_load_train, load_PACS_test, load_PACS_train,
)
from src.models.resnet import ResNet18
from src.models.wideresnet import WideResNet
from src.utils.accuracy import compute_fnr, compute_auroc
from src.utils.main_utils import make_save_path, set_seed
from src.models.score import Energy, MLPScore

parser = argparse.ArgumentParser(description="arguments for OOD generalization and detection training")

parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--method", type=str, default="T3A")
parser.add_argument("--model_name", type=str, default="model_round100.pt")
# ---------- dataset partition ----------
parser.add_argument("--id_dataset", type=str, default="PACS", help="the ID dataset")
parser.add_argument('--leave_out', type=str, default='sketch')
parser.add_argument("--ood_dataset", type=str, default="LSUN_C")
parser.add_argument("--dataset_path", type=str, default="../datasets/", help="path to dataset")
parser.add_argument("--alpha", type=float, default=0.1, help="parameter of dirichlet distribution")
parser.add_argument("--num_client", type=int, default=10, help="number of clients")
parser.add_argument("--dataset_seed", type=int, default=21, help="seed to split dataset")
parser.add_argument("--pathological", type=bool, default=False, help="using pathological method split dataset")
parser.add_argument("--class_per_client", type=int, default=2, help="classes per client")
# ---------- backbone ----------
parser.add_argument(
    "--backbone", type=str, choices=["resnet", "wideresnet"], default="resnet", help="backbone model of task"
)
# ---------- device ----------
parser.add_argument("--device", type=str, default="cuda:0", help="device")
# ---------- server configuration ----------
parser.add_argument("--checkpoint_path", type=str, default="default", help="check point path")

# ---------- optimizer --------
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=0.0005)
parser.add_argument("--use_score_model", type=bool, default=False)
parser.add_argument(
    "--score_method", type=str, choices=["msp", "energy", 'max_logit', 'ash' "sm"], default="msp"
)  # sm denots score matching
parser.add_argument('--ODG_noise_type', type=str, choices=['gaussian', 'radermacher','sphere'], default='gaussian', help='score model noise type')
parser.add_argument('--ODG_loss_types', type=str, choices=['ssm-vr', 'ssm', 'dsm','deen', 'anneal_dsm'], default='anneal_dsm', help='score model loss type')
parser.add_argument('--ODG_sampler', type=str, choices=['ld', 'ald'], default='ld', help='score model sample type')
parser.add_argument('--ODG_sample_steps', type=int, default=10, help='Langiven sample steps')
parser.add_argument('--ODG_n_slices', type=int, default=0, help='special for sliced score matching')
parser.add_argument('--ODG_mmd_kernel_num', type=int, default=2, help='number of MMD loss')
parser.add_argument('--ODG_sample_eps', type=float, default=0.01, help='Langiven sample epsilon size')
parser.add_argument('--ODG_score_learning_rate', type=float, default=0.001)
parser.add_argument('--ODG_score_momentum', type=float, default=0.)
parser.add_argument('--ODG_score_weight_decay', type=float, default=0.)
parser.add_argument('--ODG_sigma_begin', type=float, default=0.01)
parser.add_argument('--ODG_sigma_end', type=float, default=1)
parser.add_argument('--ODG_anneal_power', type=float, default=2)
parser.add_argument("--ODG_sam_rho",  type=float, default=0.5, help="hyper-param for sam& asam ")
parser.add_argument("--ODG_sam_eta", type=float, default=0.2, help="hyper-param for asam ")
parser.add_argument("--ODG_sam_adaptive", type=bool, default=True, help="hyper-param for asam ")
parser.add_argument('--num_classes', type=int, default=10, help='number of dataset classes')

def test_classification_detection_ability(backbone, in_loader, ood_loader, device, score_method="msp"):
    backbone.to(device)
    backbone.eval()

    ood_score_id = []
    ood_score_ood = []
    accuracy = []

    def ash_b(x, percentile=65):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])

        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        fill = s1 / k
        fill = fill.unsqueeze(dim=1).expand(v.shape)
        t.zero_().scatter_(dim=1, index=i, src=fill)
        return x.view(b, -1)

    with torch.no_grad():
        for data, target in in_loader:
            data, target = data.to(device), target.to(device)
            feature = backbone.intermediate_forward(data)

            if score_method == 'ash':
                feature = ash_b(feature)

            logit = backbone.fc(feature)
            pred = logit.data.max(1)[1]
            accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

            if score_method == "energy":
                ood_score_id.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
            elif score_method == "msp":
                ood_score_id.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))
            elif score_method == 'max_logit':
                ood_score_id.extend(list(np.max(logit.data.cpu().numpy(), axis=1)))
            elif score_method == 'vim':
                pass
            elif score_method == 'ash':
                ood_score_id.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))

        for data, _ in ood_loader:
            data = data.to(device)
            feature = backbone.intermediate_forward(data)

            if score_method == 'ash':
                feature = ash_b(feature)

            logit = backbone.fc(feature)
            if score_method == "energy":
                ood_score_ood.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
            elif score_method == "msp":
                ood_score_ood.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))
            elif score_method == 'max_logit':
                ood_score_ood.extend(list(np.max(logit.data.cpu().numpy(), axis=1)))
            elif score_method == 'vim':
                pass
            elif score_method == 'ash':
                ood_score_ood.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))

    backbone.cpu()

    if score_method in ["energy", 'ash', 'vim']:
        fpr95 = compute_fnr(np.array(ood_score_ood), np.array(ood_score_id))
        auroc = compute_auroc(np.array(ood_score_ood), np.array(ood_score_id))
    elif score_method in ["msp", 'max_logit']:
        fpr95 = compute_fnr(np.array(ood_score_id), np.array(ood_score_ood))
        auroc = compute_auroc(np.array(ood_score_id), np.array(ood_score_ood))

    id_accuracy = sum(accuracy) / len(accuracy)

    return id_accuracy, fpr95, auroc

def test_classification_detection_ability_t3a(backbone, in_loader, ood_loader, device, score_method="msp"):
    backbone.to(device)
    backbone.eval()

    ood_score_id = []
    ood_score_ood = []
    accuracy = []

    def ash_b(x, percentile=65):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])

        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        fill = s1 / k
        fill = fill.unsqueeze(dim=1).expand(v.shape)
        t.zero_().scatter_(dim=1, index=i, src=fill)
        return x.view(b, -1)

    from src.algorithms.T3A import T3A

    adapt_algorithm = T3A(backbone, 50)

    with torch.no_grad():
        for data, target in in_loader:
            data, target = data.to(device), target.to(device)
            feature = backbone.intermediate_forward(data)

            if score_method == 'ash':
                feature = ash_b(feature)

            logit = adapt_algorithm(data)
            pred = logit.data.max(1)[1]
            accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

            if score_method == "energy":
                ood_score_id.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
            elif score_method == "msp":
                ood_score_id.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))
            elif score_method == 'max_logit':
                ood_score_id.extend(list(np.max(logit.data.cpu().numpy(), axis=1)))
            elif score_method == 'vim':
                pass
            elif score_method == 'ash':
                ood_score_id.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))

        for data, _ in ood_loader:
            data = data.to(device)
            feature = backbone.intermediate_forward(data)

            if score_method == 'ash':
                feature = ash_b(feature)

            logit = backbone.fc(feature)
            if score_method == "energy":
                ood_score_ood.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
            elif score_method == "msp":
                ood_score_ood.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))
            elif score_method == 'max_logit':
                ood_score_ood.extend(list(np.max(logit.data.cpu().numpy(), axis=1)))
            elif score_method == 'vim':
                pass
            elif score_method == 'ash':
                ood_score_ood.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))

    backbone.cpu()

    if score_method in ["energy", 'ash', 'vim']:
        fpr95 = compute_fnr(np.array(ood_score_ood), np.array(ood_score_id))
        auroc = compute_auroc(np.array(ood_score_ood), np.array(ood_score_id))
    elif score_method in ["msp", 'max_logit']:
        fpr95 = compute_fnr(np.array(ood_score_id), np.array(ood_score_ood))
        auroc = compute_auroc(np.array(ood_score_id), np.array(ood_score_ood))

    id_accuracy = sum(accuracy) / len(accuracy)

    return id_accuracy, fpr95, auroc

def test_classification_detection_ability_odg(backbone, score_model, in_loader, ood_loader, device, score_method="msp"):
    backbone.to(device)
    score_model.to(device)
    backbone.eval()
    score_model.eval()

    ood_score_id = []
    ood_score_ood = []
    accuracy = []

    with torch.no_grad():
        for data, target in in_loader:
            data, target = data.to(device), target.to(device)
            latents = backbone.intermediate_forward(data)
            logit = backbone(data)
            scores = score_model(latents).norm(dim=-1)
            # logit = logit / scores.unsqueeze(1)
            pred = logit.data.max(1)[1]
            accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

            if score_method == "energy":
                # ood_score_id.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
                ood_score_id.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
            elif score_method == "msp":
                ood_score_id.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))
            elif score_method == "sm":
                ood_score_id.extend(list(scores.data.cpu().numpy()))

        for data, _ in ood_loader:
            data = data.to(device)
            latents = backbone.intermediate_forward(data)
            logit = backbone(data)
            scores = score_model(latents).norm(dim=-1)
            # logit = logit / scores.unsqueeze(1)
            if score_method == "energy":
                ood_score_ood.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
            elif score_method == "msp":
                ood_score_ood.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))
            elif score_method == "sm":
                ood_score_ood.extend(list(scores.data.cpu().numpy()))

    backbone.cpu()

    if score_method == "energy":
        fpr95 = compute_fnr(np.array(ood_score_ood), np.array(ood_score_id))
        auroc = compute_auroc(np.array(ood_score_ood), np.array(ood_score_id))
    elif score_method == "msp":
        fpr95 = compute_fnr(np.array(ood_score_id), np.array(ood_score_ood))
        auroc = compute_auroc(np.array(ood_score_id), np.array(ood_score_ood))
    elif score_method == "sm":
        fpr95 = compute_fnr(np.array(ood_score_ood), np.array(ood_score_id))
        auroc = compute_auroc(np.array(ood_score_ood), np.array(ood_score_id))

    id_accuracy = sum(accuracy) / len(accuracy)

    return id_accuracy, fpr95, auroc


@torch.no_grad()
def test_corrupt_accuracy(backbone, cor_loader, device):
    cor_accuracy = dict()
    backbone.to(device)
    backbone.eval()

    for cor_type, cor_loader in cor_loader.items():
        accuracy = []
        for data, target in cor_loader:
            data, target = data.to(device), target.to(device)
            logit = backbone(data)
            pred = logit.data.max(1)[1]
            accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))
        cor_accuracy[cor_type] = sum(accuracy) / len(accuracy)

    return cor_accuracy


@torch.no_grad()
def test_corrupt_accuracy_t3a(backbone, cor_loader, device):
    backbone.to(device)
    backbone.eval()
    from src.algorithms.T3A import T3A

    adapt_algorithm = T3A(backbone, 100)
    cor_accuracy = dict()
    for cor_type, cor_loader in cor_loader.items():
        adapt_algorithm.reset()
        accuracy = []
        for data, target in cor_loader:
            data, target = data.to(device), target.to(device)
            logit = adapt_algorithm(data)
            pred = logit.data.max(1)[1]
            accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))
        cor_accuracy[cor_type] = sum(accuracy) / len(accuracy)

    return cor_accuracy


def test():
    args = parser.parse_args()
    set_seed(args.seed)
    save_path = make_save_path(args)
    file_name = os.path.join(save_path, f"{args.model_name}_{args.score_method}_test.log")

    print(args.method, args.id_dataset, args.alpha)
    if args.method == "T3A":
        save_path = os.path.join(save_path, f"../FedAvg_{args.backbone}")
        file_name = os.path.join(save_path, f"{args.model_name}_t3a_test.log")

    logging.basicConfig(
        level=logging.INFO,
        filename=file_name,
        format="[%(asctime)s %(levelname)s] %(message)s",
        filemode="w",
    )
    if args.checkpoint_path == "default":
        setattr(args, "checkpoint_path", save_path)
    if args.use_score_model:
        args.score_model = Energy(net=MLPScore())
    logging.info(f"-------------------- configuration --------------------")
    for key, value in args._get_kwargs():
        logging.info(f"configuration {key}: {value}")

    corrupt_list = [
        "brightness",
        # "fog",
        # "glass_blur",
        # "motion_blur",
        # "snow",
        # "contrast",
        # "frost",
        # "impulse_noise",
        # "pixelate",
        # "defocus_blur",
        # "jpeg_compression",
        # "elastic_transform",
        # "gaussian_noise",
        # "shot_noise",
        # "zoom_blur",
    ]
    if args.id_dataset in ['cifar10', 'cifar100']:
        corrupt_list.extend(['spatter', 'gaussian_blur', 'saturate', 'speckle_noise'])

    if args.id_dataset == 'PACS':
        id_datasets, cor_datasets, num_class = load_PACS_test(args.dataset_path, args.leave_out)
        setattr(args, 'num_client', 3)
    else:
        if args.pathological:
            id_datasets, cor_datasets, num_class = pathological_load_test(
                args.dataset_path, args.id_dataset, args.num_client, args.class_per_client, corrupt_list, args.dataset_seed
            )
        else:
            id_datasets, cor_datasets, num_class = dirichlet_load_test(
                args.dataset_path, args.id_dataset, args.num_client, args.alpha, corrupt_list, args.dataset_seed
            )
    ood_dataset = load_test_ood(args.dataset_path, args.ood_dataset, args.dataset_seed, False)

    if args.backbone == "resnet":
        backbone = ResNet18(num_classes=num_class)
    elif args.backbone == "wideresnet":
        backbone = WideResNet(depth=40, num_classes=num_class, widen_factor=2, dropRate=0.3)
    else:
        raise NotImplementedError("backbone should be ResNet or WideResNet")

    device = torch.device(args.device)

    if args.method in ["FedAvg", "LogitNorm", "FedIIR", "FedATOL"]:
        backbone.load_state_dict(torch.load(os.path.join(save_path, f"{args.model_name}")))
        clients_id_accuracies = []
        clients_fpr95 = []
        clients_auroc = []
        clients_cor_accuracies = []
        weights = []
        for cid in range(args.num_client):
            id_loader = DataLoader(
                dataset=id_datasets[cid],
                batch_size=128,
                shuffle=True,
            )
            ood_loader = DataLoader(dataset=ood_dataset, batch_size=128, shuffle=True)
            cor_loader = dict()
            if args.id_dataset != 'PACS':
                for cor_type in corrupt_list:
                    cor_loader[cor_type] = DataLoader(
                        dataset=cor_datasets[cid][cor_type],
                        batch_size=128,
                        shuffle=True,
                    )
            weights.append(len(id_loader))

            id_accuracy, fpr95, auroc = test_classification_detection_ability(
                backbone, id_loader, ood_loader, device, args.score_method
            )
            if args.id_dataset != 'PACS':
                cor_accuracy = test_corrupt_accuracy(backbone, cor_loader, device)
            clients_id_accuracies.append(id_accuracy)
            clients_fpr95.append(fpr95)
            clients_auroc.append(auroc)
            if args.id_dataset != 'PACS':
                clients_cor_accuracies.append(cor_accuracy)

        logging.info(
            f"test in distribution accuracy: {sum([clients_id_accuracies[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        logging.info(
            f"test fpr95: {sum([clients_fpr95[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        logging.info(
            f"test auroc: {sum([clients_auroc[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        if args.id_dataset != 'PACS':
            for key in clients_cor_accuracies[0].keys():
                logging.info(
                    f"corrupt type {key} accuracy: {sum([clients_cor_accuracies[cid][key] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
                )
    elif args.method in ["FedRoD", "Ditto", "FedTHE", "FedICON", "FOSTER"]:
        client_id_loaders = [
            DataLoader(dataset=id_datasets[idx], batch_size=128, shuffle=True) for idx in range(args.num_client)
        ]
        ood_loader = DataLoader(dataset=ood_dataset, batch_size=128, shuffle=True)

        if args.id_dataset != 'PACS':
            train_datasets, num_class = dirichlet_load_train(
                args.dataset_path, args.id_dataset, args.num_client, args.alpha, args.dataset_seed
            )
        else:
            train_datasets, num_class = load_PACS_train(args.dataset_path, args.leave_out)
        server_args = {
            "join_ratio": 1.0,
            "checkpoint_path": args.checkpoint_path,
            "backbone": backbone,
            "device": device,
            "debug": args.debug,
            "use_score_model": args.use_score_model,
            "score_model": Energy(net=MLPScore()) if args.use_score_model else None,
            "alpha": args.alpha,
            "id_dataset": args.id_dataset,
        }
        client_args = [
            {
                "cid": cid,
                "device": device,
                "epochs": 5,
                "backbone": backbone,
                "batch_size": 128,
                "num_workers": 0,
                "pin_memory": False,
                "train_id_dataset": train_datasets[cid],
                "learning_rate": 0.1,
                "weight_decay": 0.0005,
                "momentum": 0.9,
                "use_score_model": args.use_score_model,
                "score_model": Energy(net=MLPScore()) if args.use_score_model else None,
                "score_learning_rate": 0.0001,
                "score_weight_decay": 0,
                "ODG_mmd_kernel_num": 2,
                "num_classes": backbone.fc.out_features,
            }
            for cid in range(args.num_client)
        ]
        from src.utils.main_utils import get_server_and_client

        Server, Client, client_args, server_args = get_server_and_client(args, client_args, server_args)
        server = Server(server_args)
        clients = [Client(client_args[idx]) for idx in range(args.num_client)]
        # from src.algorithms.FedRoD.server_fedrod import FedRoDServer, FedRoDClient
        # server = FedRoDServer(server_args)
        # clients = [FedRoDClient(client_args[idx]) for idx in range(args.num_client)]
        server.clients.extend(clients)
        checkpoint = torch.load(os.path.join(save_path, f"{args.model_name}"))
        client_cor_loaders = dict()
        if args.id_dataset != 'PACS':
            for cor_type in corrupt_list:
                client_cor_loaders[cor_type] = [
                    DataLoader(
                        dataset=cor_datasets[cid][cor_type],
                        batch_size=128,
                        shuffle=True,
                    )
                    for cid in range(args.num_client)
                ]
        id_accuracy, fpr95, auroc = server.test_classification_detection_ability(
            checkpoint, client_id_loaders, ood_loader, args.score_method
        )
        if args.id_dataset != 'PACS':
            cor_accuracy = server.test_corrupt_accuracy(client_cor_loaders)
        logging.info(f"test in distribution accuracy: {id_accuracy}")
        logging.info(f"test fpr95: {fpr95}")
        logging.info(f"test auroc: {auroc}")
        if args.id_dataset != 'PACS':
            for key, value in cor_accuracy.items():
                logging.info(f"corrupt type {key} accuracy: {value}")
        print(f"{args.id_dataset}", f"{args.alpha}")
        print(f"id_accuracy: {id_accuracy}, fpr95: {fpr95}, auroc: {auroc}")
        if args.id_dataset != 'PACS':
            for key, value in cor_accuracy.items():
                print(f"corrupt type {key} accuracy: {value}")
    elif args.method == "T3A":
        backbone.load_state_dict(torch.load(os.path.join(save_path, f"{args.model_name}")))
        clients_id_accuracies = []
        clients_fpr95 = []
        clients_auroc = []
        clients_cor_accuracies = []
        weights = []
        for cid in range(args.num_client):
            id_loader = DataLoader(
                dataset=id_datasets[cid],
                batch_size=128,
                shuffle=True,
            )
            ood_loader = DataLoader(dataset=ood_dataset, batch_size=128, shuffle=True)
            cor_loader = dict()
            if args.id_dataset != 'PACS':
                for cor_type in corrupt_list:
                    cor_loader[cor_type] = DataLoader(
                        dataset=cor_datasets[cid][cor_type],
                        batch_size=32,
                        shuffle=True,
                    )
            weights.append(len(id_loader))

            id_accuracy, fpr95, auroc = test_classification_detection_ability_t3a(
                backbone, id_loader, ood_loader, device, args.score_method
            )
            if args.id_dataset != 'PACS':
                cor_accuracy = test_corrupt_accuracy_t3a(backbone, cor_loader, device)
            clients_id_accuracies.append(id_accuracy)
            clients_fpr95.append(fpr95)
            clients_auroc.append(auroc)
            if args.id_dataset != 'PACS':
                clients_cor_accuracies.append(cor_accuracy)

        logging.info(
            f"test in distribution accuracy: {sum([clients_id_accuracies[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        logging.info(
            f"test fpr95: {sum([clients_fpr95[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        logging.info(
            f"test auroc: {sum([clients_auroc[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        if args.id_dataset != 'PACS':
            for key in clients_cor_accuracies[0].keys():
                logging.info(
                    f"corrupt type {key} accuracy: {sum([clients_cor_accuracies[cid][key] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
                )
    elif args.method in ["FOOGD"]:
        model_states = torch.load(os.path.join(save_path, f"{args.model_name}"))
        backbone.load_state_dict(model_states[0])
        # ---------- construct customized model ----------
        score_model = Energy(net=MLPScore())
        score_model.load_state_dict(model_states[1])
        clients_id_accuracies = []
        clients_fpr95 = []
        clients_auroc = []
        clients_cor_accuracies = []
        weights = []
        for cid in range(args.num_client):
            id_loader = DataLoader(
                dataset=id_datasets[cid],
                batch_size=128,
                shuffle=True,
            )
            ood_loader = DataLoader(dataset=ood_dataset, batch_size=128, shuffle=True)
            cor_loader = dict()
            if args.id_dataset != 'PACS':
                for cor_type in corrupt_list:
                    cor_loader[cor_type] = DataLoader(
                        dataset=cor_datasets[cid][cor_type],
                        batch_size=128,
                        shuffle=True,
                    )
            weights.append(len(id_loader))

            id_accuracy, fpr95, auroc = test_classification_detection_ability_odg(
                backbone, score_model, id_loader, ood_loader, device, score_method=args.score_method
            )
            if args.id_dataset != 'PACS':
                cor_accuracy = test_corrupt_accuracy(backbone, cor_loader, device)
            clients_id_accuracies.append(id_accuracy)
            clients_fpr95.append(fpr95)
            clients_auroc.append(auroc)
            if args.id_dataset != 'PACS':
                clients_cor_accuracies.append(cor_accuracy)

        logging.info(
            f"test in distribution accuracy: {sum([clients_id_accuracies[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        logging.info(
            f"test fpr95: {sum([clients_fpr95[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        logging.info(
            f"test auroc: {sum([clients_auroc[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        if args.id_dataset != 'PACS':
            for key in clients_cor_accuracies[0].keys():
                logging.info(
                    f"corrupt type {key} accuracy: {sum([clients_cor_accuracies[cid][key] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
                )

        print(
            f"test in distribution accuracy: {sum([clients_id_accuracies[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        print(
            f"test fpr95: {sum([clients_fpr95[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        print(
            f"test auroc: {sum([clients_auroc[cid] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
        )
        if args.id_dataset != 'PACS':
            for key in clients_cor_accuracies[0].keys():
                print(
                    f"corrupt type {key} accuracy: {sum([clients_cor_accuracies[cid][key] * weights[cid] / sum(weights) for cid in range(args.num_client)])}"
                )
    else:
        raise AttributeError("")


if __name__ == "__main__":
    test()

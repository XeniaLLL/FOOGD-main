import os
import random

import numpy as np
import torch
import copy


def make_save_path(args):
    root_path = os.path.dirname(__file__)
    root_path = os.path.join(os.path.dirname(root_path), "results")
    if args.id_dataset == 'PACS':
        dataset_setting = f'{args.id_dataset}_{args.leave_out}'
    else:
        dataset_setting = f"{args.id_dataset}_{args.alpha}alpha_{args.num_client}clients"
    task_path = dataset_setting + f"/{args.method}_{args.backbone}"
    save_path = os.path.abspath(os.path.join(root_path, task_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_server_and_client(args, client_args, server_args):
    server = None
    client = None
    if args.method == "FedAvg":
        from src.algorithms.FedAvg.client_fedavg import FedAvgClient
        from src.algorithms.FedAvg.server_fedavg import FedAvgServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
        server = FedAvgServer
        client = FedAvgClient

    elif args.method == "FedRoD":
        from src.algorithms.FedRoD.client_fedrod import FedRoDClient
        from src.algorithms.FedRoD.server_fedrod import FedRoDServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
            if args.use_score_model:
                client_args[cid]["learning_rate"] = args.learning_rate
                client_args[cid]["momentum"] = args.momentum
                client_args[cid]["weight_decay"] = args.weight_decay
                client_args[cid]["ODG_noise_type"] = args.ODG_noise_type
                client_args[cid]["ODG_loss_types"] = args.ODG_loss_types
                client_args[cid]["ODG_sampler"] = args.ODG_sampler
                client_args[cid]["ODG_n_slices"] = args.ODG_n_slices
                client_args[cid]["ODG_sample_steps"] = args.ODG_sample_steps
                client_args[cid]["ODG_mmd_kernel_num"] = args.ODG_mmd_kernel_num
                client_args[cid]["ODG_sample_eps"] = args.ODG_sample_eps
                client_args[cid]["score_model"] = copy.deepcopy(args.score_model)
                client_args[cid]["score_learning_rate"] = args.ODG_score_learning_rate
                client_args[cid]["score_momentum"] = args.ODG_score_momentum
                client_args[cid]["score_weight_decay"] = args.ODG_score_weight_decay
                client_args[cid]["num_classes"] = args.num_classes
                client_args[cid]["ODG_sigma_begin"] = args.ODG_sigma_begin
                client_args[cid]["ODG_sigma_end"] = args.ODG_sigma_end
                client_args[cid]["ODG_anneal_power"] = args.ODG_anneal_power
                client_args[cid]["ODG_sam_rho"] = args.ODG_sam_rho
                client_args[cid]["ODG_sam_eta"] = args.ODG_sam_eta
                client_args[cid]["ODG_sam_adaptive"] = args.ODG_sam_adaptive
        server = FedRoDServer
        client = FedRoDClient
    elif args.method == "LogitNorm":
        from src.algorithms.fed_logitnorm.client_logitnorm import FedLogitNormClient
        from src.algorithms.fed_logitnorm.server_logitnorm import FedLogitNormServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
        server = FedLogitNormServer
        client = FedLogitNormClient
    elif args.method == "FOSTER":
        from src.algorithms.FOSTER.client_foster import FOSTERClient
        from src.algorithms.FOSTER.server_foster import FOSTERServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
        server = FOSTERServer
        client = FOSTERClient
    elif args.method == "FedT3A":
        pass
    elif args.method == 'FedATOL':
        from src.algorithms.fedATOL.client_fedatol import FedATOLClient
        from src.algorithms.fedATOL.server_fedatol import FedATOLServer
        from src.models.atol_generator import Generator
        generator = Generator(args.latent_dim, 64, 3)
        generator.load_state_dict(torch.load(f'./pretrained_model/dcgan_{args.id_dataset}.pt'))
        server_args['generator'] = generator
        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
            client_args[cid]['generator'] = generator
            client_args[cid]['num_class'] = 10 if args.id_dataset == 'cifar10' else 100
            client_args[cid]['latent_dim'] = args.latent_dim
            client_args[cid]['mean'] = args.mean
            client_args[cid]['std'] = args.std
            client_args[cid]['ood_space_size'] = args.ood_space_size
            client_args[cid]['trade_off'] = args.trade_off
        server = FedATOLServer
        client = FedATOLClient
    elif args.method == "FedIIR":
        from src.algorithms.FedIIR.client_fediir import FedIIRClient
        from src.algorithms.FedIIR.server_fediir import FedIIRServer

        server_args["ema"] = args.ema
        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
            client_args[cid]["penalty"] = args.penalty
        server = FedIIRServer
        client = FedIIRClient
    elif args.method == "FedTHE":
        from src.algorithms.FedTHE.client_fedthe import FedTHEClient
        from src.algorithms.FedTHE.server_fedthe import FedTHEServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
        server = FedTHEServer
        client = FedTHEClient
    elif args.method == "FedICON":
        from src.algorithms.FedICON.client_fedicon import FedICONClient
        from src.algorithms.FedICON.server_fedicon import FedICONServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
        server = FedICONServer
        client = FedICONClient

    elif args.method.lower() == "FOOGD":
        from src.algorithms.FOOGD import ODGClient, ODGServer

        server_args["score_model"] = args.score_model
        server_args["num_classes"] = args.num_classes
        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
            client_args[cid]["ODG_noise_type"] = args.ODG_noise_type
            client_args[cid]["ODG_loss_types"] = args.ODG_loss_types
            client_args[cid]["ODG_sampler"] = args.ODG_sampler
            client_args[cid]["ODG_n_slices"] = args.ODG_n_slices
            client_args[cid]["ODG_sample_steps"] = args.ODG_sample_steps
            client_args[cid]["ODG_mmd_kernel_num"] = args.ODG_mmd_kernel_num
            client_args[cid]["ODG_sample_eps"] = args.ODG_sample_eps
            client_args[cid]["score_model"] = copy.deepcopy(args.score_model)
            client_args[cid]["score_learning_rate"] = args.ODG_score_learning_rate
            client_args[cid]["score_momentum"] = args.ODG_score_momentum
            client_args[cid]["score_weight_decay"] = args.ODG_score_weight_decay
            client_args[cid]["num_classes"] = args.num_classes
            client_args[cid]["ODG_sigma_begin"] = args.ODG_sigma_begin
            client_args[cid]["ODG_sigma_end"] = args.ODG_sigma_end
            client_args[cid]["ODG_anneal_power"] = args.ODG_anneal_power
            client_args[cid]["ODG_sam_rho"] = args.ODG_sam_rho
            client_args[cid]["ODG_sam_eta"] = args.ODG_sam_eta
            client_args[cid]["ODG_sam_adaptive"] = args.ODG_sam_adaptive
        server = ODGServer
        client = ODGClient

    else:
        raise NotImplementedError("method not support")

    return server, client, client_args, server_args


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

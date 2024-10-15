import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../..")
sys.path.append("src/utils/")
import logging
import os

import torch
import wandb

from config import parser
from src.data.data_partition.data_partition import pathological_load_train, dirichlet_load_train, load_PACS_train
from src.models.resnet import ResNet18
from src.models.wideresnet import WideResNet
from src.utils.main_utils import make_save_path, get_server_and_client, set_seed
from src.models.score import Energy, MLPScore


torch.set_num_threads(4)
import warnings
warnings.filterwarnings("ignore")


def run():
    args = parser.parse_args()

    if not args.debug:
        wandb.init()
        for key in dict(wandb.config):
            setattr(args, key, dict(wandb.config)[key])
        wandb.config.update(args)
        wandb.run.name = (
            f"{args.method}"
        )

    set_seed(args.seed)
    # construct save path
    save_path = make_save_path(args)
    if args.checkpoint_path == "default":
        setattr(args, "checkpoint_path", save_path)
    # init logger
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_path, "output.log"),
        format="[%(asctime)s %(levelname)s] %(message)s",
        filemode="w",
    )

    logging.info(f"-------------------- configuration --------------------")
    for key, value in args._get_kwargs():
        logging.info(f"configuration {key}: {value}")

    # ---------- dataset preprocess ----------
    print("prepare dataset...")
    if args.id_dataset == 'PACS':
        train_datasets, num_class = load_PACS_train(args.dataset_path, args.leave_out)
        setattr(args, 'num_client', 3)
    else:
        if args.pathological:
            train_datasets, num_class = pathological_load_train(
                args.dataset_path, args.id_dataset, args.num_client, args.class_per_client, args.dataset_seed
            )
        else:
            train_datasets, num_class = dirichlet_load_train(
                args.dataset_path, args.id_dataset, args.num_client, args.alpha, args.dataset_seed
            )

    # ---------- construct backbone model ----------
    print("init server and clients...")
    if args.backbone == "resnet":
        backbone = ResNet18(num_classes=num_class)
    elif args.backbone == "wideresnet":
        backbone = WideResNet(depth=40, num_classes=num_class, widen_factor=2, dropRate=0.3)
    else:
        raise NotImplementedError("backbone should be ResNet or WideResNet")

    # ---------- construct customized model ----------
    if args.method.lower() == "foogd":
        args.score_model = Energy(net=MLPScore())

    device = torch.device(args.device)

    if args.use_score_model:
        args.score_model = Energy(net=MLPScore())
    # ---------- construct server and clients ----------
    server_args = {
        "join_ratio": args.join_ratio,
        "checkpoint_path": args.checkpoint_path,
        "backbone": backbone,
        "device": device,
        "debug": args.debug,
        "use_score_model": args.use_score_model,
        "score_model": Energy(net=MLPScore()) if args.use_score_model else None,
        "alpha": args.alpha,
        "id_dataset": args.id_dataset,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "num_client": args.num_client,
        "leave_out": args.leave_out,
    }
    client_args = [
        {
            "cid": cid,
            "device": device,
            "epochs": args.local_epochs,
            "backbone": backbone,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "train_id_dataset": train_datasets[cid],
            "use_score_model": args.use_score_model,
            "score_model": Energy(net=MLPScore()) if args.use_score_model else None,
            "checkpoint_path": args.checkpoint_path,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
        }
        for cid in range(args.num_client)
    ]

    Server, Client, client_args, server_args = get_server_and_client(args, client_args, server_args)
    server = Server(server_args)
    clients = [Client(client_args[idx]) for idx in range(args.num_client)]
    server.clients.extend(clients)


    # ---------- fit the model ----------
    logging.info("------------------------------ fit the model ------------------------------")
    for t in range(args.communication_rounds):
        logging.info(f"------------------------- round {t} -------------------------")
        server.fit()

    # ---------- save the model ----------
    print("save the model...")
    server.make_checkpoint(args.communication_rounds)
    print("done.")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        run()
    else:
        sweep_configuration = {
            "method": "grid",
            "parameters": {
                "num_client": {
                    "values": [10]
                },
                "method": {
                    "values": ["FOOGD"]
                },
            },
        }

        sweep_id = wandb.sweep(
            sweep_configuration,
            project=f"{args.id_dataset}_{args.num_client}clients_{args.alpha}_ood",
        )
        wandb.agent(sweep_id, function=run)

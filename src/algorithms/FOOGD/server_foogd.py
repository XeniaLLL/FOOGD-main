import logging
import os
import time

import torch
import wandb

from .client_foogd import ODGClient
from src.algorithms.base.server_base import BaseServer
from src.data.data_partition.data_partition import (
    dirichlet_load_test,
    load_test_ood,
    load_PACS_test,
)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src.utils.accuracy import compute_fnr, compute_auroc
import numpy as np
import torch.nn.functional as F


class ODGServer(BaseServer):
    def __init__(self, server_args):
        super().__init__(server_args)
        data_path = "../datasets/"
        corrupt_list = ["brightness"]
        num_client = server_args["num_client"]
        self.score_model = server_args["score_model"]
        self.ood_dataset = load_test_ood(data_path, "LSUN_C", 21, False)
        if server_args["id_dataset"] == "PACS":
            self.id_datasets, self.cor_datasets, self.num_class = load_PACS_test(
                data_path, server_args["leave_out"],
            )
        else:
            self.id_datasets, self.cor_datasets, self.num_class = dirichlet_load_test(
                data_path, server_args["id_dataset"], num_client, server_args["alpha"], ["brightness"], 21
            )

            self.client_cor_loaders = dict()
            for cor_type in corrupt_list:
                self.client_cor_loaders[cor_type] = [
                    DataLoader(
                        dataset=self.cor_datasets[cid][cor_type],
                        batch_size=128,
                        shuffle=True,
                    )
                    for cid in range(num_client)
                ]
        self.global_round = 0
        self.lambda1 = server_args["lambda1"]
        self.lambda2 = server_args["lambda2"]

    def fit(self):
        client_net_states = []
        client_score_model_states = []
        client_train_time = []
        client_accuracy = []
        client_weights = []
        active_clients = self.select_clients()
        for client in active_clients:
            client_weights.append(len(client.train_id_dataloader))
            client: ODGClient
            start_time = time.time()
            report = client.train()
            client_net_states.append(report["backbone"])
            client_score_model_states.append(report["score_model"])
            client_accuracy.append(report["acc"])
            end_time = time.time()
            logging.info(f"client{client.cid} training time: {end_time - start_time}")
            client_train_time.append(end_time - start_time)

        print(f"average client train time: {sum(client_train_time) / len(client_train_time)}")
        print(
            f"average client accuracy: {sum([client_accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(active_clients))])}"
        )

        global_net_state = self.model_average(client_net_states, client_weights)
        global_score_model_state = self.model_average(client_score_model_states, client_weights)
        for client in self.clients:
            client.backbone.load_state_dict(global_net_state)
            client.score_model.load_state_dict(global_score_model_state)
        self.backbone.load_state_dict(global_net_state)
        self.score_model.load_state_dict(global_score_model_state)

        self.global_round += 1

    def make_checkpoint(self, current_round, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_path, f"model_{current_round}.pt")
        else:
            checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_path)

        checkpoint = {
            "clients": [client.make_checkpoint() for client in self.clients],
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint):
        if "clients" in checkpoint:
            for client_checkpoint, client in zip(checkpoint["clients"], self.clients):
                client.backbone.load_state_dict(client_checkpoint["backbone"])
                client.score_model.load_state_dict(client_checkpoint["score_model"])
        else:
            for client in self.clients:
                client.backbone.load_state_dict(checkpoint[0])
                client.score_model.load_state_dict(checkpoint[1])

    def test_corrupt_accuracy(self, client_cor_loaders):
        # aggregate model
        client_weights = [len(client.train_id_dataloader) for client in self.clients]
        global_net_state = self.model_average([client.backbone.state_dict() for client in self.clients], client_weights)
        for client in self.clients:
            client.backbone.load_state_dict(global_net_state)
        cor_accuracy = {}
        for cor_type, cor_loaders in client_cor_loaders.items():
            cor_accuracy[cor_type] = 0.0
            test_samples = [len(cor_loader) for cor_loader in cor_loaders]
            client_weights = [x / sum(test_samples) for x in test_samples]
            for client, cor_loader, w in zip(self.clients, cor_loaders, client_weights):
                cor_accuracy[cor_type] += client.test_corrupt_accuracy(cor_loader) * w
        return cor_accuracy

    def test(self):
        clients_id_accuracies = []
        clients_fpr95 = []
        clients_auroc = []
        weights = []
        for cid, client in enumerate(self.clients):
            id_loader = DataLoader(
                dataset=self.id_datasets[cid],
                batch_size=128,
                shuffle=True,
            )
            ood_loader = DataLoader(dataset=self.ood_dataset, batch_size=128, shuffle=True)
            weights.append(len(id_loader))

            id_accuracy, fpr95, auroc = self.test_classification_detection_ability_odg(
                client.backbone, client.score_model, id_loader, ood_loader, self.device, "sm"
            )
            clients_id_accuracies.append(id_accuracy)
            clients_fpr95.append(fpr95)
            clients_auroc.append(auroc)

        test_id_accuracy = sum(
            [clients_id_accuracies[cid] * weights[cid] / sum(weights) for cid in range(len(weights))]
        )
        test_fpr95 = sum([clients_fpr95[cid] * weights[cid] / sum(weights) for cid in range(len(weights))])
        test_auroc = sum([clients_auroc[cid] * weights[cid] / sum(weights) for cid in range(len(weights))])
        return test_id_accuracy, test_fpr95, test_auroc

    @staticmethod
    def test_classification_detection_ability_odg(
            backbone, score_model, in_loader, ood_loader, device, score_method="msp"
    ):
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

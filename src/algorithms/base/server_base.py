import copy
import random
from abc import abstractmethod

import torch


class BaseServer:
    def __init__(self, server_args):
        self.clients = []
        self.join_raio = server_args["join_ratio"]
        self.checkpoint_path = server_args["checkpoint_path"]
        self.backbone = copy.deepcopy(server_args["backbone"])
        self.device = server_args["device"]
        self.debug = server_args["debug"]

    def select_clients(self):
        if self.join_raio == 1.0:
            return self.clients
        else:
            return random.sample(self.clients, int(round(len(self.clients) * self.join_raio)))

    @abstractmethod
    def make_checkpoint(self, current_round):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint):
        pass

    @abstractmethod
    def fit(self):
        pass

    def evaluate(self):
        pass

    def test(self):
        pass

    @staticmethod
    def model_average(client_net_states, client_weights):
        state_avg = copy.deepcopy(client_net_states[0])
        client_weights = [w / sum(client_weights) for w in client_weights]

        for k in state_avg.keys():
            state_avg[k] = torch.zeros_like(state_avg[k])
            for i, w in enumerate(client_weights):
                state_avg[k] = state_avg[k] + client_net_states[i][k] * w

        return state_avg

    def test_classification_detection_ability(self, checkpoint, client_id_loaders, ood_loader, score_method="msp"):
        self.load_checkpoint(checkpoint)

        auroc = 0.0
        fpr95 = 0.0
        accuracy = 0.0

        test_samples = [len(id_loader) for id_loader in client_id_loaders]
        client_weights = [x / sum(test_samples) for x in test_samples]

        for client, id_loader, w in zip(self.clients, client_id_loaders, client_weights):
            client_accuracy, client_fpr95, client_auroc = client.test_classification_detection_ability(
                id_loader, ood_loader, score_method=score_method
            )
            accuracy += client_accuracy * w
            fpr95 += client_fpr95 * w
            auroc += client_auroc * w

        return accuracy, fpr95, auroc

    def test_corrupt_accuracy(self, client_cor_loaders):
        cor_accuracy = {}
        for cor_type, cor_loaders in client_cor_loaders.items():
            cor_accuracy[cor_type] = 0.0
            test_samples = [len(cor_loader) for cor_loader in cor_loaders]
            client_weights = [x / sum(test_samples) for x in test_samples]

            for client, cor_loader, w in zip(self.clients, cor_loaders, client_weights):
                cor_accuracy[cor_type] += client.test_corrupt_accuracy(cor_loader) * w

        return cor_accuracy

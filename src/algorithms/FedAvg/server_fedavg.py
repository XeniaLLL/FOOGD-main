import logging
import os
import time

import torch
import wandb

from src.algorithms.FedAvg.client_fedavg import FedAvgClient
from src.algorithms.base.server_base import BaseServer


class FedAvgServer(BaseServer):
    def __init__(self, server_args):
        super().__init__(server_args)

    def fit(self):
        client_net_states = []
        client_train_time = []
        client_accuracy = []
        client_weights = []
        active_clients = self.select_clients()
        for client in active_clients:
            client_weights.append(len(client.train_id_dataloader))
            client: FedAvgClient
            start_time = time.time()
            report = client.train()
            end_time = time.time()
            client_net_states.append(report['backbone'])
            client_accuracy.append(report['acc'])
            logging.info(f"client{client.cid} training time: {end_time - start_time}")
            client_train_time.append(end_time - start_time)

        print(f'average client train time: {sum(client_train_time) / len(client_train_time)}')
        print(
            f'average client accuracy: {sum([client_accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(active_clients))])}'
        )

        if not self.debug:
            wandb.log({
                'accuracy': sum(
                    [client_accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(active_clients))])
            })

        global_net_state = self.model_average(client_net_states, client_weights)
        for client in self.clients:
            client.backbone.load_state_dict(global_net_state)
        self.backbone.load_state_dict(global_net_state)

    def make_checkpoint(self, current_round):
        torch.save(self.backbone.state_dict(), os.path.join(self.checkpoint_path, f'model_{current_round}.pt'))

    def load_checkpoint(self, checkpoint):
        pass

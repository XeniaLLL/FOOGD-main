from abc import ABC

import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F

from src.algorithms.base.client_base import BaseClient


class FedAvgClient(BaseClient, ABC):
    def __init__(self, client_args):
        super().__init__(client_args)

        self.optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def train(self):
        self.backbone.to(self.device)

        accuracy = []
        print(f"---------- training client {self.cid} ----------")
        for epoch in range(self.epochs):
            print(f"---------- epoch {epoch}  ----------")
            self.backbone.train()
            for classifier_set in self.train_id_dataloader:
                if len(classifier_set[0]) == 1:
                    continue
                data = classifier_set[0].to(self.device)
                targets = classifier_set[1].to(self.device)
                logit = self.backbone(data)

                pred = logit.data.max(1)[1]
                accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))

                self.optimizer.zero_grad()
                loss = F.cross_entropy(logit, targets)
                loss.backward()
                self.optimizer.step()

        self.backbone.cpu()
        return {'backbone': self.backbone.state_dict(), 'acc': sum(accuracy) / len(accuracy)}

    def load_checkpoint(self, checkpoint):
        pass

    def make_checkpoint(self):
        pass

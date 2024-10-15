import copy
from abc import abstractmethod

from torch.utils.data import DataLoader


class BaseClient:

    def __init__(self, client_args):
        # ------ basic configuration ------
        self.cid = client_args['cid']
        self.device = client_args['device']
        self.epochs = client_args['epochs']
        self.backbone = copy.deepcopy(client_args['backbone'])
        self.learning_rate = client_args["learning_rate"]
        self.momentum = client_args["momentum"]
        self.weight_decay = client_args["weight_decay"]
        # ------ refer to generate and iterate dataloader ------
        self.batch_size = client_args['batch_size']
        self.num_workers = client_args['num_workers']
        self.pin_memory = client_args['pin_memory']
        # ------ refer to  dataloader generating ------
        self.train_id_dataset = client_args['train_id_dataset']
        self.train_id_dataloader = DataLoader(
            dataset=self.train_id_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
        )

    @abstractmethod
    def make_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint):
        pass

    @abstractmethod
    def train(self):
        pass

    def evaluate(self):
        pass

    def test(self):
        pass

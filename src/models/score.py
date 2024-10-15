import torch
from torch import nn
import os


class LatentModel(nn.Module):
    def __init__(self):
        super(LatentModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)


class MLPScore(nn.Module):
    def __init__(self):
        super().__init__()
        # self.main = nn.Sequential(
        #     nn.Linear(128, 1024),
        #     nn.LayerNorm(1024),
        #     nn.ELU(),
        #     nn.Linear(1024, 1024),
        #     nn.LayerNorm(1024),
        #     nn.ELU(),
        #     nn.Linear(1024, 512),
        #     nn.LayerNorm(512),
        #     nn.ELU(),
        #     nn.Linear(512, 128),
        #     nn.LayerNorm(128)
        # )

        # self.main = nn.Sequential(  # note good for cifar10
        #     nn.Linear(128, 512),
        #     # # nn.LayerNorm(256),
        #     # # nn.ELU(),
        #     # # nn.Linear(256, 256),
        #     # nn.LayerNorm(512),
        #     # nn.ELU(),
        #     # nn.Linear(512, 256),
        #     nn.LayerNorm(512),
        #     nn.ELU(),
        #     # nn.Linear(512, 1024),
        #     # nn.LayerNorm(1024),
        #     nn.Linear(512, 128)
        # )

        self.main = nn.Sequential( # note work for cifar100
            nn.Linear(128, 1024),
            # # nn.LayerNorm(256),
            # # nn.ELU(),
            # # nn.Linear(256, 256),
            # nn.LayerNorm(1024),
            # nn.ELU(),
            # nn.Linear(512, 512),
            nn.LayerNorm(1024),
            nn.ELU(),
            # nn.Linear(512, 1024),
            # nn.LayerNorm(1024),
            nn.Linear(1024, 128)
        )
        # self.ln_last=nn.LayerNorm(128)

        # self.main = nn.Sequential(
        #     nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
        #     nn.BatchNorm1d(128),
        #     nn.ELU(),
        #     nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
        # )

    def forward(self, x):
        # return self.ln_last(self.main(x))
        return self.main(x)


# --- energy model ---
class Energy(nn.Module):
    def __init__(self, net):
        """A simple energy model

        Args:
            net (nn.Module): An energy function, the output shape of
                the energy function should be (b, 1). The score is
                computed by grad(-E(x))
        """
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logp = -self.net(x).sum()  # E(x)= -log_\theta p(x)
        return torch.autograd.grad(logp, x, create_graph=True)[0]  # score = \nubla_x  -log_\theta p(x)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self

from abc import ABC

import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F

from src.algorithms.base.client_base import BaseClient
from src.utils.accuracy import compute_fnr, compute_auroc
from .functional import langevin_dynamics, anneal_langevin_dynamics
from .ksd import *
from .loss_odg import MMDLoss


class ODGClient(BaseClient, ABC):
    def __init__(self, client_args):
        super().__init__(client_args)

        base_optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=self.learning_rate,

        )
        self.optimizer = base_optimizer
        self.noise_type = client_args.get("ODG_noise_type", "gaussian")
        self.loss_type = client_args.get("ODG_loss_types", "dsm")
        self.score_model = client_args["score_model"]
        ODG_score_learning_rate = client_args["score_learning_rate"]

        self.score_optimizer = torch.optim.SGD(
            params=self.score_model.parameters(),
            lr=ODG_score_learning_rate,
        )
        self.sample = langevin_dynamics if client_args.get("ODG_sampler", None) == "ld" else anneal_langevin_dynamics
        self.sample_steps = client_args.get("ODG_sample_steps", 0)
        self.sample_eps = client_args.get("ODG_sample_eps", 0.0)
        self.mmd_loss = MMDLoss(kernel_num=client_args["ODG_mmd_kernel_num"])
        self.n_slices = client_args.get("ODG_n_slices", 0)
        self.svgd_sampler = SVGD()

        self.sigmas = (
            torch.tensor(
                np.exp(
                    np.linspace(
                        np.log(client_args.get("ODG_sigma_begin", 0.0001)),
                        np.log(client_args.get("ODG_sigma_end", 0.5)),
                        client_args["num_classes"],
                    )
                )
            )
            .float()
            .to(self.device)
        )
        self.anneal_power = client_args.get("ODG_anneal_power", 0)
        self.lambda1 = client_args["lambda1"]
        self.lambda2 = client_args["lambda2"]

    def dsm_loss2(self, x, v, sigma=0.1):
        """DSM loss from
        A Connection Between Score Matching
            and Denoising Autoencoders

        The loss is computed as
        x_ = x + v   # noisy samples
        s = -dE(x_)/dx_
        loss = 1/2*||s + (x-x_)/sigma^2||^2

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises
            sigma (float, optional): noise scale. Defaults to 0.1.

        Returns:
            DSM loss
        """
        x = x.requires_grad_()
        # v = v
        x_ = x + v * sigma  # note: perturbed sample
        s = self.score_model(x_)  # note: obtain score -\nabla_x s_model(x_)
        loss = torch.norm(s * (sigma ** 2) + v, dim=-1) ** 2  # note: matching loss
        loss = loss.mean() / 2.0
        return loss

    def dsm_loss(self, x, v, sigma=0.1):
        x = x.requires_grad_()

        x_ = x + v * sigma  # note: perturbed sample
        s = self.score_model(x_)  # note: obtain score -\nabla_x s_model(x_)
        loss = F.l1_loss(s, -v)  # note: ref from loss = coeff_dsm*F.l1_loss(self.σψ(x + self.σ * z), -z)
        loss = loss.mean()
        return loss

    # z ->x_0 ->x_1 -> x_2  -> X^T^ x^T

    def anneal_dsm_loss(self, x, v, labels):
        """anneal noise scales DSM loss from
        A Connection Between Score Matching
            and Denoising Autoencoders

        The loss is computed as
        x_ = x + v   # noisy samples
        s = -dE(x_)/dx_
        loss = 1/2*||s + (x-x_)/sigma^2||^2

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor): sampled noises
            sigma (float, optional): noise scale. Defaults to 0.1.

        Returns:
            DSM loss
        """
        used_sigma = self.sigmas[labels].view(x.shape[0], *([1] * len(x.shape[1:])))
        x = x.requires_grad_()
        # v = v
        x_ = x + v * used_sigma  # note: perturbed sample
        s = self.score_model(x_)  # note: obtain score -\nabla_x s_model(x_)]
        t = v / (used_sigma ** 2)
        loss = (
                torch.norm(s.view(s.shape[0], -1) + v.view(t.shape[0], -1), dim=-1) ** 2
                * used_sigma.squeeze() ** self.anneal_power
        )  # note: matching loss
        loss = loss.mean() / 2.0
        return loss

    def get_random_noise(self, x, n_slices=None):
        """Sampling random noises

        Args:
            x (torch.Tensor): input samples
            n_slices (int, optional): number of slices. Defaults to None.

        Returns:
            torch.Tensor: sampled noises
        """
        if n_slices is None:
            v = torch.randn_like(x, device=self.device)
        else:
            v = torch.randn((n_slices,) + x.shape, dtype=x.dtype, device=self.device)
            v = v.view(-1, *v.shape[2:])  # (n_slices*b, 2)

        if self.noise_type == "radermacher":
            v = v.sign()
        elif self.noise_type == "sphere":
            v = v / torch.norm(v, dim=-1, keepdim=True) * np.sqrt(v.shape[-1])
        elif self.noise_type == "gaussian":
            pass
        else:
            raise NotImplementedError(f"Noise type '{self.noise_type}' not implemented.")
        return v

    def get_score_matching_loss(self, x, v=None):
        """Compute loss

        Args:
            x (torch.Tensor): input samples
            v (torch.Tensor, optional): sampled noises. Defaults to None.

        Returns:
            loss
        """
        if self.loss_type == "dsm":
            v = self.get_random_noise(x, None)
            loss = self.dsm_loss2(x, v)

        elif self.loss_type == "anneal_dsm":

            v = self.get_random_noise(x, None)
            labels = torch.randint(0, len(self.sigmas), (x.shape[0],), device=x.device)

            loss = self.anneal_dsm_loss(x, v, labels)
        else:
            raise NotImplementedError(f"Loss type '{self.loss_type}' not implemented.")

        return loss

    def train(self):
        self.backbone.to(self.device)
        self.score_model.to(self.device)

        accuracy = []
        print(f"---------- training client {self.cid} ----------")
        for epoch in range(self.epochs):
            print(f"---------- epoch {epoch}  ----------")
            self.backbone.train()
            for classifier_set in self.train_id_dataloader:
                self.optimizer.zero_grad()
                if len(classifier_set[0]) == 1:
                    continue
                if isinstance(classifier_set[0], list):
                    data = torch.cat(classifier_set[0], dim=0).to(self.device)
                    targets = torch.cat(classifier_set[1], dim=0).to(self.device)
                else:
                    data = classifier_set[0].to(self.device)
                    targets = classifier_set[1].to(self.device)

                latents = self.backbone.intermediate_forward(data)
                logits = self.backbone.fc(latents)
                loss3 = F.cross_entropy(logits, targets)

                split_idx = latents.shape[0] // 2
                latents_ori, latents_aug = torch.split(latents, split_idx, dim=0)

                if split_idx > 1:
                    median_dist = median_heruistic(latents_ori, latents_aug)
                    bandwidth = 2 * np.sqrt(1. / (2 * np.log(split_idx + 1))) * torch.pow(0.5 * median_dist, 0.5)
                    loss_stein = compute_KSD(latents_ori, latents_aug, self.score_model,
                                             kernel=SE_kernel_multi, trace_kernel=trace_SE_kernel_multi,
                                             bandwidth=bandwidth, flag_U=False, flag_retain=True, flag_create=False)

                    loss3 += self.lambda1 * loss_stein

                loss3.backward()

                self.optimizer.step()
                self.score_model.to(self.device)

                latents_ori = latents_ori.data
                latents_ori = latents_ori.requires_grad_()
                self.score_optimizer.zero_grad()
                loss2 = self.get_score_matching_loss(latents_ori)
                noise = self.get_random_noise(latents_ori)
                latents_gen = self.sample(
                    score_fn=self.score_model, x=noise, eps=self.sample_eps, n_steps=self.sample_steps,
                )
                loss1 = self.lambda2 * self.mmd_loss(latents_ori, latents_gen)
                loss2 += loss1
                loss2.backward()
                self.score_optimizer.step()
                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))


        self.backbone.cpu()
        self.score_model.cpu()
        return {
            "backbone": self.backbone.state_dict(),
            "acc": sum(accuracy) / len(accuracy),
            "score_model": self.score_model.state_dict(),
        }

    def load_checkpoint(self, checkpoint):
        pass

    def make_checkpoint(self):
        checkpoint = {
            "backbone": self.backbone.state_dict(),
            "score_model": self.score_model.state_dict()
        }
        return checkpoint

    @torch.no_grad()
    def test_corrupt_accuracy(self, cor_loader):
        self.backbone.to(self.device)
        self.backbone.eval()

        accuracy = []
        for data, targets in cor_loader:
            if len(data) == 1:
                continue
            data, targets = data.to(self.device), targets.to(self.device)
            logit = self.backbone(data)
            pred = logit.data.max(1)[1]
            accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))
        return sum(accuracy) / len(accuracy)

    def test_classification_detection_ability(self, id_loader, ood_loader, score_method="sm"):
        self.backbone.to(self.device)
        self.score_model.to(self.device)
        self.backbone.eval()
        self.score_model.eval()

        ood_score_id = []
        ood_score_ood = []
        accuracy = []

        with torch.no_grad():
            for data, target in id_loader:
                data, target = data.to(self.device), target.to(self.device)
                latents = self.backbone.intermediate_forward(data)
                logit = self.backbone(data)
                scores = self.score_model(latents).norm(dim=-1)
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
                data = data.to(self.device)
                latents = self.backbone.intermediate_forward(data)
                logit = self.backbone(data)
                scores = self.score_model(latents).norm(dim=-1)
                if score_method == "energy":
                    ood_score_ood.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
                elif score_method == "msp":
                    ood_score_ood.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))
                elif score_method == "sm":
                    ood_score_ood.extend(list(scores.data.cpu().numpy()))

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
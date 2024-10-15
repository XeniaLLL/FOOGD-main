"""
this file contains parameters describing the setting (e.g. dataset, backbone model) of the experiment.
variable `parser` is used in `/main.py`.
"""
import argparse

parser = argparse.ArgumentParser(description='arguments for OOD generalization and detection training')

parser.add_argument('--wandb_mode', type=str, choices=['disabled','online', 'offline'], default='online', help='Wandb log mode')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--use_profile', type=bool, default=False)
parser.add_argument('--seed', type=int)
parser.add_argument('--method', type=str, default='FOOGD')
# ---------- dataset partition ----------
parser.add_argument('--id_dataset', type=str, default='cifar10_fourier_aug', help='the ID dataset')
parser.add_argument('--leave_out', type=str, default='art_painting')
parser.add_argument('--dataset_path', type=str, default='', help='path to dataset')
parser.add_argument('--alpha', type=float, default=0.1, help='parameter of dirichlet distribution')
parser.add_argument('--num_client', type=int, default=10, help='number of clients')
parser.add_argument('--dataset_seed', type=int, default=21, help='seed to split dataset')
parser.add_argument('--pathological', type=bool, default=False, help='using pathological method split dataset')
parser.add_argument('--class_per_client', type=int, default=2, help='classes per client')
parser.add_argument('--num_classes', type=int, default=100, help='number of dataset classes')
# ---------- backbone ----------
parser.add_argument('--backbone', type=str, choices=['resnet', 'wideresnet'], default='wideresnet', help='backbone model of task')
# ---------- device ----------
parser.add_argument('--device', type=str, default='cuda:1', help='device')
# ---------- server configuration ----------
parser.add_argument('--join_ratio', type=float, default=1., help='join ratio')
parser.add_argument('--communication_rounds', type=int, default=100, help='total communication round')
parser.add_argument('--checkpoint_path', type=str, default='default', help='check point path')
# ---------- client configuration ----------
parser.add_argument('--local_epochs', type=int, default=5, help='local epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='dataloader\'s num_workers')
parser.add_argument('--pin_memory', type=bool, default=False, help='dataloader\'s pin_memory')
# ---------- optimizer --------
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)

parser.add_argument("--use_score_model", type=bool, default=True)


parser.add_argument('--ODG_noise_type', type=str, choices=['gaussian', 'radermacher','sphere'], default='gaussian', help='score model noise type')
parser.add_argument('--ODG_loss_types', type=str, choices=['ssm-vr', 'ssm', 'dsm','deen', 'anneal_dsm'], default='anneal_dsm', help='score model loss type')
parser.add_argument('--ODG_sampler', type=str, choices=['ld', 'ald'], default='ld', help='score model sample type')
parser.add_argument('--ODG_sample_steps', type=int, default=10, help='Langiven sample steps')
parser.add_argument('--ODG_n_slices', type=int, default=12, help='special for sliced score matching')
parser.add_argument('--ODG_mmd_kernel_num', type=int, default=2, help='number of MMD loss')
parser.add_argument('--ODG_sample_eps', type=float, default=0.01, help='Langiven sample epsilon size')
parser.add_argument('--ODG_score_learning_rate', type=float, default=0.01)
parser.add_argument('--ODG_score_momentum', type=float, default=0.)
parser.add_argument('--ODG_score_weight_decay', type=float, default=0.)
parser.add_argument('--ODG_sigma_begin', type=float, default=0.01)
parser.add_argument('--ODG_sigma_end', type=float, default=1)
parser.add_argument('--ODG_anneal_power', type=float, default=2)
parser.add_argument("--ODG_sam_rho",  type=float, default=0.5, help="hyper-param for sam& asam ")
parser.add_argument("--ODG_sam_eta", type=float, default=0.2, help="hyper-param for asam ")
parser.add_argument("--ODG_sam_adaptive", type=bool, default=True, help="hyper-param for asam ")
# ---------- ODG configuration end ----------
parser.add_argument("--lambda1", type=float, default=0, help="ksd")
parser.add_argument("--lambda2", type=float, default=0, help="mmd")

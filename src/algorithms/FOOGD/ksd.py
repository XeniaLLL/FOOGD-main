##### Source: Sliced Kernelized Stein Discrepancy ######

import torch
import math

# from jax.scipy.optimize import minimize

def trace_SE_kernel_multi(sample1,sample2, bandwidth, K):
    # compute trace of second-order derivative of RBF kernel
    '''
    Compute the trace of 2 order gradient of K
    :param sample1: x : N x N x dim
    :param sample2: y : N x N x dim
    :param kwargs: kernel hyper: bandwidth
    :return:
    '''
    diff=sample1-sample2 # N x N x dim
    H=K*(2./(bandwidth**2+1e-9)*sample1.shape[-1]-4./(bandwidth**4+1e-9)*torch.sum(diff*diff,dim=-1)) # N x N
    return H


def SE_kernel(sample1, sample2, bandwidth):
    # compute RBF kernel with 1 dimensional input
    '''
    Compute the square exponential kernel
    :param sample1: x
    :param sample2: y
    :param kwargs: kernel hyper-parameter: bandwidth
    :return:
    '''
    if len(bandwidth.shape) != 1:  # 'bandwidth_array' in kwargs['kernel_hyper']:
        # bandwidth could be an array because each g_r has a unique corresponding median heuristic bandwdith.
        # bandwidth=kwargs['kernel_hyper']['bandwidth_array'] # g or * x g
        if len(sample1.shape) == 4:
            if len(bandwidth.shape) == 1:
                bandwidth_exp = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(bandwidth, dim=-1), dim=-1),
                                                dim=-1)  # g x 1 x 1 x 1
            else:
                bandwidth_exp = torch.unsqueeze(torch.unsqueeze(bandwidth, dim=-1), dim=-1)  # * x g x 1 x 1

        else:
            bandwidth_exp = torch.unsqueeze(torch.unsqueeze(bandwidth, dim=-1), dim=-1)  # g x 1 x 1
        K = torch.exp(-(sample1 - sample2) ** 2 / (
                bandwidth_exp ** 2 + 1e-9))  # g x sam1 x sam2 or g x 1 x sam1 x sam2 or * x g x sam1 x sam2
    else:
        K = torch.exp(-(sample1 - sample2) ** 2 / (bandwidth ** 2 + 1e-9))
    return K


def SE_kernel_multi(sample1, sample2, bandwidth):
    '''
    Compute the multidim square exponential kernel
    :param sample1: x : N x N x dim
    :param sample2: y : N x N x dim
    :param kwargs: kernel hyper-parameter:bandwidth
    :return:
    '''
    if len(sample1.shape) == 4:  # * x N x d
        bandwidth = bandwidth.unsqueeze(-1).unsqueeze(-1)

    sample_diff = sample1 - sample2  # N x N x dim

    norm_sample = torch.norm(sample_diff, dim=-1) ** 2  # N x N or * x N x N

    K = torch.exp(-norm_sample / (bandwidth ** 2 + 1e-9))
    return K


###############note compute for score #####################
'''
def ICA_un_log_likelihood(x,base_dist,W_inv):
    x_exp = x.unsqueeze(-1)  # num x D x 1
    W_inv_exp = W_inv.unsqueeze(0)  # num x D x D
    z = torch.matmul(W_inv_exp, x_exp).squeeze()  # num x D
    log_likelihood = base_dist.log_prob(z)  # num
    return log_likelihood
 
samples1 = x.clone().detach().requires_grad_()
samples2 = samples1.clone().detach().requires_grad_()
un_log_likelihood1 = ICA_un_log_likelihood(samples1, base_dist, W_inv)
score1 = torch.autograd.grad(un_log_likelihood1.sum(), samples1, create_graph=True)[0]
un_log_likelihood2 = ICA_un_log_likelihood(samples2, base_dist, W_inv)
score2 = torch.autograd.grad(un_log_likelihood2.sum(), samples2, create_graph=True)[0]   

'''

def median_heruistic(sample1,sample2):
    with torch.no_grad():
        G=torch.sum(sample1*sample1,dim=-1)# N or * x N note elementwise multiplication
        G_exp=torch.unsqueeze(G,dim=-2) # 1 x N or * x1 x N note

        H=torch.sum(sample2*sample2,dim=-1)
        H_exp=torch.unsqueeze(H,dim=-1) # N x 1 or * * x N x 1
        dist=G_exp+H_exp-2*sample2.matmul(torch.transpose(sample1,-1,-2)) #N x N or  * x N x N
        if len(dist.shape)==3:
            dist=dist[torch.triu(torch.ones(dist.shape))==1].view(dist.shape[0],-1)# * x (NN)
            median_dist,_=torch.median(dist,dim=-1) # *
        else:
            dist=(dist-torch.tril(dist)).view(-1)
            median_dist=torch.median(dist[dist>0.])
    return median_dist.clone().detach()




def compute_KSD(samples1, samples2, score_func, kernel, trace_kernel, bandwidth, score_sample1=None, score_sample2=None,
                flag_U=False, flag_retain=False, flag_create=False):
    # Compute KSD

    # note: score_model is modeling for \nubla_x log p_\theta(x)

    divergence_accum = 0

    samples1_crop_exp = torch.unsqueeze(samples1, dim=1).repeat(1, samples2.shape[0], 1)  # N x N(rep) x dim
    samples2_crop_exp = torch.unsqueeze(samples2, dim=0).repeat(samples1.shape[0], 1, 1)  # N(rep) x N x dim


    # Compute Term1
    if (score_sample1 is None) or (score_sample2 is None):
        score_sample1 = score_func(samples1)  # N
        # score_sample1_exp = torch.unsqueeze(score_sample1.view(samples1.size(0), -1), dim=1)

        # score_sample1 = torch.autograd.grad(torch.sum(score_sample1), samples1)[0]  # N x dim # note cancel grad
        score_sample2 = score_func(samples2)  # N
    #     score_sample2_exp = torch.unsqueeze( score_sample2.view(samples1.size(0), -1), dim=0)
    # samples1_crop_exp =  torch.unsqueeze(samples1.view(samples1.size(0), -1), dim=1).repeat(1, samples2.shape[0], 1)  # N x N(rep) x dim
    # samples2_crop_exp =  torch.unsqueeze(samples2.view(samples1.size(0), -1), dim=0).repeat(samples1.shape[0], 1, 1)  # N(rep) x N x dim
    # score_sample2 = torch.autograd.grad(torch.sum(score_sample2), samples2)[0]  # N x dim # note cancel grad


    score_sample1_exp = torch.unsqueeze(score_sample1, dim=1)  # N x 1 x dim
    score_sample2_exp = torch.unsqueeze(score_sample2, dim=0)  # 1 x N x dim

    K = kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth)

    if flag_U:
        Term1 = (K - torch.diag(torch.diag(K))) * torch.sum(score_sample1_exp * score_sample2_exp, dim=-1)  # N x N
    else:
        Term1 = (K) * torch.sum(score_sample1_exp * score_sample2_exp, dim=-1)  # N x N

    # Compute Term 2, directly use autograd for kernel gradient
    if flag_U:
        grad_K_2 = \
            torch.autograd.grad(torch.sum((K - torch.diag(torch.diag(K)))), samples2_crop_exp, retain_graph=flag_retain,
                                create_graph=flag_create)[0]  # N x N x dim
    else:
        grad_K_2 = \
            torch.autograd.grad(torch.sum((K)), samples2_crop_exp, retain_graph=flag_retain, create_graph=flag_create)[
                0]  # N x N x dim
    Term2 = torch.sum(score_sample1_exp * grad_K_2, dim=-1)  # N x N

    # Compute Term 3
    if flag_U:
        K = kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth)
        grad_K_1 = \
            torch.autograd.grad(torch.sum((K - torch.diag(torch.diag(K)))), samples1_crop_exp, retain_graph=flag_retain,
                                create_graph=flag_create)[0]  # N x N x dim

    else:
        K = kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth)

        grad_K_1 = \
            torch.autograd.grad(torch.sum((K)), samples1_crop_exp, retain_graph=flag_retain, create_graph=flag_create)[
                0]  # N x N x dim

    Term3 = torch.sum(score_sample2_exp * grad_K_1, dim=-1)  # N x N

    # Compute Term 4, manually derive the trace of high-order derivative of kernel called trace_kernel
    K = kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth)

    if flag_U:
        T_K = trace_kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth, K=K - torch.diag(torch.diag(K)))

        grad_K_12 = T_K  # N x N
    else:
        T_K = trace_kernel(samples1_crop_exp, samples2_crop_exp, bandwidth=bandwidth, K=K)

        grad_K_12 = T_K  # N x N

    Term4 = grad_K_12

    KSD_comp = torch.sum(Term1 + 1 * Term2 + 1 * Term3 + 1 * Term4)

    divergence_accum += KSD_comp

    if flag_U:
        KSD = divergence_accum / ((samples1.shape[0] - 1) * samples2.shape[0])

    else:
        KSD = divergence_accum / (samples1.shape[0] * samples2.shape[0])

    return KSD#, Term1 + Term2 + Term3 + Term4


def compute_MMD(samples1, samples2, kernel, bandwidth, flag_U=True, flag_simple_U=True):
    # samples1: N x dim
    # samples2: N x dim
    n = samples1.shape[0]
    m = samples2.shape[0]

    if m != n and flag_simple_U:
        raise ValueError('If m is not equal to n, flag_simple_U must be False')

    samples1_exp1 = torch.unsqueeze(samples1, dim=1)  # N x 1 x dim
    samples1_exp2 = torch.unsqueeze(samples1, dim=0)  # 1 x N x dim

    samples2_exp1 = torch.unsqueeze(samples2, dim=1)  # N x 1 x dim
    samples2_exp2 = torch.unsqueeze(samples2, dim=0)  # 1 x N x dim

    # Term1
    K1 = kernel(samples1_exp1, samples1_exp2, bandwidth=bandwidth)  # N x N
    if flag_U:
        K1 = K1 - torch.diag(torch.diag(K1))
    # Term3
    K3 = kernel(samples2_exp1, samples2_exp2, bandwidth=bandwidth)  # N x N
    if flag_U:
        K3 = K3 - torch.diag(torch.diag(K3))

    # Term2
    if flag_simple_U:
        K2_comp = kernel(samples1_exp1, samples2_exp2, bandwidth=bandwidth)
        K2_comp = K2_comp - torch.diag(torch.diag(K2_comp))
        K2 = K2_comp + K2_comp.t()
    else:
        K2 = 2 * kernel(samples1_exp1, samples2_exp2, bandwidth=bandwidth)  # N x N

    if flag_U:
        if flag_simple_U:
            MMD = torch.sum(K1) / (n * (n - 1)) + torch.sum(K3) / (m * (m - 1)) - 1. / (m * (m - 1)) * torch.sum(K2)

        else:
            MMD = torch.sum(K1) / (n * (n - 1)) + torch.sum(K3) / (m * (m - 1)) - 1. / (m * n) * torch.sum(K2)
    else:
        MMD = torch.sum(K1 + K3 - K2) / (m * n)

    return MMD, K1 + K3 - K2


def compute_SD(samples1, test_func, score_func, m, lam=0.1, score_sample1=None):
    # Compute Stein Discrepancy using neural net
    test_out = test_func.forward(samples1)  # N x D
    samples1_dup = samples1.unsqueeze(-2).repeat(1, m, 1)  # N x M X D
    test_out_dup = test_func.forward(samples1_dup)  # N x  M X D

    if score_sample1 is None:
        score = score_func(samples1)
        score = torch.autograd.grad(score.sum(), samples1)[0]  # N x D
    Term1 = torch.sum(score * test_out, dim=-1)  # N

    # compute term 2 with Hutchson's trick
    eps = torch.randn(1, m, samples1.shape[-1])  # 1 x M x D
    f_eps = torch.sum(test_out_dup * eps, dim=-1)  # N x M
    f_eps_grad = torch.autograd.grad(f_eps.sum(), samples1_dup, create_graph=True, retain_graph=True)[0]  # N x M x D
    eps_f_eps = torch.sum(eps * f_eps_grad, dim=-1).mean(-1)  # N

    Term2 = eps_f_eps

    divergence = torch.mean(Term1 + Term2)  # 1

    # regularize
    reg = torch.sum(test_out * test_out + 1e-6, dim=-1).mean()
    divergence_reg = divergence - lam * reg
    return divergence, divergence_reg


############# Source ksddescent project #################


def linear_stein_kernel(x, y, score_x, score_y, return_kernel=False):
    """Compute the linear Stein kernel between x and y


    Parameters
    ----------
    x : torch.tensor, shape (n, p)
        Input particles
    y : torch.tensor, shape (n, p)
        Input particles
    score_x : torch.tensor, shape (n, p)
        The score of x
    score_y : torch.tensor, shape (n, p)
        The score of y
    return_kernel : bool
        whether the original kernel k(xi, yj) should also be returned

    Return
    ------
    stein_kernel : torch.tensor, shape (n, n)
        The linear Stein kernel
    kernel : torch.tensor, shape (n, n)
        The base kernel, only returned id return_kernel is True
    """
    n, d = x.shape
    kernel = x @ y.t()
    stein_kernel = (
            score_x @ score_y.t() * kernel + score_x @ x.t() + score_y @ y.t() + d
    )
    if return_kernel:
        return stein_kernel, kernel
    return stein_kernel


def gaussian_stein_kernel(
        x, y, scores_x, scores_y, sigma, return_kernel=False
):
    """Compute the Gaussian Stein kernel between x and y


    Parameters
    ----------
    x : torch.tensor, shape (n, p)
        Input particles
    y : torch.tensor, shape (n, p)
        Input particles
    score_x : torch.tensor, shape (n, p)
        The score of x
    score_y : torch.tensor, shape (n, p)
        The score of y
    sigma : float
        Bandwidth
    return_kernel : bool
        whether the original kernel k(xi, yj) should also be returned

    Return
    ------
    stein_kernel : torch.tensor, shape (n, n)
        The linear Stein kernel
    kernel : torch.tensor, shape (n, n)
        The base kernel, only returned id return_kernel is True
    """
    _, p = x.shape
    d = x[:, None, :] - y[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    k = torch.exp(-dists / sigma / 2)
    scalars = scores_x.mm(scores_y.T)
    scores_diffs = scores_x[:, None, :] - scores_y[None, :, :]
    diffs = (d * scores_diffs).sum(axis=-1)
    der2 = p - dists / sigma
    stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
    if return_kernel:
        return stein_kernel, k
    return stein_kernel


def gaussian_stein_kernel_single(x, score_x, return_kernel=False):
    """Compute the Gaussian Stein kernel between x and x


    Parameters
    ----------
    x : torch.tensor, shape (n, p)
        Input particles
    score_x : torch.tensor, shape (n, p)
        The score of x
    sigma : float
        Bandwidth
    return_kernel : bool
        whether the original kernel k(xi, xj) should also be returned

    Return
    ------
    stein_kernel : torch.tensor, shape (n, n)
        The linear Stein kernel
    kernel : torch.tensor, shape (n, n)
        The base kernel, only returned id return_kernel is True
    """
    x_flatten = torch.squeeze(x.view(x.size(0), -1), dim=1)
    score_x_faltten = torch.squeeze(score_x.view(score_x.size(0), -1), dim=1)
    def median(tensor):
        """
        torch.median() acts differently from np.median(). We want to simulate numpy implementation.
        """
        tensor = tensor.detach().flatten()
        tensor_max = tensor.max()[None]
        return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.
    n, p = x_flatten.shape
    # Gaussian kernel:
    norms = (x_flatten ** 2).sum(-1)
    sigma = median(norms)/math.log(n)
    dists = -2 * x_flatten @ x_flatten.t() + norms[:, None] + norms[None, :]
    k = (-dists / 2 / (sigma+ 1e-6)).exp()

    # Dot products:
    diffs = (x_flatten * score_x_faltten).sum(-1, keepdim=True) - (x_flatten @ score_x_faltten.t())
    diffs = diffs + diffs.t()
    scalars = score_x_faltten.mm(score_x_faltten.t())
    der2 = p - dists / sigma
    stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
    if return_kernel:
        return stein_kernel, k
    return stein_kernel


# Author: Pierre Ablin <pierre.ablin@ens.fr>
#
# License: MIT
import torch
import numpy as np


from scipy.optimize import fmin_l_bfgs_b
from time import time


def ksdd_gradient(x0, score, step, max_iter=1000, bw=1,
                  store=False, verbose=False, clamp=None, beta=0.2):
    '''Kernel Stein Discrepancy descent with gradient descent

    Perform Kernel Stein Discrepancy descent with gradient descent.
    Since it uses gradient descent, a step size must be specified.

    Parameters
    ----------
    x0 : torch.tensor, size n_samples x n_features
        initial positions

    score : callable
        function that computes the score

    step : float
        step size

    max_iter : int
        max numer of iters

    bw : float
        bandwidth of the stein kernel

    stores : None or list of ints
        whether to stores the iterates at the indexes in the list

    verbose: bool
        wether to print the current loss

    clamp:
        if not None, should be a tuple (a, b). The points x are then
        constrained to stay in [a, b]

    Returns
    -------
    x: torch.tensor, size n_samples x n_features
        The final positions

    loss_list : list of floats
        List of the loss values during iterations

    References
    ----------
    A.Korba, P-C. Aubin-Frankowski, S.Majewski, P.Ablin.
    Kernel Stein Discrepancy Descent
    International Conference on Machine Learning, 2021.
    '''
    x = x0.clone().detach()
    n_samples, p = x.shape
    x.requires_grad = True
    if store:
        storage = []
        timer = []
        t0 = time()
    loss_list = []
    n = None
    for i in range(max_iter + 1):
        if store:
            timer.append(time() - t0)
            storage.append(x.clone())
        scores_x = score(x)
        # if kernel == 'gaussian':
        K = gaussian_stein_kernel_single(x, scores_x, bw)
        # else:
        #     K = imq_kernel(x, x, scores_x, scores_x, g=bw, beta=beta)
        loss = K.sum() / n_samples ** 2
        loss.backward()
        loss_list.append(loss.item())
        if verbose and i % 100 == 0:
            print(i, loss.item())
        with torch.no_grad():
            x[:] -= step * x.grad
            if n is not None:
                x[:] -= n
            if clamp is not None:
                x = x.clamp(clamp[0], clamp[1])
            x.grad.data.zero_()
        x.requires_grad = True
    x.requires_grad = False
    if store:
        return x, storage, timer
    else:
        return x

#
# def ksdd_lbfgs(x0, score, kernel='gaussian', bw=1.,
#                max_iter=10000, tol=1e-12, beta=.5,
#                store=False, verbose=False):
#     '''Kernel Stein Discrepancy descent with L-BFGS
#
#     Perform Kernel Stein Discrepancy descent with L-BFGS.
#     L-BFGS is a fast and robust algorithm, that has no
#     critical hyper-parameter.
#
#     Parameters
#     ----------
#     x0 : torch.tensor, size n_samples x n_features
#         initial positions
#
#     score : callable
#         function that computes the score
#
#     kernl : 'gaussian' or 'imq'
#         which kernel to choose
#
#     max_iter : int
#         max numer of iters
#
#     bw : float
#         bandwidth of the stein kernel
#
#     tol : float
#         stopping criterion for L-BFGS
#
#     store : bool
#         whether to stores the iterates
#
#     verbose: bool
#         wether to print the current loss
#
#     Returns
#     -------
#     x: torch.tensor, size n_samples x n_features
#         The final positions
#
#     References
#     ----------
#     A.Korba, P-C. Aubin-Frankowski, S.Majewski, P.Ablin.
#     Kernel Stein Discrepancy Descent
#     International Conference on Machine Learning, 2021.
#     '''
#     x = x0.clone().detach()
#
#     n_samples, p = x.shape
#     if store:
#         class callback_store():
#             def __init__(self):
#                 self.t0 = time()
#                 self.mem = []
#                 self.timer = []
#
#             def __call__(self, x):
#                 self.mem.append(torch.copy(x))
#                 self.timer.append(time() - self.t0)
#
#             def get_output(self):
#                 storage = [torch.tensor(x.reshape(n_samples, p),
#                                         dtype=torch.float32)
#                            for x in self.mem]
#                 return storage, self.timer
#         callback = callback_store()
#     else:
#         callback = None
#
#     def loss_and_grad(x):
#         x.requires_grad = True
#         scores_x = score(x)
#         if kernel == 'gaussian':
#             stein_kernel = gaussian_stein_kernel_single(x, scores_x, bw)
#         else:
#             stein_kernel = linear_stein_kernel(x, x, scores_x, scores_x)
#         loss = stein_kernel.sum()
#         loss.backward()
#         grad = x.grad
#         return loss.item(), np.float64(grad.numpy().ravel())
#
#     t0 = time()
#     options={
#         'maxiter': max_iter,
#         'gtol': 1e-10,
#     }
#     x, f, d = minimize(loss_and_grad, x.ravel(), method='BFGS',
#                             tol=tol, options=options)
#     if verbose:
#         print('Took %.2f sec, %d iterations, loss = %.2e' %
#               (time() - t0, d['nit'], f))
#     output = torch.tensor(x.reshape(n_samples, p), dtype=torch.float32)
#     if store:
#         storage, timer = callback.get_output()
#         return output, storage, timer
#     return output
#

# import numpy as np
from scipy.spatial.distance import squareform, pdist
# from torch.nn.functional import pdist
from torch import Tensor
import math
class SVGD():

    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):
        # sq_dist = pdist(theta)
        # pairwise_dists = (torch.norm(sq_dist) ** 2).to(theta)
        sq_dist = pdist(theta.cpu().detach().numpy())
        pairwise_dists = squareform(sq_dist) ** 2
        pairwise_dists = Tensor(pairwise_dists).to(theta)
        if h < 0:  # if h < 0, using median trick
            h = torch.median(pairwise_dists)
            h = torch.sqrt(0.5 * h / math.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = torch.exp(-pairwise_dists / (h**2+1e-9) / 2)

        dxkxy = -torch.matmul(Kxy, theta)
        sumkxy = torch.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + torch.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1, alpha=0.9):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = x0.clone()

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h=bandwidth)
            grad_theta = (torch.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

            # adagrad
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = torch.divide(grad_theta, fudge_factor + torch.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

        return theta
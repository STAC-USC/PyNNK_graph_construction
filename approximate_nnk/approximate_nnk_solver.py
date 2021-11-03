# __author__ = "shekkizh"
# """Custom solver for solving batched nnk objective - motivated by ISTA algorithm"""

import torch


@torch.no_grad()
def approximate_nnk(AtA, b, x_init, x_tol=1e-6, num_iter=100, eta=0.05, normalize=False):
    """
    Solves an approximation to Non-Negative Kernel regression by ISTA type optimization
    objective = min_x>0 0.5*(xt*AtA*x) - xt*b
    :param AtA: shape = [batch_size, S, S] where S is the size of the subset of data neighbors selected
    :param b: shape = [batch_size, S, 1]
    :param x_init: starting point for optimization. shape = [batch_size, S, 1]
    :param x_tol: minimum value of weight allowed for solution
    :param num_iter: maximum number of iterations to run the optimization
    :param eta: learning rate for iterative update. When "None", the value is set to 1/max_eig_value
    :param normalize: Boolean flag for normalizing the solution to sum to 1
    :return: x_opt: shape [batch_size, S], Approximate solution to NNK
             error: shape [batch_size], Error associated with current solution
    """
    if eta is None:  # this slows down the solve - can get a fixed eta by taking mean over some sample
        L = torch.max(torch.linalg.eigvals(AtA).abs(), 1, keepdim=True)[0]
        eta = 1. / L

    b = b.unsqueeze(2)
    x_opt = x_init.unsqueeze(2)
    for t in range(num_iter):
        x_opt = x_opt.add(eta * b.sub(torch.bmm(AtA, x_opt))).clamp(min=0, max=1.)

    error = 0.5 - torch.sum(x_opt * b.sub(0.5 * torch.bmm(AtA, x_opt)), dim=1)
    if normalize:
        x_opt = torch.nn.functional.normalize(x_opt.squeeze(), p=1, dim=1)
    else:
        x_opt = x_opt.squeeze()
    x_opt[x_opt < x_tol] = 0
    return x_opt, error.squeeze()

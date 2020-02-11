__author__ = "shekkizh"
"""Custom solver for solving non negative quadratic programs with positive definite matrices"""

import numpy as np
from scipy.linalg.decomp_cholesky import cho_factor, cho_solve


def non_negative_qpsolver(A, b, x_init, x_tol, check_tol=-1):
    '''
    Solves (1/2)x.T A x - b.T x
    :param x_init: Initial value for solution x
    :param x_tol: Smallest allowed non zero value for x. Values below x_tol are zero'd out
    :param check_tol: Allowed tolerance for stopping criteria
    :return: x_opt, error
    '''
    epsilon = np.finfo(float).eps
    if check_tol < 0:
        check_tol = x_tol

    n = A.shape[0]
    # A = A + 1e-6 * np.eye(n)
    maxiter = 50 * n
    itr = 0
    x_opt = np.reshape(x_init, (n, 1))
    non_pruned_elements = np.ones((n, 1), dtype=np.bool)  # F tracks elements pruned based on negativity
    check = 1
    while (check > check_tol) and (itr < maxiter):
        non_pruned_elements = np.logical_and(x_opt > epsilon, non_pruned_elements)
        x_opt = np.zeros((n, 1))
        x_opt[non_pruned_elements] = cho_solve(
            cho_factor(A[non_pruned_elements[:, 0], :][:, non_pruned_elements[:, 0]]),
            b[non_pruned_elements[:, 0]])

        itr = itr + 1
        N = x_opt < epsilon
        if sum(N) > 0:
            check = np.max(np.abs(x_opt[N]))
        else:
            check = 0

    x_opt[x_opt < x_tol] = 0
    return x_opt[:,0], check

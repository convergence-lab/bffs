from __future__ import print_function

__author__ = "Masashi Kimura <kimura@convergence-lab.com>"
__status__ = "production"
__version__ = "0.0.0"
__date__    = "3 December 2018"

import numpy as np
import scipy
from  scipy.sparse import csc_matrix
import miosqp
import time

from sklearn.base import BaseEstimator

class BF(BaseEstimator):
    def __init__(self, itermax=100, verbose=False, n_jobs=1, delta=1e-3, eps=1e-6, seed=12345):
        self._itermax = itermax
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._delta = delta
        self._eps = eps
        # np.random.seed(seed)
    def fit(self, X, y):
        if len(X.shape) != 2:
            raise ValueError("the dimension of X should be 2.")

        self._X = X
        self._y = y

        self._data_len = len(X)
        a = 1
        c = X.shape[1]

        for i in range(self._itermax):
            b = (a + c) //2
            while a != b:
                b = (a + c) //2
                fa, _, _ = self._solve_MIQP(a)
                fb, _, _ = self._solve_MIQP(b)
                fc, coef, argx = self._solve_MIQP(c)
                rdab = (fa - fb) / (fa * abs(b-a + self._eps))
                rdbc = (fb - fc) / (fb * abs(b-c + self._eps))
                if rdbc > self._delta and rdab > -self._delta:
                    a = b
                else:
                    c = b
            if c == 1:
                k0 = c + 1
                fk0 = c + 1
            else:
                k0 = c
                fk0 = c
            fk0m1, _, _ = self._solve_MIQP(k0-1)
            fk0p1, _, _ = self._solve_MIQP(k0+1)
            rdm1_0 = abs((fk0m1 - fk0) / (fk0m1 * abs(fk0 - fk0m1)))
            rd0_p1 = abs((fk0 - fk0p1) / (fk0 * abs(fk0p1 - fk0)))
            cost = min(rdm1_0, rd0_p1)
            if self._verbose:
                print(f"iteration {i}: cost {cost}")
            if cost >= self._eps:
                break
            else:
                a = 1
        self._cost = cost
        self._selected_x = argx
        self._coef = coef[argx]
        return c

    def transform(self, X):
        return X[:, self._selected_x]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def selected(self):
        return self._selected_x

    def coef(self):
        return self._coef

    def _solve_MIQP(self, k):
        n = self._data_len
        m = self._X.shape[1]
        i_idx = np.random.choice(np.arange(0, m), k)
        beta = self._X
        P = csc_matrix(2*np.dot(beta.T, beta))
        q = - 2 * np.dot(self._X.T, self._y)
        A = csc_matrix(np.ones((m, m)))
        l = k * np.ones(m)
        u = k * np.ones(m)
        i_l = np.zeros(k, dtype=np.int)
        i_u = np.ones(k, dtype=np.int)
        miosqp_settings = {
            # integer feasibility tolerance
            'eps_int_feas': 1e-03,
            # maximum number of iterations
            'max_iter_bb': 1000,
            # tree exploration rule
            #   [0] depth first
            #   [1] two-phase: depth first until first incumbent and then  best bound
            'tree_explor_rule': 1,
            # branching rule
            #   [0] max fractional part
            'verbose': False,
            'branching_rule': 0,
            'print_interval': 1}

        osqp_settings = {
            'eps_abs': 1e-03,
            'eps_rel': 1e-03,
            'eps_prim_inf': 1e-04,
            'verbose': False}

        model = miosqp.MIOSQP()
        model.setup(P, q, A, l, u, i_idx, i_l, i_u,
                    miosqp_settings,
                    osqp_settings)
        result = model.solve()
        argx = np.argsort(result.x)[::-1][:k]
        return result.upper_glob, result.x, argx

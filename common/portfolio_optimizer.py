import pandas as pd
import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer(object):
    def __init__(self):
        return


    def _calc_utility_reversed_sign(self, w, alpha, cov, l):
        '''
        l: Lagrange multiplier
        '''

        stats = self._calc_stats(w, alpha, cov)
        port_alpha = stats['alpha']
        port_variance = stats['var']
        u = -1 * (port_alpha - l * port_variance)
        return u


    def _calc_stats(self, w, alpha, cov):

        w = w.reshape(-1, 1)

        # calculate total alpha
        port_alpha = (alpha.values * w).sum()

        # calculate portfolio variance
        V = np.matrix(cov.values)
        port_variance =  (w.T * V * w)[0, 0]
        return {'alpha': port_alpha, 'var': port_variance}


    def optimize(self, alpha, cov, l, bounds, constraints):
        '''
        l: Lagrange multiplier
        '''

        w_init = np.array(len(alpha) * [1 / len(alpha)])  # equal weight
        opt = minimize(self._calc_utility_reversed_sign, w_init, args=(alpha, cov, l), method='SLSQP', 
                       bounds=bounds, constraints=constraints, tol=1e-10)
        w_opt = opt.x
        stats = self._calc_stats(w_opt, alpha, cov)
        return w_opt, stats

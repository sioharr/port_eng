import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from scipy.optimize import minimize
from .db import create_engine_for_db
from .risk_model import RiskModel


class PortfolioConstructor(object):
    def __init__(self, univ, alpha, risk_model, engine=None):
        self.univ = univ
        self.alpha = alpha
        
        self._engine = engine or create_engine_for_db()
        self._rm = RiskModel(risk_model, self._engine)


    def _load_alpha_data(self, date):
        sql = f'''
            select id, value
            from alphas
            where date = '{date.strftime('%Y%m%d')}'
            and factor = '{self.attrib}'
        '''

        data = pd.read_sql(sql, con=self._engine)
        data.set_index('id', inplace=True)
        return data


    def _load_univ_data(self, date):
        sql = f'''
            select slug
            from benchmark_wt_kaggle
            where name = '{self.univ}'
            and date = '{date.strftime('%Y%m%d')}'
        '''

        univ = pd.read_sql(sql, con=self._engine)['slug'].to_list()
        return univ


    def load_data(self, date):
        cov = self._rm.load_cov(date)
        univ = self._load_univ_data(date)
        alpha = self._load_alpha_data(date)
        
        # get the intersection
        univ = cov.index.intersection(univ).intersection(alpha.index)
        cov = cov.reindex(univ, axis=0).reindex(univ, axis=1)
        V = np.matrix(cov.values)
        alpha = alpha.reindex(univ)


    def clac_port_stats(self, univ, cov, alpha, w):
        # winsorize
        alpha['value'] = winsorize(alpha['value'], limits=(0.05, 0.05))
        # standardize
        alpha = (alpha - alpha.mean()) / alpha.std()

        # calculate total alpha
        port_alpha = (alpha * w).sum()

        # calculate portfolio variance
        port_variance =  w.T * V * w

        return port_alpha, port_variance


    def mapto_constraints(self, constraints):
        pass

    def optimze(self):
        w_init = 1 / len(univ) # equal weight
        

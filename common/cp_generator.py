import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from .db import create_engine_for_db
from .risk_model import RiskModel


class CpGenerator(object):
    def __init__(self, model, univ, attribute, engine=None):
        self.attrib = attribute
        self.univ = univ
        self._engine = engine or create_engine_for_db()
        self._rm = RiskModel(model, self._engine)


    def _load_attrib_data(self, date):
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


    def generate_cp(self, date):
        cov = self._rm.load_cov(date)
        univ = self._load_univ_data(date)
        a = self._load_attrib_data(date)

        # get the intersection
        univ_valid = cov.index.intersection(univ).intersection(a.index)
        cov = cov.reindex(univ_valid, axis=0).reindex(univ_valid, axis=1)
        V = np.matrix(cov.values)      
        a = a.reindex(univ_valid)

        # winsorize
        a['value'] = winsorize(a['value'], limits=(0.05, 0.05))
        # standardize
        a = (a - a.mean()) / a.std()
        a = np.matrix(a)

        # calculate weights
        V_i = np.linalg.inv(V)
        h = (V_i * a) / (a.T * V_i * a)
        h = pd.DataFrame(index=univ_valid, data=h, columns=['weight'])
        return h

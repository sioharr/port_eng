import pandas as pd
import numpy as np
import functools


class RiskModel(object):
    def __init__(self, model, engine=None):
        self.model = model
        if not engine:
            from .db import create_engine_for_db
            engine = create_engine_for_db()
        self._engine = engine


    @functools.lru_cache(maxsize=128)
    def load_exp(self, date):
        sql = f'''
            select *
            from market_data.vw500_exp
            where date = {date.strftime('%Y%m%d')}
        '''

        data = pd.read_sql(sql, con=self._engine)
        data = data.pivot(index='id', columns='factor', values='value')
        data.sort_index(axis=1, inplace=True)
        return data


    @functools.lru_cache(maxsize=128)
    def load_fact_cov(self, date):
        sql = f'''
            select *
            from market_data.vw500_cov_matrix
            where date = {date.strftime('%Y%m%d')}
        '''

        data = pd.read_sql(sql, con=self._engine)
        data = data.pivot(index='factor1', columns='factor2', values='cov_over_pct_ret_annualized')
        data.sort_index(inplace=True)
        data.sort_index(axis=1, inplace=True)
        
        return data


    @functools.lru_cache(maxsize=128)
    def load_srisk(self, date):
        sql = f'''
            select *
            from market_data.vw500_spec_risk_pct_annualized
            where date = {date.strftime('%Y%m%d')}
        '''

        data = pd.read_sql(sql, con=self._engine)
        data = data.set_index('name')[['spec_risk']]
        return data


    def load_cov(self, date):
        exp = self.load_exp(date)
        fact_cov = self.load_fact_cov(date)
        srisk = self.load_srisk(date)

        ids = exp.index.intersection(srisk.index)
        exp = exp.reindex(ids).values
        srisk = np.diag(srisk.reindex(ids)['spec_risk'].values)
        ## TODO: sort fact_cov and exp columns

        exp = np.matrix(exp)
        fact_cov = np.matrix(fact_cov)
        srisk = np.matrix(srisk)
        
        cov = exp * fact_cov * exp.T + srisk
        cov_df = pd.DataFrame(data=cov, columns=ids, index=ids)
        return cov_df

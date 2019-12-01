import pandas as pd
import numpy as np
from .db import create_engine_for_db


class CpAnalyzer(object):
    def __init__(self, factor_name, mode='daily', holding_period=30, peak_ahead=0, engine=None):
        self.factor_name = factor_name
        self.mode = mode
        self.holding_period = holding_period
        self.peak_ahead = peak_ahead
        self._engine = engine or create_engine_for_db()

    def _load_ret(self, date):
        if self.mode == 'daily':
            start_date = date - pd.DateOffset(self.peak_ahead)
            stop_date = date + pd.DateOffset(self.holding_period)
        elif self.mode == 'monthly':
            start_date = date - pd.DateOffset(months=self.peak_ahead)
            stop_date = date + pd.DateOffset(months=self.holding_period)
        
        sql = f'''
            select `slug` as `id`, `date`, `daily` as `return`
            from return_kaggle
            where slug in (
                select id
                from characteristic_portfolio
                where factor = '{self.factor_name}'
                and date = '{date.strftime('%Y%m%d')}'
            )
            and date between '{start_date.strftime('%Y%m%d')}' and '{stop_date.strftime('%Y%m%d')}'
            order by `id`
        '''

        data = pd.read_sql(sql, con=self._engine)
        # cap the daily return at 100%
        data = data.pivot(index='date', columns='id', values='return').fillna(0)
        data.clip(-1.0, 1.0, inplace=True)

        return data


    def _load_cp_weight(self, date):
        sql = f'''
            select id, weight
            from characteristic_portfolio
            where factor = '{self.factor_name}'
            and date = '{date.strftime('%Y%m%d')}'
            order by `id`
        '''

        data = pd.read_sql(sql, con=self._engine)
        return data.set_index('id')


    def _calc_hdp_ret(self, date):
        wt = self._load_cp_weight(date)
        ret = self._load_ret(date)

        
        if self.mode == 'monthly':
            lret = np.log(ret + 1)
            cumlret = lret.resample('1M').sum()
            cumret = np.exp(cumlret) - 1
            cumret['holding_period'] = np.round((cumret.index - date) / np.timedelta64(1, 'M'))
            cumret['holding_period'] = cumret['holding_period'].astype(int)

        elif self.mode == 'daily':
            cumret = ret
            cumret['holding_period'] = (cumret.index - date).days

        cumret.set_index('holding_period', inplace=True)
        hpret = cumret.apply(lambda row: (row * wt['weight']).sum(), axis=1)
        hpret = hpret.to_frame().T
        hpret.index.name = 'cp_date'
        hpret.index = [date]
        return hpret

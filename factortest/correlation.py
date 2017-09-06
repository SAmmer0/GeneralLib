#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-09-05 09:12:42
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
本模块用于计算因子的IC和自相关性
'''
# 本地文件
from fmanager.factors.utils import convert_data
from fmanager.api import get_factor_dict
from factortest.const import WEEKLY, MONTHLY
from factortest.utils import HDFDataProvider, MonRebCalcu, WeekRebCalcu


class ICCalculator(object):
    '''
    用于计算因子的IC，即期初的因子值与期末的收益率之间的相关性
    '''

    def __init__(self, factor_provider, quote_provider, reb_calc):
        '''
        Parameter
        ---------
        factor_provider: DataProvider
            因子值数据
        quote_provider: DataProvider
            用于计算收益的行情数据（一般为复权后价格数据）
        reb_calc: RebCalcu
            用于获取时间分段的数据
        '''
        self._factor_provider = factor_provider
        self._quote_provider = quote_provider
        self._reb_dates = reb_calc.reb_points

    def __call__(self):
        '''
        计算IC
        Return
        ------
        out: pd.Series
            Index为升序排序后的换仓时间，数据为对应的IC值，最后一个换仓日没有对应的收益，值设置为NA，
            后续时间内，如果有股票退市，直接将其收益和因子值做提出处理
        '''
        def calc_IC(df):
            f = df.xs('factor', level=1).iloc[0]
            p = df.xs('quote', level=1).iloc[0]
            return f.corr(p)
        # 加载数据
        start_time = min(self._reb_dates)
        end_time = max(self._reb_dates)
        factor_data = self._factor_provider.get_paneldata(start_time, end_time).\
            reindex(self._reb_dates)
        quote_data = self._quote_provider.get_paneldata(start_time, end_time).\
            reindex(self._reb_dates)
        quote_data = quote_data.pct_change().shift(-1)
        merged_data = convert_data([factor_data, quote_data], ['factor', 'quote'])
        by_time = merged_data.groupby(level=0)
        out = by_time.apply(calc_IC)
        return out


class FactorICTemplate(object):
    '''
    计算因子IC的模板
    '''

    def __init__(self, factor_name, start_time, end_time, reb_type=MONTHLY):
        '''
        Parameter
        ---------
        factor_name: str
            因子的名称，要求必须能在fmanager.api.get_factor_dict中找到
        start_time: datetime or other compatible type
            测试的开始时间
        end_time: datetime or other compatible type
            测试的结束时间
        reb_type: str, default MONTHLY
            换仓日计算的规则，目前只支持月度(MONTHLY)和周度(WEEKLY)
        '''
        self._start_time = start_time
        self._end_time = end_time
        factor_dict = get_factor_dict()
        self._factor_provider = HDFDataProvider(factor_dict[factor_name]['abs_path'],
                                                start_time, end_time)
        self._load_rebcalculator(reb_type)
        self._quote_provider = HDFDataProvider(factor_dict['ADJ_CLOSE']['abs_path'],
                                               start_time, end_time)

    def _load_rebcalculator(self, reb_type):
        '''
        加载换仓日计算器
        Parameter
        ---------
        reb_type: str
            换仓日计算的规则，目前只支持月度(MONTHLY)和周度(WEEKLY)
        '''
        valid_freq = [MONTHLY, WEEKLY]
        assert reb_type in valid_freq, \
            'Rebalance date method setting ERROR, you provide {yp}, '.format(yp=reb_type) +\
            'right choices are {rc}'.format(rc=valid_freq)
        if reb_type == MONTHLY:
            self._rebcalculator = MonRebCalcu(self._start_time, self._end_time)
        else:
            self._rebcalculator = WeekRebCalcu(self._start_time, self._end_time)

    def run(self):
        '''
        进行IC的计算
        '''
        calculator = ICCalculator(self._factor_provider, self._quote_provider, self._rebcalculator)
        return calculator()

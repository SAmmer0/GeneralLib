#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-09-05 09:12:42
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
本模块用于计算因子的IC和自相关性
'''
# 系统库文件
from collections import namedtuple
import pdb
# 第三方库
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
# 本地文件
from fmanager.factors.utils import convert_data
from fmanager import get_factor_dict
from factortest.const import WEEKLY, MONTHLY
from factortest.utils import HDFDataProvider, load_rebcalculator, NoneDataProvider


class ICCalculator(object):
    '''
    用于计算因子的IC（包括滞后IC），即期初的因子值与期末的收益率之间的相关性
    '''

    def __init__(self, factor_provider, quote_provider, reb_calc, universe_provider, offset=1):
        '''
        Parameter
        ---------
        factor_provider: DataProvider
            因子值数据
        quote_provider: DataProvider
            用于计算收益的行情数据（一般为复权后价格数据）
        reb_calc: RebCalcu
            用于获取时间分段的数据
        universe_provider: DataProvider, default None
            用于规定计算IC过程中的Universe
        offset: int, default 1
            用于设置因子值与收益率之间相隔的期数，要求为不小于1的整数,1即表示传统的IC（滞后一期）
        '''
        self._factor_provider = factor_provider
        self._quote_provider = quote_provider
        self._reb_dates = reb_calc.reb_points
        self._offset = offset
        self._universe_provider = universe_provider

    def __call__(self):
        '''
        计算（滞后）IC
        Return
        ------
        out: namedtuple(ICAnalysisResult)
            包含两个结果，IC和Rank IC，对于每个数据Index为升序排序后的换仓时间，数据为对应的IC值，
            最后一个换仓日没有对应的收益，值设置为NA，后续时间内，如果有股票退市，直接将其收益和因子值做剔除处理
        '''
        def calc_IC(df):
            # pdb.set_trace()
            universe = df.iloc[2]
            f = df.iloc[1].loc[universe == 1]
            p = df.iloc[0].loc[universe == 1]
            return f.corr(p)

        def calc_RankIC(df):
            df = df.dropna(axis=1)
            universe = df.iloc[2]
            f = df.iloc[1].loc[universe == 1]
            p = df.iloc[0].loc[universe == 1]
            return spearmanr(f, p).correlation

        # 加载数据
        start_time = min(self._reb_dates)
        end_time = max(self._reb_dates)
        factor_data = self._factor_provider.get_paneldata(start_time, end_time).\
            reindex(self._reb_dates)
        quote_data = self._quote_provider.get_paneldata(start_time, end_time).\
            reindex(self._reb_dates)
        universe_data = self._universe_provider.get_paneldata(start_time, end_time)
        # pdb.set_trace()
        if universe_data is None:  # 没有对universe做要求
            universe_data = pd.DataFrame(np.ones((len(factor_data), len(factor_data.columns))),
                                         index=factor_data.index, columns=factor_data.columns)
        else:
            universe_data = universe_data.reindex(self._reb_dates)
        quote_data = quote_data.pct_change().shift(-self._offset)
        merged_data = convert_data([factor_data, quote_data, universe_data],
                                   ['factor', 'quote', 'universe'])
        by_time = merged_data.groupby(level=0)
        ic = by_time.apply(calc_IC)
        rank_ic = by_time.apply(calc_RankIC)
        ICAnalysisResult = namedtuple('ICAnalysisResult', ['IC', 'Rank_IC'])
        out = ICAnalysisResult(IC=ic, Rank_IC=rank_ic)
        return out


class FactorICTemplate(object):
    '''
    计算因子IC的模板
    '''

    def __init__(self, factor_name, start_time, end_time, universe=None, offset=1,
                 reb_type=MONTHLY):
        '''
        Parameter
        ---------
        factor_name: str
            因子的名称，要求必须能在fmanager.get_factor_dict中找到
        start_time: datetime or other compatible type
            测试的开始时间
        end_time: datetime or other compatible type
            测试的结束时间
        universe: str or DataProvider, default None
            使用的universe名称，如果为str类型，要求必须能在fmanager.list_allfactor()中找到
        offset: int, default 1
            因子与收益率之间相隔的期数，要求为不小于1的整数，1即表示传统的IC
        reb_type: str, default MONTHLY
            换仓日计算的规则，目前只支持月度(MONTHLY)和周度(WEEKLY)
        '''
        # self._start_time = start_time
        # self._end_time = end_time
        factor_dict = get_factor_dict()
        self._factor_provider = HDFDataProvider(factor_dict[factor_name]['abs_path'],
                                                start_time, end_time)
        self._rebcalculator = load_rebcalculator(reb_type, start_time, end_time)
        self._quote_provider = HDFDataProvider(factor_dict['ADJ_CLOSE']['abs_path'],
                                               start_time, end_time)
        if universe is None:
            self._universe_provider = NoneDataProvider()
        elif isinstance(universe, str):
            self._universe_provider = HDFDataProvider(factor_dict[universe]['abs_path'],
                                                      start_time, end_time)
        else:
            self._universe_provider = universe
        self._offset = offset

    def __call__(self):
        '''
        进行IC的计算
        '''
        calculator = ICCalculator(self._factor_provider, self._quote_provider, self._rebcalculator,
                                  self._universe_provider, self._offset)
        return calculator()


class ICDecay(object):
    '''
    计算因子（rank）IC的衰减情况
    具体如下：
        假设计算10期IC衰减情况，则分别计算每一个滞后期对应的IC的序列，然后求平均值，返回这10期的
        所有的IC平均值
    '''

    def __init__(self, factor_name, start_time, end_time, universe=None, period_num=10,
                 reb_type=MONTHLY):
        '''
        Parameter
        ---------
        factor_name: str
            因子名称，要求能在fmanager.get_factor_dict中找到
        start_time: datetime like
            测试的开始时间
        end_time: datetime like
            测试的结束时间
        universe: str
            使用的universe名称，要求能在fmanager.list_allfactor()中找到
        period_num: int, default 10
            IC衰减的最大期数
        reb_type: str, default MONTHLY
            换仓日计算的规则，目前只支持月度(MONTHLY)和周度(WEEKLY)
        '''
        factor_dict = get_factor_dict()
        _factor_provider = HDFDataProvider(factor_dict[factor_name]['abs_path'],
                                           start_time, end_time)
        _rebcalculator = load_rebcalculator(reb_type, start_time, end_time)
        _quote_provider = HDFDataProvider(factor_dict['ADJ_CLOSE']['abs_path'],
                                          start_time, end_time)
        if universe is None:
            _universe_provider = NoneDataProvider()
        else:
            _universe_provider = HDFDataProvider(factor_dict[universe]['abs_path'],
                                                 start_time, end_time)
        self._IC_calcus = {offset: ICCalculator(_factor_provider, _quote_provider,
                                                _rebcalculator, _universe_provider, offset)
                           for offset in range(1, period_num + 1)}

    def __call__(self):
        '''
        计算不同滞后期的平均IC

        Return
        ------
        out: pd.Series
            index为滞后期的期数，value为平均的IC
        '''
        res = {offset: calculator() for offset, calculator in self._IC_calcus.items()}
        ic_decay = {offset: result.IC for offset, result in res.items()}
        rankic_decay = {offset: result.Rank_IC for offset, result in res.items()}
        ic_decay = pd.DataFrame(ic_decay)
        rankic_decay = pd.DataFrame(rankic_decay)
        ICDecayResult = namedtuple('ICDecayResult', ['IC', 'Rank_IC'])
        out = ICDecayResult(IC=ic_decay, Rank_IC=rankic_decay)
        return out


class FactorAutoCorrelation(FactorICTemplate):
    '''
    计算因子的自相关函数
    '''

    def __init__(self, factor_name, start_time, end_time, universe=None, reb_type=MONTHLY):
        super().__init__(factor_name, start_time, end_time, universe=universe,
                         reb_type=reb_type)

    def __call__(self):
        '''
        进行自相关性的计算
        '''
        def acf(df):    # 计算自相关系数
            universe = df.iloc[2]
            f = df.iloc[0].loc[universe == 1]
            p = df.iloc[1].loc[universe == 1]
            return f.corr(p)

        def rank_acf(df):  # 计算排序的自相关系数
            df = df.dropna(axis=1)
            universe = df.iloc[2]
            f = df.iloc[0].loc[universe == 1]
            p = df.iloc[1].loc[universe == 1]
            return spearmanr(f, p).correlation
        reb_dates = self._rebcalculator.reb_points
        start_time = min(reb_dates)
        end_time = max(reb_dates)
        factor_data = self._factor_provider.get_paneldata(start_time, end_time).reindex(reb_dates)
        last_factor_data = factor_data.shift(1)
        # 加载universe数据
        universe_data = self._universe_provider.get_paneldata(start_time, end_time)
        if universe_data is None:  # 没有universe的限制
            universe_data = pd.DataFrame(np.ones((len(factor_data), len(factor_data.columns))),
                                         index=factor_data.index, columns=factor_data.columns)
        else:
            universe_data = universe_data.reindex(reb_dates)
        merged_data = convert_data([factor_data, last_factor_data, universe_data],
                                   ['now', 'last', 'universe'])
        by_time = merged_data.groupby(level=0)
        acf_res = by_time.apply(acf)
        racf_res = by_time.apply(rank_acf)
        AutoCorrelationResult = namedtuple('AutoCorrelationResult', ['acf', 'Rank_acf'])
        return AutoCorrelationResult(acf=acf_res, Rank_acf=racf_res)

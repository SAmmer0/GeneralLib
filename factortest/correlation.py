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
from collections import namedtuple, OrderedDict
import pdb
from functools import reduce
# 第三方库
from scipy.stats import spearmanr
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
# 本地文件
from fmanager.factors.utils import convert_data
from fmanager import get_factor_dict, query, get_factor_detail, get_universe
from factortest.const import WEEKLY, MONTHLY
from factortest.utils import HDFDataProvider, load_rebcalculator, NoneDataProvider
from datatoolkits import winsorize, standardlize, extract_factor_OLS

# --------------------------------------------------------------------------------------------------
# 类


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
        def calc_IC(df, method):
            # pdb.set_trace()
            universe = df.iloc[2]
            f = df.iloc[1].loc[universe == 1]
            p = df.iloc[0].loc[universe == 1]
            return f.corr(p, method=method)

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
        ic = by_time.apply(calc_IC, method='pearson')
        rank_ic = by_time.apply(calc_IC, method='spearman')
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
    计算因子的自相关函数，即在每个时间点本期的因子值与上期的因子值的横截面相关性
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


# --------------------------------------------------------------------------------------------------
# 函数
def fv_correlation(factors, start_time, end_time, freq=MONTHLY, average=True, method='pearson'):
    '''
    计算不同因子的因子值之间的相关系数矩阵

    Parameter
    ---------
    factors: list like
        因子名称列表，必须能够在fmanager.list_allfactor()中找到
    start_time: datetime like
        计算相关系数矩阵的起始时间
    end_time: datetime like
        计算相关系数矩阵的终止时间
    freq: str, default const.MONTHLY
        计算协方差矩阵的频率，目前只支持周度（WEEKLY）和月度（MONTHLY）
    average: boolean, default True
        是否返回相关系数矩阵平均后的值，默认进行平均的处理
    method: string, default pearson
        计算相关系数的方法，支持['pearson', 'spearman', 'kendall']

    Return
    ------
    out: pd.DataFrame or OrderDict
        如果average参数为True，则返回pd.DataFrame，反之返回OrderDict，key为计算的时间，value为相关系数矩阵
    '''
    rebs = load_rebcalculator(freq, start_time, end_time)
    datas = []
    for f in factors:
        tmp_data = query(f, (start_time, end_time))
        tmp_data = tmp_data.reindex(rebs.reb_points)
        datas.append(tmp_data)
    datas = convert_data(datas, factors)
    by_time = datas.groupby(level=0)
    out = OrderedDict()
    for t in by_time.groups:
        tmp = by_time.get_group(t).reset_index(level=0, drop=True)
        out[t] = tmp.T.corr(method=method)
    if average:
        out = reduce(lambda x, y: x + y, out.values()) / len(out)
    return out


def _get_stocks_character(stocks, character_data, method='median'):
    '''
    辅助计算函数，根据给定的股票名称和特征数据，计算该组股票对应的特征数据

    Parameter
    ---------
    stocks: list like
        需要计算特征数据的股票列表
    character_data: pd.Series
        股票特征的数据
    method: str, default median
        综合计算该组股票特征的方法，支持计算中位数（median）和均值(mean)，默认为中位数

    Return
    ------
    out: float
        对应该组股票的特征值
    '''
    data = character_data.reindex(stocks)
    if method == 'median':
        out = data.median()
    elif method == 'mean':
        out = data.mean()
    else:
        raise ValueError('Unsupported method({mtd})'.format(mtd=method))
    return out


def get_group_factorcharacter(group, factor_name, method='median'):
    '''
    计算给定组合的某个因子的特征值的时间序列

    Parameter
    ---------
    group: dict
        格式为{time: list}，list中的数据为股票代码
    factor_name: str
        需要计算的特征值的因子，要求能在fmanager.list_allfactor()中找到
    method: str, default median
        给每组计算对应综合特征值的方法，目前只支持均值（mean）和中位数（median），默认为中位数

    Return
    ------
    out: pd.Series
        给定组合的特征值时间序列，index为时间，即group中的key
    '''
    start_time = min(group.keys())
    end_time = max(group.keys())
    factor_path = get_factor_detail(factor_name)['abs_path']
    factor_data = HDFDataProvider(factor_path, start_time, end_time)
    out = {}
    for t in sorted(group):
        out[t] = _get_stocks_character(group[t], factor_data.get_csdata(t), method)
    return pd.Series(out)


def factor_purify(tobe_purified, other_factors, start_time, end_time, normalize=True,
                  winsorize_threshold=0.01, universe=None):
    '''
    使用回归的方法剔除其他因子对目标因子的影响，即使用目标因子对其他因子做横截面上的回归，
    然后取残差，作为新的因子值

    Parameter
    ---------
    tobe_purified: str
        需要被纯化的因子名称
    other_factors: list like
        作为自变量的因子，格式为[factor1, factor2, ...]
    start_time: datetime like
        纯化因子数据的开始时间
    end_time: datetime like
        纯化因子数据的结束时间
    normalize: boolean, default True
        是否在回归前对异常值进行winsorize处理，并将各个因子的数据转换为z-score
    winsorize_threshold: float, default 0.01
        在winsorize处理时传入的参数，共2*n*winsorize_threshold个数据将会被进行winsorize处理
    universe: iterable, default None
        股票的universe，默认None表示从fmanager.get_universe中获取

    Return
    ------
    out: pd.DataFrame
        经过纯化后的因子数据，index为时间，columns为universe中的股票代码
    '''
    # 加载数据
    raw_data = query(tobe_purified, (start_time, end_time))
    factors_data = list()
    factors_data.append(raw_data)
    for f in other_factors:
        tmp_data = query(f, (start_time, end_time))
        factors_data.append(tmp_data)
    factors_tag = [tobe_purified] + list(other_factors)

    # universe获取
    if universe is None:
        universe = get_universe()
    # 对数据进行正则化处理
    if normalize:
        new_data = list()
        for data in factors_data:
            tmp = data.apply(lambda x: standardlize(winsorize(x, (winsorize_threshold,
                                                                  1-winsorize_threshold))), axis=1)
            tmp = tmp.loc[:, sorted(universe)]
            new_data.append(tmp)
        factors_data = new_data
    data = convert_data(factors_data, factors_tag)
    by_time = data.groupby(level=0)

    def calc_resid(x):
        raw_index = x.columns
        x = x.reset_index(level=0, drop=True).T.dropna(axis=0, how='any')
        res = extract_factor_OLS(x, factor_col=tobe_purified, x_cols=other_factors,
                                 standardlization=False)
        # pdb.set_trace()
        res = res.reindex(raw_index)
        return res
    out = by_time.apply(calc_resid)
    return out

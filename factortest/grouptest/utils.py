#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-08-17 16:49:52
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
该模块定义一些工具类，用于测试
__version__ = 1.0.0
修改日期：2017-08-17
修改内容：
    初始化
'''
# 标准库
from abc import ABCMeta, abstractmethod
import pdb
# from functools import wraps
# 第三方库
import pandas as pd
from datatoolkits import isclose
import numpy as np
# 本地库
# from ..utils import NoneDataProvider
from fmanager.database import NaS
from fmanager import get_universe
# --------------------------------------------------------------------------------------------------
# 常量定义
CASH = 'Cash'

# --------------------------------------------------------------------------------------------------
# 证券类


class EqlWeightCalc(object):
    '''
    等权重持仓计算器
    '''

    def __init__(self):
        pass

    def calc_weight(self, secu_list, **kwargs):
        '''
        权重计算函数

        Parameter
        ---------
        secu_list: list like
            需要分配权重的证券列表
        kwargs: dict like arguments
            其他计算权重需要的参数，比如说市值加权时需要时间

        Return
        ------
        out: dict
            权重分配结果，格式为{secu_code: w}
        '''
        avg_w = 1. / len(secu_list)
        return dict(zip(secu_list, [avg_w] * len(secu_list)))

    def __call__(self, secu_list, **kwargs):
        '''
        功能同calc_weight，方便调用
        权重计算函数

        Parameter
        ---------
        secu_list: list like
            需要分配权重的证券列表
        kwargs: dict like arguments
            其他计算权重需要的参数，比如说市值加权时需要时间

        Return
        ------
        out: dict
            权重分配结果，格式为{secu_code: w}
        '''
        return self.calc_weight(secu_list, **kwargs)


class MkvWeightCalc(EqlWeightCalc):
    '''
    市值加权权重计算器
    '''

    def __init__(self, mkv_provider):
        '''
        Parameter
        ---------
        mkv_provider: DataProvider
            用于获取市值的数据提供器
        '''
        self._mkv_provider = mkv_provider

    def calc_weight(self, secu_list, **kwargs):
        '''
        权重计算函数

        Parameter
        ---------
        secu_list: list like
            需要分配权重的证券列表
        kwargs: dict like parameter
            其他计算权重所需的参数，必须包含'date'参数，'date'参数的类型为与datetime相兼容的类型

        Return
        ------
        out: dict
            权重分配结果，格式为{code: w}
        '''
        assert 'date' in kwargs, 'Error, "date" parameter must be provided!'
        date = pd.to_datetime(kwargs.get('date'))
        mkv = {code: self._mkv_provider.get_data(date, code) for code in secu_list}
        # pdb.set_trace()
        total_mkv = np.sum(list(mkv.values()))
        out = {code: mkv[code] / total_mkv for code in mkv}
        return out


# --------------------------------------------------------------------------------------------------
# 函数


def stock_filter_template(st_provider, tradedata_provider, stockpool_provider,
                          industry_provider, group_num):
    '''
    模板函数，用于生成一般排序回测函数

    Parameter
    ---------
    st_provider: HDFDataProvider
        记录ST数据的数据提供器
    tradedata_provider: HDFDataProvider
        记录是否可以交易的数据提供器
    stockpool_provider: DataProvider
        记录股票池的数据提供器，可以是NoneDataProvider，表明当前没有股票池的限制
    industry_provider: DataProvider
        记录股票所属行业的数据提供器，可以是NoneDataProvider，表明当前不用对数据进行行业中性化
    group_num: int
        分组数量

    Return
    ------
    out: function
        可供BackTest类作为参数的stock_filter函数
    '''
    def _inner(date, fd_provider):
        st_data = st_provider.get_csdata(date)
        trade_data = tradedata_provider.get_csdata(date)
        factor_data = fd_provider.get_csdata(date)
        stockpool_data = stockpool_provider.get_csdata(date)
        industry_data = industry_provider.get_csdata(date)
        data = pd.DataFrame({'data': factor_data, 'st_data': st_data, 'trade_data': trade_data})
        if stockpool_data is not None:    # 表示当前有股票池的限制
            data = data.assign(stockpool=stockpool_data)
        else:
            data = data.assign(stockpool=[1] * len(data))
        if industry_data is not None:   # 表明当前要求数据进行行业中性化
            data = data.assign(industry=industry_data)
            data = data.loc[data.industry != NaS]
            # data['data'] = data.groupby('industry').data.transform(lambda x: x - x.mean())
        else:
            data = data.assign(industry=[NaS] * len(data))
        # pdb.set_trace()
        data = data.loc[(data.trade_data == 1) & (data.st_data == 0) & (data.stockpool == 1), :].\
            dropna(subset=['data'], axis=0)
        by_ind = data.groupby('industry')
        data = data.assign(datag=by_ind.data.transform(lambda x: pd.qcut(x, group_num,
                                                                         labels=range(group_num))))
        by_group_id = data.groupby('datag')
        out = {g: by_group_id.get_group(g).index.tolist()
               for g in by_group_id.groups}
        return out
    return _inner


def transholding(holding):
    '''
    因为回测实例中记录的持仓数据的格式为{time: {port_id: list or dict}}，需要将其转换为
    {port_id: {time: dict or list}}

    Parameter
    ---------
    holding: dict
        结构为{time: {port_id: list or dict}}

    Return
    ------
    out: dict
        结构为{port_id: {time: dict or list}}
    '''
    tmp_df = pd.DataFrame(holding)
    tmp_df = tmp_df.T.to_dict()
    return tmp_df


def holding2stockpool(holding, port_id, universe=None):
    '''
    将给定编号的持仓字典数据转换为pd.DataFrame形式的数据

    Parameter
    ---------
    holding: dict
        回测返回的持仓列表，结构为{time: {port_id: list}}

    port_id: int
        要求必须为有效的组合号码
    universe: iterable, default None
        最新的（指与因子数据同步的）universe，默认为None表示直接从存储universe的数据文件中获取

    Return
    ------
    out: pd.DataFrame
        其中，属于该持仓中的股票对应的值为1，不属于的为0。数据的索引上的时间与持仓的换仓时间（一般为
        计算日）相同
    '''
    if universe is None:
        universe = get_universe()
    universe = sorted(universe)
    holding = transholding(holding)
    holding = holding[port_id]
    out = dict()
    for t in holding:
        tmph = holding[t]
        tmps = pd.Series(np.ones((len(tmph), )), index=tmph)
        tmps = tmps.reindex(universe).fillna(0)
        out[t] = tmps
    out = pd.DataFrame(out).T
    return out

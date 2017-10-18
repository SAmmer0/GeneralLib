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


class Instrument(object, metaclass=ABCMeta):
    '''
    证券基类 Instrument(secu_code, num)
    '''

    def __init__(self, secu_code, num=0, quote_provider=None):
        '''
        Parameter
        ---------
        secu_code: str
            证券代码
        num: float, default 0
            证券数量
        quote_provider: DataProvider, default None
            用于更新行情数据
        '''
        self.code = secu_code
        self.num = num
        self._quote_provider = quote_provider
        self.price = None

    def increase_num(self, num):
        '''
        增加给定数量的证券

        Parameter
        ---------
        num: float
            增加数量的值，要求为正数

        Return
        ------
        out: self
        '''
        assert num > 0, ValueError('Number increased should be positive, you provide {num}'
                                   .format(num=num))
        self.num += num
        return self

    def decrease_num(self, num):
        '''
        减少给定数量的证券

        Parameter
        ---------
        num: float
            减少数量的值，范围为(0, self.num]

        Return
        ------
        out: self
        '''
        assert num > 0 and num <= self.num,\
            ValueError('Number decreased should between 0 and {hn}, you provide {num}'.
                       format(hn=self.num, num=num))
        self.num -= num
        return self

    def adjust_to(self, num):
        '''
        将证券的数量调整到指定的数量

        Parameter
        ---------
        num: float
            新的证券数量，要求必须为正

        Return
        ------
        out: self
        '''
        assert num > 0, ValueError('Number should be positive, you provide {num}'.format(num=num))
        self.num = num
        return self

    @abstractmethod
    def refresh_price(self, date):
        '''
        刷新当前证券的价格

        Parameter
        ---------
        date: datetime like
            价格的刷新时间

        Return
        ------
        out: float
            当前最新价格
        '''
        pass

    def refresh_value(self, date):
        '''
        刷新当前证券的价值

        Parameter
        ---------
        date: datetime like
            刷新的时间

        Return
        ------
        out: float
            当前持有证券的价值
        '''
        self.refresh_price(date)
        return self.num * self.price

    def copy(self):
        return Instrument(self.code, self.num, self._quote_provider)

    def __eq__(self, other):
        return self.code == other.code

    def __hash__(self):
        return hash(self.code)

    def __repr__(self):
        return 'Instrument(code={code}, num={num})'.format(code=self.code, num=self.num)

    def __str__(self):
        return self.__repr__()


class Cash(Instrument):
    '''
    现金类
    '''

    def refresh_price(self, date):
        '''
        现金的价格始终为1
        '''
        self.price = 1
        return self.price

    def copy(self):
        return Cash(self.num)

    def __init__(self, num):
        super().__init__(CASH, num)

    def __repr__(self):
        return 'Cash(num={num})'.format(num=self.num)


class Stock(Instrument):
    '''
    股票类
    '''

    def __init__(self, secu_code, num=0, quote_provider=None):
        '''
        Parameter
        ---------
        secu_code: str
            证券代码
        num: float, default 0
            证券数量
        quote_provider: DataProvider, default None
            用于更新行情数据
        '''
        super().__init__(secu_code, num, quote_provider)
        self._last_refresh_time = None

    def refresh_price(self, date):
        '''
        刷新当前证券的价格

        Parameter
        ---------
        date: datetime like
            价格的刷新时间

        Return
        ------
        out: float
            当前最新价格

        Notes
        -----
        股票有可能碰到退市的情况，导致后面的价格为NaN，如果碰到这种情况，将沿用前一个非NaN的价格
        隐含的假设就是能够在退市前最后一个交易日卖出
        '''
        if self._last_refresh_time is None or self._last_refresh_time != pd.to_datetime(date):
            # 表示当前没有价格缓存或者缓存已经过时
            tmp = self._quote_provider.get_data(date, self.code)
            if not pd.isnull(tmp):  # 当前获取的行情有效
                self.price = tmp
            self._last_refresh_time = pd.to_datetime(date)
        return self.price

    def copy(self):
        res = Stock(self.code, self.num, self._quote_provider)
        res._last_refresh_time = self._last_refresh_time
        res.price = self.price
        return res

    def __repr__(self):
        return 'Stock(code={code}, num={num})'.format(code=self.code, num=self.num)


class Portfolio(object):
    '''
    资产组合类
    '''

    def __init__(self, ini_cap=0, residual_cash=100):
        '''
        Parameter
        ---------
        init_cap: float
            初始资金
        residual_cash: float, default 100
            留存资金
        '''
        self.position = {CASH: Cash(ini_cap)}
        # self._quote_provider = quote_provider
        self._last_refresh_time = None
        self._residual_cash = residual_cash

    # def refresh_quote(self, date):
    #     '''
    #     刷新行情数据

    #     Parameter
    #     ---------
    #     date: datetime like
    #         数据更新的时间
    #     '''
    #     date = pd.to_datetime(date)
    #     if date != self._last_refresh_time:  # 需要调用新的数据
    #         self._quote_cache = self._quote_provider.get_csdata(date)
    #         self._last_refresh_time = date

    # def get_quote(self, code, date):
    #     '''
    #     获取给定证券的行情数据

    #     Parameter
    #     ---------
    #     code: str
    #         证券代码
    #     date: datetime like
    #         获取数据的时间

    #     Return
    #     ------
    #     out: float
    #         给定证券的价格
    #     '''
    #     self.refresh_quote(date)
    #     return self._quote_cache.loc[code]

    def sell_instrmt(self, instrmt, num, date, transaction_cost=0):
        '''
        组合执行卖出证券的操作，将会在对应的证券中减少对应的数量，如果证券数量接近于0，则直接从
        组合中剔除，并将卖出后收到的现金（扣除交易费用）加到现金中

        Parameter
        ---------
        instrmt: str or Instrument
            需要卖出的证券
        num: float
            卖出证券的数量，要求为正数
        date: datetime like
            卖出证券的时间
        transaction_cost: float
            卖出证券的交易成本占比，有效范围为[0, 1)
        '''
        assert num > 0, ValueError('Sell number should be positive, you provide {num}'.
                                   format(num=num))
        assert transaction_cost < 1 and transaction_cost >= 0, \
            ValueError('The range of transaction cost is [0, 1), you provider {tc}'.
                       format(tc=transaction_cost))
        if isinstance(instrmt, str):
            assert instrmt in self.position, ValueError('Invalid instrument \'{name}\''
                                                        .format(name=instrmt))
            instrmt = self.position[instrmt]
        else:
            instrmt = self.position[instrmt.code]
        assert instrmt.code != CASH, ValueError('Cash cannot be sold!')
        assert num <= instrmt.num, \
            ValueError('Maximum number can be sold is {mnum}, you provide {num}'.
                       format(mnum=instrmt.num, num=num))
        instrmt_price = instrmt.refresh_price(date)
        sell_value = instrmt_price * num * (1 - transaction_cost)   # 计算卖出证券的价值（剔除交易成本后）
        instrmt.decrease_num(num)
        if isclose(instrmt.num, 0):     # 证券数量接近于0，从持仓中剔除
            del self.position[instrmt.code]
        self.position[CASH].increase_num(sell_value)

    def buy_instrmt(self, instrmt, date, transaction_cost=0):
        '''
        组合执行买入证券的操作

        Parameter
        ---------
        instrmt: Instrument
            需要买入的证券
        date: datetime like
            买入证券的时间
        transaction_cost: float
            买入证券的交易成本占比，有效范围为[0, 1)
        '''
        assert instrmt.num > 0, ValueError('Buy number should be positive, you provide {num}'.
                                           format(num=instrmt.num))
        assert transaction_cost < 1 and transaction_cost >= 0, \
            ValueError('The range of transaction cost is [0, 1), you provider {tc}'.
                       format(tc=transaction_cost))
        assert instrmt.code != CASH, ValueError('Cash cannot be bought!')
        buy_value = instrmt.refresh_value(date) * (1 + transaction_cost)
        assert buy_value <= self.position[CASH].num, \
            'Error, Value of instrument to be bought exceeds residual cash!'
        # 将买入的证券添加到组合中
        if instrmt.code in self.position:
            self.position[instrmt.code].increase_num(instrmt.num)
        else:
            self.position[instrmt.code] = instrmt
        # 减少现金
        self.position[CASH].decrease_num(buy_value)

    def rebalance2targetweight(self, target_weight, date, buy_cost=0, sell_cost=0):
        '''
        将组合的持仓调整到指定的证券和指定的权重

        Parameter
        ---------
        target_holding: dict
            目标持仓，格式为{Instrument: weight}
        date: datetime like
            调仓的时间
        buy_cost: float
            买入证券的交易成本
        sell_cost: float
            卖出证券的交易成本
        '''
        valid_money = self.refresh_value(date) * (1 - sell_cost) / (1 + buy_cost) - \
            self._residual_cash     # 可使用的有效资金要去除潜在的交易成本和留存现金
        target_holding = {}
        for instrmt in target_weight:
            price = instrmt.refresh_price(date)
            num = valid_money * target_weight[instrmt] / price
            try:
                target_holding[instrmt.code] = instrmt.adjust_to(num)
            except AssertionError:
                pdb.set_trace()
        self._rebalance2targetnum(target_holding, date, buy_cost, sell_cost)

    def _rebalance2targetnum(self, target_holding, date, buy_cost=0, sell_cost=0):
        '''
        将组合持仓调整到指定的证券和指定的持仓量

        Parameter
        ---------
        target_holding: dict
            目标持仓，格式为{code: Instrument}
        date: datetime like
            调仓的时间
        buy_cost: float
            买入证券的交易成本
        sell_cost: float
            卖出证券的交易成本
        '''
        trading_task = self._calcu_diff(target_holding)
        buy_task = trading_task['buy']
        sell_task = trading_task['sell']
        # pdb.set_trace()
        for code in sell_task:
            self.sell_instrmt(code, sell_task[code], date, sell_cost)
        for code in buy_task:
            self.buy_instrmt(buy_task[code], date, buy_cost)

    def _calcu_diff(self, target_holding):
        '''
        计算目标持仓当前持仓的差别

        Parameter
        ---------
        target_holding: dict
            目标持仓，格式为{secu_code: Instrument}

        Return
        ------
        out: dict
            持仓差别的计算结果，格式为{'sell': {code: num}, 'buy': {code: num}}，其中sell表示需要
            卖出的证券，buy表示需要买入的证券
        '''
        target_codes = set(target_holding.keys())
        holding_codes = set(self.position.keys()).difference([CASH])
        sell_codes = holding_codes.difference(target_codes)
        buy_codes = target_codes.difference(holding_codes)
        adjust_codes = holding_codes.intersection(target_codes)

        tobe_sold = {code: self.position[code].num for code in sell_codes}
        tobe_bought = {code: target_holding[code].copy() for code in buy_codes}
        for code in adjust_codes:
            diff = target_holding[code].num - self.position[code].num
            if isclose(diff, 0):
                continue
            if diff > 0:
                tobe_bought[code] = target_holding[code].copy().adjust_to(diff)
            else:
                tobe_sold[code] = -diff
        return {'buy': tobe_bought, 'sell': tobe_sold}

    def refresh_value(self, date):
        '''
        计算当前组合的总值

        Parameter
        ---------
        date: datetime like
            计算总值的时间

        Return
        ------
        out: float
            当前组合总值
        '''
        value = 0
        for instrmt in self.position.values():
            value += instrmt.refresh_value(date)
        return value


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

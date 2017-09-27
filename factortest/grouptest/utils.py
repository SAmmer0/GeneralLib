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
    金融工具基类类，用来代表当前持有的金融工具
    '''

    def __init__(self, code, num=None, quote_provider=None, allow_short=False):
        '''
        Parameter
        ---------
        code: str, default None
            金融工具的代码，可以为None
        num: double or float or int, default None
            持有的金融工具的数量
        quote_provider: DataProvider
            用于提供相关行情数据
        allow_short: bool, default False
            是否允许卖空，默认为False，即不允许
        '''
        self.code = code
        self._num = num
        self.unit_price = None  # 单位价格
        self.quote_provider = quote_provider  # 行情数据提供器
        self.allow_short = allow_short  # 是否允许卖空

    @abstractmethod
    def refresh_price(self, date):
        '''
        更新该工具的最新价格

        Parameter
        ---------
        date: str or other types that can be converted by pd.to_datetime
            更新价格的时间
        '''
        pass

    def refresh_value(self, date):
        '''
        计算该金融工具的价值（市值）

        Parameter
        ---------
        date: str or other types that can be converted by pd.to_datetime
            计算价值（市值）的时间

        Return
        ------
        out: float
            当前金融工具的价值（市值）
        '''
        self.refresh_price(date)
        return self._num * self.unit_price

    def increase_num(self, num):
        '''
        增加持有的金融工具的数量

        Parameter
        ---------
        num: int or float
            增加的金融工具的数量，要求为正数
        '''
        assert num > 0, 'Error, added number({num}) should be positive'.format(num=num)
        self._num += num

    def descrease_num(self, num):
        '''
        减少持有的金融工具的数量

        Parameter
        ---------
        num: int or float
            减少的金融工具的数量，要求为负数

        Notes
        -----
        减少前会检查数量是否足够，如果不能卖空，则最大的减少数量为当前持有的数量
        '''
        assert num > 0, 'Error, descrease number({num}) should be positive'.format(num=num)
        if not self.allow_short:
            assert num <= self._num,\
                'Short is not allowed, ' + \
                'you cannot descrease instrument number below ZERO' + \
                '(max_num={mm}, you provide={yp})'.format(mm=self._num, yp=num)
        self._num -= num

    def sell(self, num, date):
        '''
        模拟金融工具卖出的过程

        Parameter
        ---------
        num: int or float
            需要卖出的数量，必须为正数
        date: str, datetime or other compatible type
            卖出的日期

        Return
        ------
        out: float
            卖出的金融工具的价值（市值）
        '''
        self.refresh_price(date)
        sell_value = self.unit_price * num
        self.descrease_num(num)
        return sell_value

    def construct_from_value(self, value, date):
        '''
        按照价值构造该金融工具

        Parameter
        ---------
        value: float
            需要买入的市值
        date: datetime or other compatible types
            买入的时间

        Return
        ------
        out: Instrument
            将本实例返回
        '''
        self.refresh_price(date)
        if self.unit_price is None:
            pdb.set_trace()
        self._num = value / self.unit_price
        return self

    @property
    def num(self):
        return self._num

    def copy(self):
        '''
        返回一个金融工具的拷贝
        '''
        return Instrument(code=self.code, num=self._num, quote_provider=self.quote_provider)

    def __repr__(self):
        return 'Instrument(Code={code}, Number:={num})'.format(code=self.code, num=self.num)

    def __str__(self):
        return self.__repr__()


class Cash(Instrument):
    '''
    现金类
    '''

    def __init__(self, num=None):
        '''
        Parameter
        ---------
        num: double or float or int, default None
            持有的现金的数量
        '''
        super().__init__(code=CASH, num=num)
        self.unit_price = 1

    def refresh_price(self, date):
        '''
        现金的价格永远都是1
        '''
        pass


class Stock(Instrument):
    '''
    股票类
    '''

    def __init__(self, code, num=None, quote_provider=None, allow_short=False):
        '''
        Parameter
        ---------
        code: str
            股票代码
        num: double or float or int, default None
            持有的股票的数量
        quote_provider: DataProvider
            用于提供行情数据
        '''
        super().__init__(code, num, quote_provider, allow_short)
        self._last_refresh_time = None  # 用于记录上次价格刷新的时间，判断是否需要从行情数据提供器来获取数据

    def refresh_price(self, date):
        '''
        从行情提供器中获取给定时间的股票价格
        Parameter
        ---------
        date: str or other types that can be converted by pd.to_datetime
            计算价值（市值）的时间
        '''
        if self._last_refresh_time is None or self._last_refresh_time != pd.to_datetime(date):
            # 表示当前缓存已经过时或者没有缓存
            tmp = self.quote_provider.get_data(date, self.code)
            if not pd.isnull(tmp):  # 只有价格不为NA时才更新价格，否则沿用之前的价格（为了处理退市的情况）
                self.unit_price = tmp
            self._last_refresh_time = pd.to_datetime(date)


class Portfolio(object):
    '''
    资产组合类，是金融工具的一个集合
    '''

    def __init__(self, cash=0):
        self.positions = {CASH: Cash(cash)}

    def add_instrument(self, instrmt):
        '''
        向组合中添加金融工具

        Parameter
        ---------
        instrmt: Instrument or its subclass
            需要被加入到组合的金融工具
        '''
        assert not self.iscontained(instrmt.code), \
            "Error, instrument({code}) already exists".format(code=instrmt.code)
        self.positions[instrmt.code] = instrmt

    def remove_instrument(self, instrmt_name):
        '''
        根据提供的金融工具的名称，从持仓中剔除该金融工具

        Parameter
        ---------
        instrmt_name: str
            需要被剔除的金融工具的名称
        '''
        assert self.iscontained(instrmt_name), \
            'Error, instrument({code}) is not contained in porfolio'.format(code=instrmt_name)
        assert instrmt_name != CASH, 'Error, cash item can not be removed'
        del self.positions[instrmt_name]

    def iscontained(self, instrmt_name):
        '''
        检查某个名称的金融工具是否在持仓中

        Parameter
        ---------
        instrmt_name: str
            需要检测的金融工具名
        '''
        return instrmt_name in self.positions

    def sell_instrument(self, instrmt_name, date, num):
        '''
        卖出给定的金融工具，并将收到的钱添加到现金中

        Parameter
        ---------
        instrmt_name: str
            需要卖出的金融工具名称
        date: str, datetime or other compatible type
            卖出的日期
        num: int or float
            卖出的数量
        '''
        assert instrmt_name != CASH, 'Error, cash cannot be selled'
        assert self.iscontained(instrmt_name), \
            'Error, instrument({name}) is not contained in the portfolio'.format(name=instrmt_name)
        instrmt = self.positions[instrmt_name]
        tmp_cash = instrmt.sell(num, date)
        cash = self.positions[CASH]
        cash.increase_num(tmp_cash)
        if isclose(instrmt.num, 0, abs_tol=1e-5):  # 当前持仓接近于0
            self.remove_instrument(instrmt_name)

    def buy_instrument(self, instrmt, date):
        '''
        买入给定的金融工具

        Parameter
        ---------
        instrmt: Instrument
            需要买入的金融工具
        date: str, datetime or other compatible type
            买入的日期
        '''
        assert instrmt.code != CASH, 'Error, cash cannot be traded'
        instrumt_value = instrmt.refresh_value(date)
        cash_value = self.positions[CASH].refresh_value(date)
        assert instrumt_value <= cash_value, 'Error, cash is not enough'
        # pdb.set_trace()
        self.positions[CASH].descrease_num(instrumt_value)
        self.add_instrument(instrmt)

    def refresh_value(self, date):
        '''
        计算当前资产组合的价值（市值）

        Parameter
        ---------
        date: str, datetime or other compatible type
            计算组合市值的时间

        Return
        ------
        out: float
            组合包含的所有金融工具的价值（市值）
        '''
        value = 0
        for instrmt in self.positions:
            value += self.positions[instrmt].refresh_value(date)
        return value

    def to_list(self):
        '''
        将Portfolio转换为证券代码列表，列表中只包含非现金项目

        Return
        ------
        out: list
            当前组合中的证券代码列表
        '''
        return list(self.positions.keys())

    def sell_all(self, date):
        '''
        将组合中的所有金融工具卖出

        Parameter
        ---------
        date: datetime or other compatible types
            卖出金融工具的时间

        Return
        ------
        out: float
            当前组合的总价值（市值）
        '''
        pos_codes = sorted(self.positions.keys())
        for pos_code in pos_codes:
            if pos_code != CASH:
                self.sell_instrument(pos_code, date, self.positions[pos_code].num)
        return self.refresh_value(date)

    def buy_seculist(self, secu_list, date):
        '''
        买入证券列表中所有的证券

        Parameter
        ---------
        secu_list: list like
            证券列表，内容为[Instrument, ...]
        date: datetime or compatible types
            买入的时间
        '''
        for secu in secu_list:
            self.buy_instrument(secu, date)

    def __repr__(self):
        res = list()
        for instrumt in self.positions.values():
            res.append(str(instrumt))
        return str(res)

    def __str__(self):
        return self.__repr__()


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
            data['data'] = data.groupby('industry').data.transform(lambda x: x - x.mean())
        # pdb.set_trace()
        data = data.loc[(data.trade_data == 1) & (data.st_data == 0) & (data.stockpool == 1), :].\
            dropna(subset=['data'], axis=0)
        data = data.assign(datag=pd.qcut(data.data, group_num, labels=range(group_num)))
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

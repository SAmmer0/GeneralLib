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
from abc import ABCMeta, abstractclassmethod
# 第三方库
import pandas as pd
from datatoolkits import isclose
# ------------------------------------------------------------------------------
# 常量定义
CASH = 'Cash'

# ------------------------------------------------------------------------------
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

    @abstractclassmethod
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
            assert num < self._num,\
                ('Short is not allowed, ',
                 'you cannot descrease instrument number below ZERO',
                 '(max_num={mm}, you provide={yp})'.format(mm=self._num, yp=num))
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

    @property
    def num(self):
        return self._num

    def copy(self):
        '''
        返回一个金融工具的拷贝
        '''
        return Instrument(code=self.code, num=self._num, quote_provider=self.quote_provider)


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
            self.unit_price = self.quote_provider.get_data(self.code, date)
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
        cash.inscrease_num(tmp_cash)
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
        self.positions[CASH].descrease_num(cash_value)
        self.add_instrument(instrmt)

    def refresh_value(self, date):
        value = 0
        for instrmt in self.positions:
            value += self.positions[instrmt].refresh_value(date)
        return value

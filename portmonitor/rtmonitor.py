#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-17 15:01:53
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
利用从tushare获取的行情数据对组合进行实时监控
'''
from pdb import set_trace
from collections import namedtuple
from time import sleep, time
from abc import ABCMeta, abstractclassmethod

import pandas as pd
from tushare import get_realtime_quotes
import numpy as np

from datatoolkits import drop_suffix
from portmonitor.const import CASH, MONING_START, MONING_END, NOON_START, NOON_END

# --------------------------------------------------------------------------------------------------
# 实时行情刷新类
RTData = namedtuple('RTData', ['time', 'data'])


class PortfolioRefresher(object):
    '''
    对tushare接口进行包装，提供接口返回给定组合的实时走势
    '''

    def __init__(self, port_data, standardlize=True):
        '''
        Parameter
        ---------
        port_data: portmonitor.manager.PortfolioData
            已经经过持仓更新的持仓数据
        standardlize: boolean, default True
            是否需要对组合的价值进行归一，如果进行归一化处理，每次刷新返回的值都是组合净值变动
        '''
        if standardlize:
            self._port_basevalue = port_data.last_asset_value
        else:
            self._port_basevalue = 1
        self._port_holding = pd.Series({drop_suffix(c): n for c, n in port_data.curholding.items()})
        self.id_ = port_data.id
        self._data = []

    def refresh(self):
        '''
        从tushare获取最新的行情数据，并计算最新的组合价值

        Return
        ------
        out: float
            返回当前组合的最新价值（净值）
        '''
        quote = get_realtime_quotes(self._port_holding.index.tolist())
        quote = quote.set_index('code').price.apply(np.float)
        quote[CASH] = 1
        # set_trace()
        port_value = self._port_holding.dot(quote) / self._port_basevalue
        return port_value

    def __call__(self):
        '''
        生成器，根据给定的频率定时刷新数据，并返回，返回的数据包含时间和组合价值
        '''
        while True:
            t = time()
            data = self.refresh()
            self._data.append(RTData(time=t, data=data))
            yield t, data


# --------------------------------------------------------------------------------------------------
# 数据展示类


class Displayer(object, metaclass=ABCMeta):
    '''
    用于实时展示数据
    '''

    def __init__(self, rt_data_source, rest_checker=None, freq=2):
        '''
        Parameter
        ---------
        rt_data_source: PortfolioRefresh
            实时数据更新器，要求为PortfolioRefresh类型，可以通过__call__返回迭代器，该迭代器可以返回
            需要展示的实时数据，数据为RTData类型
            （其实这个地方可以要求参数为PortfolioRefresh列表，用于实时显示多个数据）
            （之所以要使用PortfolioRefresh是为了获取对应数据的ID）
        rest_checker: callable
            用于检查当前是否是休市期间（后续将建立一个RestChecker类型，用于检查）
        freq: float, default 2
            数据更新频率，默认为2秒
        '''
        self._data_source = rt_data_source
        self._rest_checker = rest_checker
        self._freq = freq

    @abstractclassmethod
    def show(self):
        '''
        核心函数，用于展示实时数据
        '''
        pass


class PrintDisplayer(Displayer):
    def show(self):
        pass


if __name__ == '__main__':
    from portmonitor import MonitorManager
    monitor = MonitorManager(show_progress=False)
    monitor.update_all()
    refresher = PortfolioRefresher(monitor['SMALL_CAP'])

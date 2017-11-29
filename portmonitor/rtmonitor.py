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
from time import sleep
from abc import ABCMeta, abstractclassmethod
import datetime as dt

import pandas as pd
from tushare import get_realtime_quotes
import numpy as np


from datatoolkits import drop_suffix
from portmonitor.const import CASH

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
            t = dt.datetime.now()
            data = self.refresh()
            self._data.append(RTData(time=t, data=data))
            yield t, data

# --------------------------------------------------------------------------------------------------
# 休市检测类


class RestChecker(object, metaclass=ABCMeta):
    '''
    用于检测当前时间是否处于休市
    '''

    def __init__(self, periods=None):
        '''
        Parameter
        ---------
        periods: iterable, default None
            元素为tuple(start_time:tuple(hour, minute), end_time:(hour, minute)))，即交易的开始和
            结束时间，默认为None表示使用国内正常的股票交易时间，即9:30-11:30和13:00-15:00
        '''
        if periods is None:
            periods = [((9, 30), (11, 30)), ((13, 0), (15, 0))]
        today = dt.datetime.now().date()

        def format_time(x):
            return dt.datetime.combine(today, dt.time(*x))
        self._periods = [(format_time(t[0]), format_time(t[1])) for t in periods]
        self._index = 0

    def __call__(self):
        '''
        判断当前的时间是否处于某个休市期

        Return
        ------
        is_rest: boolean
            当前是否处于休市期间，如果是则返回True，不是则返回False
        next_open_time: datetime
            如果当前处于休市，则返回下一个休市的时间，如果当前是当天最后休市时间段或者当前没有处于休
            市状态，则返回None
        '''
        now = dt.datetime.now()
        if self._index >= len(self._periods):
            return True, None
        for idx in range(self._index, len(self._periods)):
            cur_period = self._periods[idx]
            if now < cur_period[0]:     # 当前处于当前交易时间段开始前
                out = True, cur_period[0]
                self._index = idx
                break
            if now <= cur_period[1]:    # 当前处于交易时间
                out = False, None
                self._index = idx
                break
        else:
            out = True, None
            self._index = len(self._periods)
        return out

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
            （其实这个地方可以要求参数为PortfolioRefresh列表，用于实时显示多个数据，可以考虑使用继承
            的方法来对其进行修改）
            （之所以要使用PortfolioRefresh是为了获取对应数据的ID）
        rest_checker: RestChecker
            用于检查当前是否是休市期间
        freq: float, default 2
            数据更新频率，默认为2秒
        '''
        self._data_source = rt_data_source
        if rest_checker is None:
            rest_checker = RestChecker()
        self._rest_checker = rest_checker
        self._freq = freq
        self._id = rt_data_source.id_

    @abstractclassmethod
    def show(self):
        '''
        核心函数，用于展示实时数据
        '''
        pass

    def _rest_check(self):
        '''
        检查当前是否休市，如果休市则自动设置休眠，如果整天交易已经结束，会raise KeyboardInterrupt
        '''
        sleep_gap = 2   # 两个交易时间段之间休市时，按照分段时间进行休眠，防止无法中途停止
        is_resting, to_time = self._rest_checker()
        if is_resting:  # 当前处于休市时间
            if to_time is None:  # 表示当天全部交易已经结束
                print("当天已休市")
                raise KeyboardInterrupt
            # 交易期间休市
            now = dt.datetime.now()
            print("日内休市")
            while True:
                now = dt.datetime.now()
                if now >= to_time:
                    return
                sleep(sleep_gap)
        else:   # 防止数据更新过于频繁
            sleep(self._freq)


class PrintDisplayer(Displayer):
    '''
    以打印的方式显示当前数据
    '''

    def show(self):
        while True:
            try:
                data_time, data = next(self._data_source())
                print(data_time.strftime('%H:%M:%S'), " ", self._id)
                print('{:.2%}'.format(data - 1))
                self._rest_check()
            except KeyboardInterrupt:
                return


class MultiPrintDisplayer(Displayer):
    '''
    多数据显示
    '''

    def __init__(self, rt_data_sources, rest_checker=None, freq=2):
        '''
        Parameter
        ---------
        rt_data_sources: dict
            key为refresher的ID，value为PortfolioRefresh类型
        rest_checker: RestChecker
            用于检查当前是否处于休市期间
        freq: float, default 2
            数据更新频率
        '''
        self._data_sources = rt_data_sources
        if rest_checker is None:
            rest_checker = RestChecker()
        self._rest_checker = rest_checker
        self._freq = freq

    def show(self):
        '''
        显示多个组合的实时数据
        '''
        from tabulate import tabulate
        table = []
        header = ['Portfolio', 'Time', 'Chg']
        data_sources = self._data_sources
        while True:
            try:
                for port_id in sorted(data_sources.keys()):    # 获取行情数据，并更新表格
                    data_time, data = next(data_sources[port_id]())
                    cur_row = [port_id, data_time.strftime('%H:%M:%S'), '{:.2%}'.format(data - 1)]
                    table.append(cur_row)
                print(tabulate(table, headers=header))
                print('\n')
                table = []
                self._rest_check()
            except KeyboardInterrupt:
                return


class GraphDisplayer(Displayer):
    '''
    以动态图的形式显示当前数据
    '''

    def show(self):
        pass


if __name__ == '__main__':
    from portmonitor import MonitorManager
    monitor = MonitorManager(show_progress=False)
    monitor.update_all()
    refreshers = {port_id: PortfolioRefresher(monitor[port_id])
                  for port_id in monitor}
    displayer = MultiPrintDisplayer(refreshers)
    displayer.show()

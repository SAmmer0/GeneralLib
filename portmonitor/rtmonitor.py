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
import datetime as dt
from time import sleep
from abc import ABCMeta, abstractclassmethod

import pandas as pd
from tushare import get_realtime_quotes
import numpy as np

from datatoolkits import drop_suffix
from portmonitor.const import CASH

# --------------------------------------------------------------------------------------------------
# 实时行情刷新类


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

    def refresh(self):
        '''
        从tushare获取最新的行情数据，并计算最新的组合价值
        '''
        quote = get_realtime_quotes(self._port_holding.index.tolist())
        quote = quote.set_index('code').price.apply(np.float)
        quote[CASH] = 1
        # set_trace()
        port_value = self._port_holding.dot(quote) / self._port_basevalue
        return port_value


# --------------------------------------------------------------------------------------------------
# 实时监控类
RTData = namedtuple('RTData', ['time', 'data'])


class RTMonitor(object):
    '''
    实时监控管理类，添加需要实时监控的实例，并对实例进行统一更新、展示
    '''

    def __init__(self, monitor_manager, displayer, freq=2):
        '''
        Parameter
        ---------
        monitor_manager:  portmonitor.manager.MonitorManager
            包含需要被监控的组合的组合管理器
        displayer: Displayer
            用于展示数据的方法，要求必须要有show方法，传入参数为RTMonitor对象
        freq: int, default 2
            实时更新的间隔（秒），要求为正整数
        '''
        self._moni_target = {port_id: PortfolioRefresher(monitor_manager[port_id])
                             for port_id in monitor_manager}
        self._displayer = displayer
        self._freq = freq
        self.rtdata = list()    # 存储实时的组合数据，格式为[RTData(time, {port_id: nav}), ...]

    def update(self):
        '''
        定时更新时触发
        '''
        cur_time = dt.datetime.now().time
        rt_nav = {}
        for port_id in self._moni_target:
            rt_nav[port_id] = self._moni_target[port_id].refresh()
        self.rtdata.append(RTData(cur_time, rt_nav))

    def start(self):
        '''
        启动监控
        '''
        while True:
            try:
                self.update()
                self._displayer.show(self)
                sleep(self._freq)
            except KeyboardInterrupt:
                return

# --------------------------------------------------------------------------------------------------
# 展示类


class Displayer(object, metaclass=ABCMeta):
    '''
    显示类的基类
    '''

    def __init__(self):
        pass

    @abstractclassmethod
    def show(self, rt_moni):
        '''
        展示相关数据

        Parameter
        ---------
        rt_moni: RTMonitor
            实时监控器的实例
        '''
        pass


class PrintLatestDisplayer(Displayer):
    '''
    采用打印的方式展示数据
    '''

    def __init__(self):
        from pprint import pprint
        self._printer = pprint

    def show(self, rt_moni):
        self._printer(rt_moni.rtdata[-1])


if __name__ == '__main__':
    from portmonitor import MonitorManager
    monitor = MonitorManager(show_progress=False)
    monitor.update_all()
    rtmonitor = RTMonitor(monitor, PrintLatestDisplayer())
    rtmonitor.start()

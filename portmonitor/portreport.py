#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-26 10:04:22
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
报告组合近期的表现，目前包含近一个交易日、近一周、近一个月的收益，以及近一个周、近一个月的日波动率和
最大回撤
'''
# 第三方模块
from numpy import nan
import pandas as pd
# 本地模块
from dateshandle import tds_shift, get_tds
from factortest.utils import MonRebCalcu, WeekRebCalcu
from portmonitor.const import WEEKLY, MONTHLY, DAILY
from report import max_drawn_down

# --------------------------------------------------------------------------------------------------
# 函数


def get_last_date(date, freq):
    '''
    获取给定日期和给定报告频率的最近一个起始日期

    Parameter
    ---------
    date: datetime like
        当前日期
    freq: str
        给定的频率，只能为[DAILY, WEEKLY, MONTHLY]之一

    Return
    ------
    out: pd.TimeStamp

    Notes
    -----
    该函数的主要功能是查找距离给定日期最近的，并且与报告频率相关的日期，用来计算这个日期到给定日期
    的相关数据
    这个日期的计算方法如下：
    若freq为DAILY，即日频，则直接返回date上一个交易日的日期
    若freq为WEEKLY，即周频，则返回date上一个周的最后一个交易日的日期
    若freq为MONTHLY，即月频，则返回date上一个月的最后一个交易日的日期
    '''
    start_time = tds_shift(date, 30)
    if freq == DAILY:
        tds = get_tds(start_time, date)
        return tds[-2]
    if freq == WEEKLY:
        reb = WeekRebCalcu(start_time, date)
    elif freq == MONTHLY:
        reb = MonRebCalcu(start_time, date)
    else:
        raise ValueError('Unknown \'freq\' parameter({param})'.format(param=freq))
    return reb.reb_points[-2]


def parse_report(rpt):
    '''
    将Report类中的结果转换为pd.DataFrame

    Parameter
    ---------
    rpt: Report
        需要被解析的rpt对象

    Return
    ------
    out: pd.DataFrame
        列分别为daily、weekly、monthly，行分别为return、vol、mdd
    '''
    res = pd.DataFrame({'daily': rpt.daily, 'weekly': rpt.weekly, 'monthly': rpt.monthly}).T
    return res

# --------------------------------------------------------------------------------------------------
# 类


class Report(object):
    '''
    组合报告类
    '''

    def __init__(self, port_data):
        '''
        Parameter
        ---------
        port_data: portmonitor.manager.PortfolioData
            组合的相关数据
        '''
        self._port_data = port_data
        self._update_time = port_data.update_time
        self._last_time = {f: get_last_date(self._update_time, f) for f in [WEEKLY, MONTHLY]}
        self._result_cache = {}  # 用于记录不同的频率下的结果

    @property
    def daily(self):
        '''
        返回最近一个交易日的收益
        '''
        try:
            out = self._result_cache[DAILY]
        except KeyError:
            ret = self._port_data.assetvalue_ts.pct_change().iloc[-1]
            self._result_cache[DAILY] = {'return': ret, 'vol': nan, 'mdd': nan, 'mdd start': nan,
                                         'mdd end': nan}
            out = self._result_cache[DAILY]
        return out

    def _get_period_report(self, freq):
        '''
        获取给定频率区间的报告，报告内容包含这段时间的收益率、日频波动率和最大回撤

        Parameter
        ---------
        freq: str
            时间频率，只支持[WEEKLY, MONTHLY]

        out: dict
            结果报告，格式为{'return': ret, 'vol': vol, 'mdd': mdd}

        Notes
        -----
        对于一些组合，监控的时间太短（不超过一个周或者一个月），那么返回的结果是由现有的数据计算得来的，
        即比如说本组合是本周一开始监控的，现在是周四，则在计算收益时，是计算的从周一到当前的收益而不是
        上周五到本周四的收益，同样如果数据如果过少不能计算波动率和最大回撤，对应数据为NaN
        '''
        start_time = self._last_time[freq]
        data = self._port_data.assetvalue_ts
        data = data.loc[(data.index >= start_time) & (data.index <= self._update_time)]
        ret = data.iloc[-1] / data.iloc[0] - 1
        vol = data.pct_change().std()
        mdd, mdd_start, mdd_end = max_drawn_down(data)
        out = {'return': ret, 'vol': vol, 'mdd': mdd, 'mdd start': mdd_start, 'mdd end': mdd_end}
        return out

    @property
    def weekly(self):
        '''
        近一个周的组合情况
        '''
        try:
            out = self._result_cache[WEEKLY]
        except KeyError:
            out = self._get_period_report(WEEKLY)
            self._result_cache[WEEKLY] = out
        return out

    @property
    def monthly(self):
        '''
        近一个月组合情况
        '''
        try:
            out = self._result_cache[MONTHLY]
        except KeyError:
            out = self._get_period_report(MONTHLY)
            self._result_cache[MONTHLY] = out
        return out

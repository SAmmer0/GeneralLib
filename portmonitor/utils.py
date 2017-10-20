#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-20 09:57:18
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
用于定义基础的监控模块类（包含所有的监控相关信息）和其他辅助的函数
'''
# 第三方库
from pandas import to_datetime
# 本地库
from factortest.grouptest.utils import MkvWeightCalc, EqlWeightCalc
from factortest.utils import MonRebCalcu, WeekRebCalcu
from factortest.const import TOTALMKV_WEIGHTED, MONTHLY
from portmonitor.const import LONG


class MonitorConfig(object):
    '''
    监控信息类，包含所有的相关监控该信息
    '''

    def __init__(self, stock_filter, add_time, port_id, weight_method=TOTALMKV_WEIGHTED,
                 rebalance_type=MONTHLY, init_cap=1e10, port_type=LONG):
        '''
        Parameter
        ---------
        stock_filter: function
            筛选股票的函数，要求形式为stock_filter(datetime like: date)->list
        add_time: datetime like
            添加的时间
        port_id: str
            组合的id，不能与其他组合重复，是组合的唯一标识
        weight_method: str, default TOTALMKV_WEIGHTED
            加权的方法，目前支持factortest.const中的[EQUAL_WEIGHTED, TOTALMKV_WEIGHTED, FLOATMKV_WEIGHTED]
            对应的为等权、总市值加权、流通市值加权
        rebalance_type: str, default MONTHLY
            换仓日计算方法，目前支持factortet.const中的[MONTHLY, WEEKLY]
            注：换仓的时间是在该日期的下一个交易日
        init_cap: int, default 1e10
            初始的资本金
        port_type: str, default LONG
            组合的类型，目前支持LONG（做多）, SHORT（做空）
        '''
        self.stock_filter = stock_filter
        self.weight_method = weight_method
        self.rebalance_type = rebalance_type
        self.init_cap = init_cap
        self.add_time = to_datetime(add_time)
        self.port_id = port_id
        self.port_type = port_type

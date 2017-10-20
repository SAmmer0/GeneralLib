#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-20 13:56:53
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
组合设置的模板
整个组合设置包含4个参数，分别如下：
股票筛选函数：function(datetime like: date)-> list
    返回的list为股票持仓，即格式为[code1, code2, ...]
股票加权方式：string类型，目前只支持factortest.const中的[EQUAL_WEIGHTED, TOTALMKV_WEIGHTED, FLOATMKV_WEIGHTED]，
    对应的是等权、总市值加权、流通市值加权
换仓频率：string类型，目前只支持factortest.const中的[MONTHLY, WEEKLY]，对应的是月频和周频，月频表示
    按照上个月的最后一个交易日的数据计算组合中的新持仓，然后在本月的第一个交易日（以收盘价）进行换仓
初始资金: int类型，指组合的初始资本
'''
from portmonitor.utils import MonitorConfig


def stock_filter(date):
    return ['000001.SZ', '002230.SZ', '600519.SH']


# 监控配置必须以portfolio命名
portfolio = MonitorConfig(stock_filter, '2017-10-20', 'TEMPLATE')  # 其他采用默认参数

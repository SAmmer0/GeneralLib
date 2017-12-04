#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-01 10:39:09
# @Author  : Hao Li (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
低beta组合
'''
from portmonitor.utils import MonitorConfig, factor_stockfilter_template


stock_filter = factor_stockfilter_template('BETA', group_id=0, group_num=20)


# 监控配置必须以portfolio命名
portfolio = MonitorConfig(stock_filter, '2017-12-01', 'LOW_BETA')  # 其他采用默认参数

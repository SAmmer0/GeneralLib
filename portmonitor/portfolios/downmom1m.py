#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-09 09:08:11
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
低一月动量组合
'''
from portmonitor.utils import MonitorConfig, factor_stockfilter_template


stock_filter = factor_stockfilter_template('MOM_1M', group_id=0)


# 监控配置必须以portfolio命名
portfolio = MonitorConfig(stock_filter, '2017-11-09', 'DOWN_MOM1M')  # 其他采用默认参数

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-25 15:53:16
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
小市值组
'''
from portmonitor.utils import MonitorConfig, factor_stockfilter_template


stock_filter = factor_stockfilter_template('TOTAL_MKTVALUE', group_id=0)


# 监控配置必须以portfolio命名
portfolio = MonitorConfig(stock_filter, '2017-10-20', 'SMALL_CAP')  # 其他采用默认参数

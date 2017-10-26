#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-26 16:29:29
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
低特异波动率组
'''
from portmonitor.utils import MonitorConfig, factor_stockfilter_template


stock_filter = factor_stockfilter_template('SPECIAL_VOL', group_id=0)


# 监控配置必须以portfolio命名
portfolio = MonitorConfig(stock_filter, '2017-10-26', 'SPECIAL_VOL_LOW')  # 其他采用默认参数

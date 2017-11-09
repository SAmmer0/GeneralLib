#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-09 09:12:41
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
低三月动量组合
'''
from portmonitor.utils import MonitorConfig, factor_stockfilter_template


stock_filter = factor_stockfilter_template('MOM_3M', group_id=0)


# 监控配置必须以portfolio命名
portfolio = MonitorConfig(stock_filter, '2017-11-09', 'DOWN_MOM3M')  # 其他采用默认参数

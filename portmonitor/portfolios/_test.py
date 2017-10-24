#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-23 16:19:32
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

from portmonitor.utils import MonitorConfig


def stock_filter(date):
    return ['002852.SZ']


# 监控配置必须以portfolio命名
portfolio = MonitorConfig(stock_filter, '2017-10-20', 'TEST')  # 其他采用默认参数

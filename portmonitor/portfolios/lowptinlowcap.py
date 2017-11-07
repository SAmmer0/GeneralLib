#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-07 09:29:08
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
小市值组中筛选低前景值股票
'''
from portmonitor.utils import MonitorConfig, query_data_bydate
from fmanager import query


def stock_filter(date):
    data = query_data_bydate(date, {'TOTAL_MKTVALUE': 'cap', 'PT_VALUE_1W': 'pt_value',
                                    'ST_TAG': 'st_tag', 'TRADEABLE': 'trade_data'})
    data = data.loc[(data.st_tag == 0) & (data.trade_data == 1)].dropna(subset=['cap', 'pt_value'],
                                                                        axis=0)
    data = data.loc[data.cap <= data.cap.quantile(0.2)]
    data = data.loc[data.pt_value <= data.pt_value.quantile(0.2)]
    out = data.index.tolist()
    return out


# 监控配置必须以portfolio命名
portfolio = MonitorConfig(stock_filter, '2017-11-07', 'LOWPT_IN_SC')  # 其他采用默认参数

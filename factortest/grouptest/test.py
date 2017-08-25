#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-08-21 15:10:07
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

# 第三方库
import pandas as pd
# 本地的库文件
from ..utils import HDFDataProvider
from .utils import Stock, Portfolio, EqlWeightCalc, MonRebCalcu, MkvWeightCalc
from .backtest import Backtest, BacktestConfig

def utils_test():
    quote_provider = HDFDataProvider(r'E:\factordata\basicfactors\quote\CLOSE.h5', '2015-01-01',
                                     '2017-08-01')
    kdxf = Stock('002230.SZ', 1000, quote_provider)
    port = Portfolio(100000)
    
    port.buy_instrument(kdxf, '2017-08-01')
    return port, kdxf

def bt_test():
    start_time = '2013-01-01'
    end_time = '2017-08-20'
    st_provider = HDFDataProvider(r'E:\factordata\basicfactors\general\ST_TAG.h5',
                                  start_time, end_time)
    fmkv_provider = HDFDataProvider(r'E:\factordata\basicfactors\quote\FLOAT_MKTVALUE.h5',
                                    start_time, end_time)
    tradestatus_provider = HDFDataProvider(r'E:\factordata\basicfactors\general\TRADEABLE.h5',
                                           start_time, end_time)
    def mkv_filter(date, st_provider=st_provider, fmkv_provider=fmkv_provider,
                   tradestatus_provider=tradestatus_provider):
        mkv_data = fmkv_provider.get_csdata(date)
        st_data = st_provider.get_csdata(date)
        trade_data = tradestatus_provider.get_csdata(date)
        data = pd.DataFrame({'mkv_data': mkv_data, 'st_data': st_data, 
                             'trade_data': trade_data})
        data = data.loc[(data.trade_data == 1)&(data.st_data == 0), :].dropna(subset=['mkv_data'], axis=0)
        data = data.assign(fmkv_group=pd.qcut(data.mkv_data, 10, labels=range(10)))
        by_group_id = data.groupby('fmkv_group')
        out = {g: by_group_id.get_group(g).index.tolist()
               for g in by_group_id.groups}
        return out
    quote_provider = HDFDataProvider(r'E:\factordata\basicfactors\quote\CLOSE.h5', start_time, end_time)
    conf = BacktestConfig(start_time, end_time, quote_provider, MkvWeightCalc(fmkv_provider), tradestatus_provider,
                          MonRebCalcu(start_time, end_time))
    bt = Backtest(conf, mkv_filter, st_provider=st_provider, fmkv_provider=fmkv_provider, 
                  tradestatus_provider=tradestatus_provider)
    bt.run_bt()
    return bt

def bt_test2():
    start_time = '2013-01-01'
    end_time = '2017-08-20'
    st_provider = HDFDataProvider(r'F:\factordata\basicfactors\general\ST_TAG.h5',
                                  start_time, end_time)
    fmkv_provider = HDFDataProvider(r'F:\factordata\basicfactors\quote\TOTAL_MKTVALUE.h5',
                                    start_time, end_time)
    tradestatus_provider = HDFDataProvider(r'F:\factordata\basicfactors\general\TRADEABLE.h5',
                                           start_time, end_time)
    def mkv_filter(date, st_provider=st_provider, fmkv_provider=fmkv_provider,
                   tradestatus_provider=tradestatus_provider):
        mkv_data = fmkv_provider.get_csdata(date)
        st_data = st_provider.get_csdata(date)
        trade_data = tradestatus_provider.get_csdata(date)
        data = pd.DataFrame({'mkv_data': mkv_data, 'st_data': st_data, 
                             'trade_data': trade_data})
        data = data.loc[(data.trade_data == 1)&(data.st_data == 0), :].dropna(subset=['mkv_data'], axis=0)
        data = data.assign(fmkv_group=pd.qcut(data.mkv_data, 10, labels=range(10)))
        by_group_id = data.groupby('fmkv_group')
        out = {g: by_group_id.get_group(g).index.tolist()
               for g in by_group_id.groups}
        return out
    quote_provider = HDFDataProvider(r'F:\factordata\basicfactors\quote\ADJ_CLOSE.h5', start_time, end_time)
    conf = BacktestConfig(start_time, end_time, quote_provider, EqlWeightCalc(), tradestatus_provider,
                          MonRebCalcu(start_time, end_time))
    bt = Backtest(conf, mkv_filter, st_provider=st_provider, fmkv_provider=fmkv_provider,
                  tradestatus_provider=tradestatus_provider)
    bt.run_bt()
    return bt    
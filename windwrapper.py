#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-02-05 19:50:06
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
本模块将Wind提供的API进行封装，将其函数进行整合，变为经常使用的形式，并以适当的的格式返回

__version__ = 1.0
修改日期：2017年2月5日
修改内容：
    将其他模块中的函数移动到该模块中，目前该模块包含get_data, get_index_constituent, get_tds_wind,
    check_connection

__version__ = 1.1
修改日期：2017-03-30
修改内容：
    在get_data函数中添加选项参数add_stockcode

__version__ = 1.1.1
修改日期：2017-04-21
修改内容：
    为get_tds_wind添加错误代码检测
'''
__version__ = '1.1.1'

try:
    from WindPy import w
except ImportError:
    print('Warning: fail to import WindPy')
import pandas as pd
import datetime as dt
import dateshandle


def check_connection():
    if not w.isconnected():
        w.start()
    return


def get_data(code, field, startDate, endDate, add_stockcode=True, highFreq=False,
             highFreqBarSize='barSize=5', priceAdj=False, time_std=True):
    '''
    从wind中获取数据，可以获取高频率（1天及以上）和低频率（1分钟-1天内）的数据，数据以字典的形式返回
    @param:
        code: 需要获取数据的Wind代码，必须为字符串型
        field: 需要获取数据的类型，可以为列表或者元组或者字符串类型，例如['close', 'open', 'high',
            'low']等价于('close', 'open', 'high', 'low')等价于'close, open, high, low'，常用的数据类型如
            {close: 收盘价，open: 开盘价，high: 最高价，low: 最低价，volume: 成交量}，其他可参考MATLAB中的
            w.menu wsi和wsd来查询
        startDate: 需要数据的开始时间，可以为datetime.datetime类型，也可以是字符串类型，如果是高频率
            的数据，应该是用datetime.datetime类型提供，且提供的时间最好精确到分钟
        endDate: 需要数据的结束时间，与startDate的要求相同
        add_stockcode: 是否在返回的DataFrame中添加股票代码列，默认为添加(True)
        highFreq: 表示是否需要取高频率的数据，bool型
        highFreqBarSize: 提取的高频率数据的频率设定，默认为5分钟线，即barSize=5，其他设置可以类似，
            但是要求必须为字符串，形式只能为barSize=n，n为需要的数据的频率
        priceAdj: 是否需要复权，默认不需要，bool型，只能对有复权选项的证券使用，否则返回的数据会有错误
        time_std: 是否将时间由Wind格式转换为正常格式，默认为True
    @return:
        data: pd.DataFrame格式数据，其中另外会添加time列，记录时间
    '''
    check_connection()
    if highFreq:
        if priceAdj:
            rawData = w.wsi(code, field, startDate, endDate, highFreqBarSize, 'PriceAdj=F')
        else:
            rawData = w.wsi(code, field, startDate, endDate, highFreqBarSize)
    else:
        if priceAdj:
            rawData = w.wsd(code, field, startDate, endDate, 'PriceAdj=F')
        else:
            rawData = w.wsd(code, field, startDate, endDate)
    assert rawData.ErrorCode == 0, rawData.Data[0][0]
    data = dict(zip(field, rawData.Data))
    data['time'] = rawData.Times
    data = pd.DataFrame(data)
    if add_stockcode:
        data['code'] = [code] * len(data)
    if time_std:
        data['time'] = dateshandle.wind_time_standardlization(data.time)
    return data


def get_index_constituent(index, date):
    '''
    用于从Wind中获取指数的成分
    @param:
        index: 需要获取成分的指数代码
        date: 获取指数成分的日期
    @return:
        indexConstituent: 字典，键为成分的代码，值为成分股的中文名称
    '''
    check_connection()
    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, '%Y-%m-%d')
    data = w.wset('sectorconstituent', 'date=' + date.strftime('%Y-%m-%d') + ';windcode=' + index)
    assert data.ErrorCode == 0, data.Data[0][0]
    indexConstituent = dict(zip(data.Data[1], data.Data[2]))
    return indexConstituent


def get_tds_wind(startTime, endTime):
    '''
    从Wind中获取交易日序列
    @param:
        startTime: 交易日的起始日期，要求为dt.datetime格式或者YYYY-MM-DD格式
        endTime: 交易日的终止日期，同上述要求
    @return:
        tds: 交易日列表
    '''
    check_connection()
    tds = w.tdays(startTime, endTime)
    assert tds.ErrorCode == 0, tds.Data[0][0]
    return tds.Data[0]

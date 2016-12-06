#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-12-06 09:33:33
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
本工具用来从Wind中获取需要的数据
__version__ = 1.0
修改日期：2016年12月6日
修改内容：
     初始化，添加基本的获取数据函数get_data
'''


from WindPy import w
import pickle


def get_data(code, field, startDate, endDate, highFreq=False, highFreqBarSize='barSize=5',
             priceAdj=False, write2File=None):
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
        highFreq: 表示是否需要取高频率的数据，bool型
        highFreqBarSize: 提取的高频率数据的频率设定，默认为5分钟线，即barSize=5，其他设置可以类似，
            但是要求必须为字符串，形式只能为barSize=n，n为需要的数据的频率
        priceAdj: 是否需要复权，默认不需要，bool型，只能对有复权选项的证券使用，否则返回的数据会有错误
        write2File: 是否需要把数据写入文件供以后使用方便，默认为None，即不写入，如果需要写入，则需要
            提供写入的文件路径（字符串形式），目前只能把文件写到pickle文件中，方便提取
    @return:
        data: 字典类型数据，形式为{field: data}，其中另外会添加time列，记录时间
    '''
    if not w.isconnected():
        w.start()
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
    data = dict(zip(field, rawData.Data))
    data['time'] = rawData.Times
    if write2File:
        with open(write2File, 'wb') as f:
            pickle.dump(data, f)
    return data

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

__version__ = 1.1
修改日期：2016年12月8日
修改内容：
    添加数据存储到文件的函数和从文件中读取pd数据的函数

__version__ = 1.2
修改日期：2016年12月20日
修改内容：
    1. 添加获取指数成分的函数 get_index_constituent
    2. 添加获取区间交易日的函数get_tds
'''
__version__ = 1.2

from WindPy import w
import pandas as pd
import pickle


def get_data(code, field, startDate, endDate, highFreq=False, highFreqBarSize='barSize=5',
             priceAdj=False):
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
    assert rawData.ErrorCode == 0, rawData.Data[0][0]
    data = dict(zip(field, rawData.Data))
    data['time'] = rawData.Times
    return data


def data2file(data, filePath, toPd=True, readable=False):
    '''
    将数据写入文件保存，存储分为两种功能，一种就是单纯的存储，另外一种就是将数据存到一些文件（例如csv）
    方便导出查看。
    @param:
        data: 即为原始数据
        filePath: 为保存文件的路径，包括文件所在的文件夹和文件名以及后缀
        toPd: 是否保存为pandas的形式，默认为True。为False时，文件将被保存为pickle的形式；当为True时，
            将把文件先转换为pd.DataFrame的形式，具体存储形式需要根据后面的readable来设定。
        readable: 只有当toPd为True时，才会考虑该选项。若为True，则将数据保存为csv的格式，此时需要要求
            文件后缀为.csv，方便可以使用外部软件查看。若为False，则将数据保存为pickle的格式，此时对后缀
            无要求
    @return:
        succeed: bool型，如果保存成功则为True，反之则为False
    '''
    if toPd:
        try:
            writeData = pd.DataFrame(data)
        except TypeError:
            print('Data can not be write as pandas, a binary file is saved')
            with open(filePath, 'wb') as f:
                pickle.dump(data, filePath)
            return True
        if readable:
            suffix = filePath.split('.')[-1]
            assert suffix == 'csv', TypeError('unsupported readable file')
            writeData.to_csv(filePath)
            return True
        else:
            writeData.to_pickle(filePath)
            return True
    else:
        with open(filePath, 'wb') as f:
            pickle.dump(data, filePath)
        return True


def read_pdFile(filePath, timeColName='time', startTime=None, endTime=None, sep=','):
    '''
    用于从文件中读取金融数据，要求文件必须以pd.DataFrame的形式组织，返回pd.DataFrame的形式
    @param:
        filePath: 文件路径，包含文件所在的文件夹和文件名以及后缀。函数自行判断存储文件的格式，若后缀
            为csv，则可以提供sep参数来读取其他形式的csv文件；若后缀为其他形式，则按照pickle格式读取
        timeColName: 时间所在列的列名，默认为time，即表明时间列的列名为time
        startTime: 金融数据的开始时间，默认为None，即返回所有时间段的数据
        endTime: 金融数据的结束时间，默认为None，即返回所有时间段的数据
        sep: 当filePath的后缀为csv时启用，默认为逗号，即读取的文件为常规csv格式（以逗号为分割），若为
            其他形式的分隔符，需要自行提供
    @return:
        resData: pd.DataFrame格式，按照时间升序（由远到近）排序
    '''
    suffix = filePath.split('.')[-1]
    if suffix == 'csv':
        data = pd.read_csv(filePath, sep=sep)
    else:
        data = pd.read_pickle(filePath)
    if startTime is None:     # 表示当开始时间未给定时，采用数据中的开始时间
        startTime = data[timeColName].iloc[0]
    if endTime is None:
        endTime = data[timeColName].iloc[-1]
    data = data.loc[(data[timeColName] <= endTime) & (data[timeColName] >= startTime), :]
    data = data.sort_values(timeColName)
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
    if not w.isconnected():
        w.start()
    data = w.wset('sectorconstituent', 'date='+date.strftime('%Y-%m-%d')+';windcode='+indexCode)
    assert data.ErrorCode == 0, data.Data[0][0]
    indexConstituent = dict(zip(data.Data[1], data.Data[2]))
    return indexConstituent


def get_tds(startTime, endTime, colName='td'):
    '''
    从Wind中获取交易日
    @param:
        startTime: 交易日的起始日期，要求为dt.datetime格式或者YYYY-MM-DD格式
        endTime: 交易日的终止日期，同上述要求
        colName: 提供的交易日序列的列名，默认为td
    @return:
        tds: 字典{'td': 交易日序列}
    '''
    if not w.isconnected():
        w.start()
    tds = w.tdays(startTime, endTime)
    return {colName: tds.Data[0]}

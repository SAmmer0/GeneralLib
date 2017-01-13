#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-12-26 10:27:26
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
本程序用于对一些数据进行常用的预处理
__version__ = 1.0

修改日期：2016年12月26日
修改内容：
    添加基本函数gen_pdcolumns

__version__ = 1.1
修改日期：2017年1月4日
修改内容：
    添加在数据库中股票代码后添加后缀的函数add_suffix
'''
__version__ = 1.1
import pandas as pd
import datetime as dt
import numpy as np
import six


def gen_pdcolumns(data, operations):
    '''
    用于在data中添加新的列
    @param:
        data: 原始数据，要求为pd.DataFrame的格式
        operations: 需要添加的列，格式为{'colName': (func, {parameter dict})}或者{'colName': func}
            其中，func表示需要对数据进行处理的函数，要求函数只能调用每一行的数据，返回一个结果，且func
            的第一个参数为行数据，其他参数通过key=value的形式调用
    @return:
        res: 返回一个新的数据集，因为修改不是在原数据上进行的，因此需要采用以下方式才能将结果影响到原
            数据：data = gen_pdcolumns(data, operations)
    '''
    assert isinstance(data, pd.DataFrame), ('Parameter \"data\"" type error! Request ' +
                                            'pd.DataFrame, given %s' % str(type(data)))
    assert isinstance(operations, dict), ('Parameter \"operations\" type error! Request dict,' +
                                          'given %s' % str(type(operations)))
    res = data.copy()
    for col in operations:
        assert isinstance(col, str), 'Column name should be str, given %s' % str(type(col))
        operation = operations[col]
        if hasattr(operation, '__len__'):   # 表明为列表或其他列表类的类型
            assert len(operation) == 2, ('Operation paramter error! Request formula should like' +
                                         '(func, {parameter dict})')
            func = operation[0]
            params = operation[1]
            res[col] = res.apply(lambda x: func(x, **params), axis=1)
        else:
            res[col] = res.apply(func, axis=1)
    return res


def add_suffix(code):
    '''
    从数据库中获取的股票代码添加后缀，以60开头的代码添加.SH，其他添加.SZ
    @param:
        code: 需要转换的代码
    @return:
        转换后的代码
    注：转换前会检测是否需要转换，但是转换前不会检测代码是否合法
    '''
    if code.endswith('.SH') or code.endswith('.SZ'):
        return code
    if code.startswith('60'):   #
        suffix = '.SH'
    else:
        suffix = '.SZ'
    return code+suffix


def map_data(rawData, days, cols=None, fromNowOn=False):
    '''
    将给定一串时点的数据映射到给定的连续时间上，映射规则如下：
    若fromNowOn为True时，则在rawData中给定时点以及该时点后的时间的值等于该时点的值
    若fromNowOn为False时，则在rawData中给定时点后的时间的值等于该时点的值
    最终得到的时间序列数据为pd.DataFrame的格式，数据列为在rawData中在当前时间前（或者包含当前时间）
    的时间点所对应的数据
    例如：若rawData中包含相邻两项为(2010-01-01, 1), (2010-02-01, 2)，且fromNowOn=True，则结果中，从
    2010-01-01起到2010-02-01（不包含当天）的对应的值均为1，若fromNowOn=False，则结果中，从2010-01-01
    （不包含当天）起到2010-02-01对应的值为1
    @param:
        rawData: 原始时点上的数据，要求格式为[(time, value), ...]，不要求按照顺序排列
        days: 所需要映射的日期序列，要求为列表或者元组（注：要求时间格式均为datetime.datetime）
        cols: 返回结果的列名，要求为字典，形式为{'value': valueColName, 'time': timeColName}，默认为
            None表明使用默认的形式，即{'value': 'value', 'time': 'time'}
        fromNowOn: 默认为False，即在给定日期之后的序列的值才等于该值
    @return:
        pd.DataFrame格式的处理后的数据
    '''
    if cols is None:
        cols = {'value': 'value', 'time': 'time'}
    rawData = sorted(rawData, key=lambda x: x[0])
    valueIdx = 0
    while rawData[valueIdx][0] < days[0]:    # 将rawData的下标移动到有交叉时间段的地方
        valueIdx += 1
        if valueIdx >= len(rawData):    # idx移动到末尾，则返回最后一个日期的值
            return pd.DataFrame({cols['time']: days, cols['value']: [rawData[-1][1]]*len(days)})
    if fromNowOn:
        less_than = lambda x, y: x <= y
    else:
        less_than = lambda x, y: x < y
    dayIdx = 0
    # 将days的下标移动到有交叉的地方
    while not less_than(rawData[valueIdx][0], days[dayIdx]):
        dayIdx += 1
        if dayIdx >= len(days):
            print(u'Warning, no intersection data')   # 无交叉的时间段，返回nan
            return pd.DataFrame({cols['time']: days, cols['value']: [np.nan]*len(days)})
    valueList = list()
    timeList = list()
    for idx in six.moves.range(dayIdx, len(days)):
        try:
            nextDay = rawData[valueIdx][0]
        except IndexError:
            nextDay = None
        if not nextDay is None and less_than(nextDay, days[idx]):
            valueIdx += 1
        valueList.append(rawData[valueIdx-1][1])
        timeList.append(days[idx])
    res = {cols['time']: timeList, cols['value']: valueList}
    res = pd.DataFrame(res)
    return res


if __name__ == '__main__':
    startDate = dt.datetime(2016, 1, 1)
    days = list()
    for i in six.moves.range(100):
        startDate = startDate + dt.timedelta(days=1)
        days.append(startDate)
    endDate = startDate + dt.timedelta(days=1)
    nextDays = list()
    for i in six.moves.range(100):
        endDate = endDate + dt.timedelta(days=1)
        nextDays.append(endDate)
    # rawData = list((nextDays[i], i) for i in six.moves.range(10, len(nextDays), 5))
    rawData = list((days[i], i) for i in six.moves.range(10, len(days), 5))
    data = map_data(rawData, nextDays, fromNowOn=False)

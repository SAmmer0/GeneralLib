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

__version__ 1.2
修改日期：2017年1月13日
修改内容：
    添加map_data函数

__version__ 1.3
修改日期：2017年1月18日
修改内容：
    修复了map_data中的BUG，使其能够正确返回交叉时间段内的数据结果
'''
__version__ = 1.3
import pandas as pd
import datetime as dt
import numpy as np
import six
import copy


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


def drop_suffix(code, suffixLen=3, suffix=('.SH', '.SZ')):
    '''
    将Wind等终端获取的数据的后缀转换为数据库中无后缀的代码
    @param:
        code: 需要转换的代码
        suffixLen: 后缀代码的长度，包含'.'，默认为3
        suffix: 后缀的类型，默认为('.SH', '.SZ')
    @return:
        转换后的代码
    '''
    for s in suffix:
        if code.endswith(s):
            break
    else:
        return code
    return code[:-suffixLen]


def map_data(rawData, days, timeCols='time', fromNowOn=False):
    '''
    将给定一串时点的数据映射到给定的连续时间上，映射规则如下：
    若fromNowOn为True时，则在rawData中给定时点以及该时点后的时间的值等于该时点的值，因此第一个日期无论
    其是否为数据更新的时间点，都会被抛弃
    若fromNowOn为False时，则在rawData中给定时点后的时间的值等于该时点的值
    最终得到的时间序列数据为pd.DataFrame的格式，数据列为在当前时间下，rawData中所对应的最新的值，对应
    方法由映射规则给出
    例如：若rawData中包含相邻两项为(2010-01-01, 1), (2010-02-01, 2)，且fromNowOn=True，则结果中，从
    2010-01-01起到2010-02-01（不包含当天）的对应的值均为1，若fromNowOn=False，则结果中，从2010-01-01
    （不包含当天）起到2010-02-01对应的值为1
    注：该函数也可用于其他非时间序列的地方，只需要有序列号即可，那么rowData的数据应为[(idx, value), ...]
    @param:
        rawData: 原始时点上的数据，要求格式为[(time, value), ...]，不要求按照顺序排列，其中value为字典
            形式的数据，其基本格式为{colName: value}
        days: 所需要映射的日期序列，要求为列表或者元组（注：要求时间格式均为datetime.datetime）
        timeCols: 时间列的列名，默认为time
        fromNowOn: 默认为False，即在给定日期之后的序列的值才等于该值
    @return:
        pd.DataFrame格式的处理后的数据，数据长度与参数days相同
    注：可以直接使用pd.resample方法以及索引选择来计算，后期将改变或者删除
    '''
    rawData = sorted(rawData, key=lambda x: x[0])
    valueIdx = 0
    while rawData[valueIdx][0] < days[0]:    # 将rawData的下标移动到有交叉时间段的地方
        valueIdx += 1
        if valueIdx >= len(rawData):    # idx移动到末尾，则返回最后一个日期的值
            tmp = rawData[-1][1]
            res = dict()
            for key in tmp:
                res[key] = [tmp[key]]*len(days)
            res[timeCols] = days
            return pd.DataFrame(res)
    if fromNowOn:
        less_than = lambda x, y: x <= y
    else:
        less_than = lambda x, y: x < y
    dayIdx = 0
    # 将days的下标移动到有交叉的地方
    valueList = list()
    intersectionDay = valueIdx if valueIdx == 0 else valueIdx - 1
    while not less_than(rawData[intersectionDay][0], days[dayIdx]):
        tmpData = dict()
        for key in rawData[0][1]:
            tmpData[key] = np.nan
        tmpData[timeCols] = days[dayIdx]
        valueList.append(tmpData)
        dayIdx += 1
        if dayIdx >= len(days):
            return pd.DataFrame(valueList)

    for idx in six.moves.range(dayIdx, len(days)):
        try:
            nextDay = rawData[valueIdx][0]
        except IndexError:
            nextDay = None
        if nextDay is not None and less_than(nextDay, days[idx]):
            valueIdx += 1
        tmpData = copy.deepcopy(rawData[valueIdx-1][1])
        tmpData[timeCols] = days[idx]
        valueList.append(tmpData)
    res = pd.DataFrame(valueList).sort_values(timeCols).reset_index(drop=True)
    return res


def date_processing(date, dateFormat=None):
    '''
    用于检查日期的类型，如果为str则转换为datetime的格式
    @param:
        date: 输入的需要处理的日期
        dateFormat: 日期转换的格式，默认为None，表示格式为YYYY-MM-DD
    @return:
        按照转换方法转换后的格式
    '''
    if dateFormat is None:
        dateFormat = '%Y-%m-%d'
    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, dateFormat)
    return date

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
    synDays = days+nextDays
    # rawData = list((nextDays[i], {j: i*j for j in range(3)})
    #                for i in six.moves.range(10, len(nextDays), 5))
    rawData = list((days[i], {j: i*j for j in range(3)}) for i in six.moves.range(10, len(days), 5))
    # rawData = list((synDays[i], {j: i*j for j in range(3)})
    #                for i in six.moves.range(10, len(synDays), 5))
    data = map_data(rawData, nextDays, fromNowOn=False)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-02-05 19:59:17
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
该模块用于日常数据中的日期的处理

__version__ = 1.0
修改日期：2017年2月5日
修改内容：
    初始化

__version__ = 1.0.1
修改日期：2017-05-05
修改内容：
    为get_tds添加装饰器，使其返回标准格式的时间

__version__ = 1.0.2
修改日期：2017-05-11
修改内容：
    修改get_nth_tds的唤起错误的条件

__version__ = 1.1.0
修改日期：2017-05-12
修改内容：
    添加tdcount函数
'''
__version__ = '1.1.0'

from windwrapper import get_tds_wind

import datetime as dt
import pickle
import pandas as pd


def date_formater(date, dateFormat=None):
    '''
    用于检查日期的类型，如果为str则转换为datetime的格式
    @param:
        date: 输入的需要处理的日期
    @return:
        按照转换方法转换后的格式
    注：当前该程序可以删除，使用dateutil.parser.parse代替
    '''
    if dateFormat is None:
        dateFormat = '%Y-%m-%d'
    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, dateFormat)
    return date


def time_trans_wrapper(func):
    '''
    装饰器，将get_tds转变为返回标准时间格式（而不是wind时间格式）的函数
    @param:
        func: 需要被转换的函数
    @return:
        转换后的函数
    '''
    def inner(*args, **kwargs):
        tds = func(*args, **kwargs)
        tds = wind_time_standardlization(tds)
        return tds
    return inner


@time_trans_wrapper
def get_tds(startTime, endTime, fileName="F:\\GeneralLib\\CONST_DATAS\\tradingDays.pickle"):
    '''
    获取给定时间区间内的交易日，该函数会先检测是否有公用文件，如果有会从文件中读取交易日的数据，读取之后会将
    所读取的日期与给定的日期做比较，然后返回需要的交易日，如果所需要的交易日超过了文件的范围，则会从
    Wind中下载时间区间为(min(startTime, fileStartTime), max(endTime, fileEndTime))的交易日数据，并写入
    文件数据
    @param:
        startTime: 交易日的起始日期，要求为dt.datetime格式或者YYYY-MM-DD格式
        endTime: 交易日的终止日期，同上述要求
        fileName="F:\GeneralLib\CONST_DATAS\tradingDays.pickle": 从文件中读取交易日数据，默认在公用路径中
    @return:
        tds: 交易日列表
    备注：
        若startTime, endTime均为交易日，则二者均被包含到结果中，且microsecond=5000
    '''
    try:
        with open(fileName, 'rb') as f:
            rawTDs = pickle.load(f)
    except FileNotFoundError:  # 更改为Python2的IOError
        tds = get_tds_wind(startTime, endTime)
        with open(fileName, 'wb') as f:
            pickle.dump(tds, f)
        return tds

    def str2dt(dtStr):    # 用于将字符串转换为dt.datetime
        return time2wind(dt.datetime.strptime(dtStr, '%Y-%m-%d'))
    if isinstance(startTime, str):
        startTime = str2dt(startTime)
    else:
        startTime = time2wind(startTime)
    if isinstance(endTime, str):
        endTime = str2dt(endTime)
    else:
        endTime = time2wind(endTime)
    fileStartTime = rawTDs[0]
    fileEndTime = rawTDs[-1]
    if (fileStartTime > time2wind(startTime) or
            fileEndTime < time2wind(endTime)):     # 当前时间区间超过了文件中的范围
        # 当提供的时间为非交易日时，则每次调用都会超出文件范围，导致需要调用Wind
        tds = get_tds_wind(min(fileStartTime, startTime), max(fileEndTime, endTime))
        with open(fileName, 'wb') as f:
            pickle.dump(tds, f)
        tds = pd.Series(tds)
        tds = tds[(tds >= startTime) & (tds <= endTime)].tolist()
        return tds
    else:
        rawTDs = pd.Series(rawTDs)
        tds = rawTDs[(rawTDs >= startTime) & (rawTDs <= endTime)].tolist()
        return tds


def wind_time_standardlization(data, colName=None):
    '''
    将从Wind中获取的数据的时间标准化，由于Wind中获取的数据的时间的微秒均为5000，需要标准化与数据库
    数据相匹配
    @param:
        data: 原始数据，要求为pd.DataFrame的类型，或者可迭代的类型（此时要求该迭代类型中数据为时间数
            据)，或者为单独的dt.datetime类型
        colName: 时间所在列的名称，只有在输入为pd.DataFrame时有效
    @return:
        resData: 标准化处理后的数据，即将微秒设为0
    '''
    if isinstance(data, pd.DataFrame):
        resData = data.copy()
        resData[colName] = resData[colName].apply(lambda x: x.replace(microsecond=0))
    elif isinstance(data, dt.datetime):
        resData = data.replace(microsecond=0)
    else:
        resData = [x.replace(microsecond=0) for x in data]
    return resData


def time2wind(date):
    '''
    将时间转换为Wind常用的时间，即将microsecond变为5000
    @param:
        date: dt.datetime格式的时间
    @return:
        转换后的Wind格式时间
    '''
    return date.replace(microsecond=5000)


def get_nth_day(days, transFunc, nth, timeColName='time', to_df=True):
    '''
    用于将日期按照提供的方法transFunc分类，将分类后的数据按照时间（升序）排列，取出其中某个数据（第
    nth个点），组成新的时间序列
    @param:
        days: 时间列表，要求为pd.DataFrame的格式或者可迭代的形式
        transFunc: 用于将时间进行分类的函数，形式为transFunc(date)->str
        nth: 选取分类后的第n个数据作为结果数据，若时间按照给定方法分类后的最小长度为x，则nth应该在
            [-x, x]之间，索引下标的方式与Python的方式相同
        timeColName: 若提供的是pd.DataFrame的格式，则需要提供时间列的列名，默认为time
        to_df: 是否返回DataFrame格式的数据，默认为True
    @return:
        时间序列，类型为pd.DataFrame或者list类型
    '''
    if not isinstance(days, pd.DataFrame):
        days = pd.DataFrame({timeColName: days})
    days['catagory'] = days[timeColName].apply(transFunc)
    daysGroup = days.groupby('catagory')
    cataCnt = daysGroup.count()[timeColName].min()
    if nth > cataCnt - 1 or nth < -cataCnt:
        raise IndexError('index out of range')
    resDays = daysGroup.apply(lambda x: x[timeColName].iloc[nth])
    resDays = pd.DataFrame({timeColName: resDays}).sort_values(timeColName).reset_index(drop=True)
    if not to_df:
        resDays = resDays[timeColName].tolist()
    return resDays


def get_latest_report_dates(date, num, reverse=True):
    '''
    获取最近的几个报告期，例如date=2015-03-18 num=3，则结果返回[dt.datetime(2014, 12, 31),
    dt.datetime(2014, 9, 30), dt.datetime(2014, 6, 30)]，若date本身为报告期，则结果中包含date

    @param:
        date: 日期，可以为yyyy-mm-dd或者datetime的格式
        num: 报告期的数量
        reverse: 是否按照降序排列，默认为True
    @return:
        num个最近的报告期列表(若date为报告期，则包含date)，以dt.datetime的格式返回，按降序排列
    '''
    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, '%Y-%m-%d')
    rptDates = ['0331', '0630', '0930', '1231']
    optionalRes = list()
    for year in range(date.year - num // 4 - 1, date.year + 1):
        for rptd in rptDates:
            optionalRes.append(dt.datetime.strptime(str(year) + rptd, '%Y%m%d'))
    optionalRes = [x for x in optionalRes if x <= date]
    optionalRes = optionalRes[-num:]
    if reverse:
        optionalRes = sorted(optionalRes, reverse=reverse)
    return optionalRes


def is_continuous_rptd(dates):
    '''
    判断报告期是否连续
    例如：20160331， 20160630， 20160930， 20161231， 20170331为连续
    @param:
        dates: 需要做出判断的日期序列，要求为dt.datetime的格式
    @return:
        连续返回True，反之返回False
    注：对于长度为一的序列，全部返回True
    '''
    if len(dates) == 1:
        return True
    dates = sorted(dates)
    last_date = dates[-1]
    rpt_dates = get_latest_report_dates(last_date, len(dates), reverse=False)
    return all(d == rpt for d, rpt in zip(dates, rpt_dates))


def tdcount(days, end_time, method='close'):
    '''
    计算给定的一些日期到指定截至日期之间交易日的数量
    Parameter
    ---------
    days: list like, elements can be str or datetime, or single element
        起始时间的序列，不要求一定为交易日，如果为非交易日，则计算其后最近的交易日到end_time之间
        交易日的数量，要求days中的元素不能有重复的
    end_time: str or datetime or type that can be converted by pd.to_datetime
        截止日期，如果end_time不是交易日，则计算到小于end_time的最近交易日之间交易日的数量
    method: str default close
        计算交易日数量的方法，可选的方法有'close', 'half-close'，具体计算方法见Notes

    Return
    ------
    out: list
        对应的每个日期与end_time之间交易日的数量

    Notes
    -----
    计算方法包含三种：
        close: 包含首尾的日期在内
        half-close: 只包含首或者尾
    三者的关系如下：
        close = half-close + 1
    '''
    if isinstance(days, (str, dt.datetime)):
        days = [days]
    days = [pd.to_datetime(t) for t in days]
    min_day = min(days)
    tds = get_tds(min_day, end_time)
    out = [len([d for d in tds if d >= t and d <= end_time]) for t in days]
    if method == 'half-close':
        out = [x - 1 for x in out]
    return out


if __name__ == '__main__':
    days = list(pd.date_range('2017-01-01', periods=20, freq='D'))
    test = tdcount(days, days[-1])

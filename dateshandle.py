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
    1. 添加tdcount函数
    2. 添加get_rebtd函数

__version__ = 1.1.1
修改日期：2017-06-01

__version__ = 1.1.2
修改日期：2017-07-19
修改内容：
    添加get_recent_td用于获取最近的交易日

__version__ = 1.1.2
修改日期：2017-07-31
修改内容：
    添加tds_count函数，用于计算时间段内交易日数量

__version__ = 1.1.3
修改日期：2017-09-04
修改内容：
    添加tds_shift函数，用于将交易日往前推给定个交易日的数量
'''
__version__ = '1.1.2'


import pdb
from windwrapper import get_tds_wind

import datetime as dt
import pickle
from itertools import groupby
import pandas as pd
import sysconfiglee

# --------------------------------------------------------------------------------------------------
# 常量设置
TDS_FILE_PATH = sysconfiglee.get_config('tds_file_path')
# --------------------------------------------------------------------------------------------------


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
def get_tds(startTime, endTime, fileName=TDS_FILE_PATH):
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
    # return date.replace(microsecond=5000)
    return date.replace(hour=0, minute=0, second=0, microsecond=0)     # wind修改了交易日的格式，去除掉了微秒


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


def get_rebtd(start_time, end_time, freq='M', nth=-1):
    '''
    用于计算换仓的时间点
    Parameter
    ---------
    start_time: type that can be converted by pd.to_datetime
        开始时间
    end_time: type that can be converted by pd.to_datetime
        结束时间
    freq: str
        频率，目前仅支持月度和周度，即'M'和'W'
    nth: int
        定投时点，目前仅支持0和-1，分别表示在周期出和周期末定投

    Return
    ------
    out: list
        根据参数计算出的定投时间序列
    '''
    tds = get_tds(start_time, end_time)
    if freq == 'M':
        time_format = '%Y-%m'
    elif freq == 'W':
        time_format = '%Y-%W'
    else:
        raise ValueError('valid "freq" values are (M, W), you provide {}'.format(freq))
    out = get_nth_day(tds, lambda x: x.strftime(time_format), nth, to_df=False)
    return out


def get_recent_td(day):
    '''
    获取在给定日期之前距离给定日期最近的交易日

    Parameter
    ---------
    day: str or datetime
        给定需要获取最近交易日的日期

    Return
    ------
    out: pd.Timestamp
        给定日期的最近交易日
    '''
    offset = dt.timedelta(30)
    start_date = pd.to_datetime(day) - offset
    tds = get_tds(start_date, day)
    assert len(tds) > 0, "Error, time duration too short"
    return tds[-1]


def tds_count(start_time, end_time):
    '''
    获取给定时间段内的交易日的数量（包含首尾）

    Parameter
    ---------
    start_time: type that can be converted by pd.to_datetime
        开始时间
    end_time: type that can be converted by pd.to_datetime
        结束时间

    Notes
    -----
    计数包含起始的时间
    '''
    tds = get_tds(start_time, end_time)
    return len(tds)


def tds_shift(date, offset):
    '''
    将给定的交易日往前推或者往后推，使得得到的结果到当前交易日这段时间（包含首尾）至少包含offset+1个交易日
    Parameter
    ---------
    date: datetime like
        锚定的开始往前推的时间
    offset: int
        期间至少需要包含的交易日的数量，如果offset大于等于0表示往前推（过去），offset小于0表示往后推（未来）

    Return
    ------
    out: datetime like
        返回的时间使得这个时间与参数date之间至少包含offset个交易日
    '''
    if offset < 0:
        offset = - offset
        sign = -1
    else:
        sign = 1
    shift_days = int(offset / 20 * 31)
    date = pd.to_datetime(date)
    res = date - (pd.Timedelta('30 day') + pd.Timedelta('%d day' % shift_days)) * sign
    return res


def tds_pshift(date, offset):
    '''
    给定日期往前推移（过去），使得返回的结果到当前这段时间内（包含首尾）一共包含offset个交易日，即如果首尾中有一个
    不需要包含进去，则长度为offset

    Parameter
    ---------
    date: datetime like
        锚定的开始往前推的时间
    offset: int
        期间需要包含的交易日数量，必须为正数

    Return
    ------
    out: datetime like
        返回的时间使得这个时间与参数date之间（包含首尾）恰好包含offset个交易日
    '''
    assert offset > 0, "offset参数不合法，必须为正数，提供的参数为{}".format(offset)
    pre_date = tds_shift(date, offset)
    tds = get_tds(pre_date, date)
    return tds[-offset]


def tds_fshift(date, offset):
    '''
    tds_pshift的互补版本，能够精确往后推（未来），使得返回的结果到当天这段时间（包含收尾）一共包含
    offset个交易日
    Parameter
    ---------
    date: datetime like
        锚定的开始往前推的时间
    offset: int
        期间需要包含的交易日数量，必须为正数

    Return
    ------
    out: datetime like
        返回的时间使得这个时间与参数date之间（包含首尾）恰好包含offset个交易日
    '''
    assert offset > 0, 'offset参数不合法，必须为正数，提供的参数为{}'.format(offset)
    forward_date = tds_shift(date, -offset)
    tds = get_tds(date, forward_date)
    return tds[offset - 1]


def get_period_end(dates, fmt='%Y-%m', td_flag=True):
    '''
    对日期按照一定的规则进行分组，并返回每个分组的最后一个日期

    Parameter
    ---------
    dates: iterable
        元素为datetime类型或者其子类类型
    fmt: string
        将时间进行分组的方法，通过调用strftime来进行分组，默认分组模式为%Y-%m，即按照月度分组
    td_flag: boolean
        标识当前是否是仅限于交易日，True标识对于最后一日的判断是基于交易日而非自然日，默认为True

    Return
    ------
    out: list
        元素与输入类型相同，按照升序排列

    Notes
    -----
    并不是所有的分组中都会产生有效的最后日期，例如，如果12月份的时间数据不足，最后一个数据为'2017-12-03'，
    程序会通过自动判断来剔除这个数据，即结果中没有12月份的最后一个日期的数据
    '''
    by_freq = groupby(dates, lambda x: x.strftime(fmt))
    if td_flag:
        def next_day_func(x):
            return tds_fshift(x, 2)
    else:
        def next_day_func(x):
            return x + dt.timedelta(1)
    out = []
    for k, ds in by_freq:
        tmp_max = max(ds)
        if tmp_max.strftime(fmt) != next_day_func(tmp_max).strftime(fmt):    # 表明当前最大值并非为该分组下最后一日
            out.append(tmp_max)
    return sorted(out)


if __name__ == '__main__':
    res = get_recent_td('2017-01-31')
    print(res)

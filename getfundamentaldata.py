#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-01-15 17:01:37
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
用于从数据库中获取基本面数据
基本面数据包含两种：流量数据，存量数据

__version__ = 1.0
修改日期：2017年1月15日
修改内容：
    初始化，添加基本的函数
'''
import processingdata as psd
import datetime as dt
import pandas as pd


def get_latest_report_dates(date, num, reverse=True):
    '''
    获取最近的几个报告期，例如date=2015-03-18 num=3，则结果返回[dt.datetime(2014, 12, 31),
    dt.datetime(2014, 9, 30), dt.datetime(2014, 6, 30)]，若date本身为报告期，则结果中包含date

    @param:
        date: 日期，可以为yyyy-mm-dd或者datetime的格式
        num: 报告期的数量
        reverse: 是否按照降序排列，默认为True
    @return:
        num个最近的报告期列表，以dt.datetime的格式返回，按降序排列
    '''
    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, '%Y-%m-%d')
    rptDates = ['0331', '0630', '0930', '1231']
    optionalRes = list()
    for year in range(date.year-num//4-1, date.year+1):
        for rptd in rptDates:
            optionalRes.append(dt.datetime.strptime(str(year)+rptd, '%Y%m%d'))
    optionalRes = [x for x in optionalRes if x <= date]
    optionalRes = sorted(optionalRes, reverse=reverse)
    return optionalRes[:num]


def get_flows_data_factory(sqlStatment, cols, num=4, rawDataHandle=None, **kwargs):
    '''
    用于生产从报表中获取流量数据的函数
    @param:
        sqlStatement: SQL语句，从数据库中获取所需要指标的语句，要求语句中需要包含{code}, {startDate},
            {endDate}，{date}这几项，便于匹配获取数据
        cols: 列表形式，内容依次为列名，顺序确定依次为[dataCol, publDateCol, endDateCol]
        num: 回数报告期的期数（按季度算），默认为4，即TTM
        rawDataHandle: 用于处理原始数据的函数，其基本形式应为rawDathandle(data, **kwargs)，默认为None，
            即对数据不做处理
        kwargs: 字典形式，为传入rawDataHandle的其他参数
    @return:
        用于获取所要求时点的流量数据的函数
    '''
    assert len(cols) == 3, 'cols parameter error!'

    def inner(db, code, date):
        '''
        用于从数据库中获取给定时点上的{num}期（季）的总和数据
        @param:
            db: 数据库对象
            code: 需要获取的数据对应的证券代码，可以为带后缀或者不带后缀的股票代码
            date: 获取数据的时点
        @return:
            给定时点上，可以获得的最近{num}各报告期的总和数据
        '''.format(num=num)
        code = psd.drop_suffix(code)
        date = psd.date_processing(date)
        latestRptDates = get_latest_report_dates(date, num+4)
        earliestRpt, nearestRpt = latestRptDates[-1], latestRptDates[0]
        statement = sqlStatment.format(code=code, startDate=earliestRpt.strftime('%Y-%m-%d'),
                                       endDate=nearestRpt.strftime('%Y-%m-%d'),
                                       date=date.strftime('%Y-%m-%d'))
        data = db.fetchall(statement)
        stdData = list()
        for d in data:  # 用于将数据库中数字类型转化为float
            tmpData = list()
            for tmpD in d:
                if not isinstance(tmpD, dt.datetime):
                    tmpD = float(tmpD)
                tmpData.append(tmpD)
            tmpData = dict(zip(cols, tmpData))
            stdData.append(tmpData)
        stdData = pd.DataFrame(stdData)

        def process_data(df):
            df = df.sort_values(cols[1], ascending=False).reset_index(drop=True)
            return df.iloc[0]
        processedData = stdData.groupby(cols[2]).apply(process_data)
        processedData = processedData.sort_values(
            cols[2], ascending=False).reset_index(drop=True)
        assert len(processedData) >= num, 'length of data is not enough'
        if not rawDataHandle is None:
            processedData[cols[0]] = processedData[cols[0]].apply(rawDataHandle, **kwargs)
        return processedData.iloc[:num][cols[0]].sum()
    return inner


def get_point_data_factory(sqlStatment, colNum, rawDataHandle=None, **kwargs):
    '''
    用于生产从报表中获取时点数据的函数
    @param:
        sqlStatement: SQL语句，从数据库中获取所需要指标的语句，要求语句中需要包含{code}, {startDate},
            {endDate}，{date}这几项，便于匹配获取数据，且数据要求所需要的数据在第一行返回，即数据需要按照endDate
            降序排列，且publDate<当前日期以避免错误
        colNum: 时点数据所在的列索引下标
        rawDataHandle: 用于处理原始数据的函数，其基本形式应为rawDathandle(data, **kwargs)，默认为None，
            即对数据不做处理
        kwargs: 字典形式，为传入rawDataHandle的其他参数
    @return:
        用于获取给定时间上可以获得的存量数据的函数
    '''
    def inner(db, code, date):
        '''
        用于从数据库中获取给定时点上的最近一期的财务报表存量数据
        @param:
            db: 数据库对象
            code: 需要获取的数据对应的证券代码，可以为带后缀或者不带后缀的股票代码
            date: 获取数据的时点
        @return:
            给定时点上，可以获得的最近一期的存量数据
        '''
        code = psd.drop_suffix(code)
        date = psd.date_processing(date)
        latest4RptDates = get_latest_report_dates(date, 4)
        earliestRpt, nearestRpt = latest4RptDates[-1], latest4RptDates[0]
        statement = sqlStatment.format(code=code, startDate=earliestRpt, endDate=nearestRpt,
                                       date=date)
        data = db.fetchone(statement)
        data = float(data[colNum])
        if not rawDataHandle is None:
            data = rawDataHandle(data, **kwargs)
        return data
    return inner


def get_data_TS_factory(dateSQL, handle):
    '''
    用于生产获取时间序列数据的函数
    @param:
        dateSQL: 获取数据更新日期的SQL语句，要求内部包含{code}, {firstTD}, {lastTD}，且结果数据中仅有
            日期一列，要求SQL语句中返回的数据要包含第一个以及最后一个交易日
        handle: 实际获取数据的函数，要求形式为handle(db, code, date)
    @return:
        能够用于获取更新日期及其对应数据的时间序列函数
    '''
    def inner(db, code, startDate, endDate):
        '''
        用于获取所给定的时间序列内，财报更新时间点及对应的数据构成的时间序列
        @param:
            db: 数据库对象
            code: 需要获取的数据对应的证券代码，可以为带后缀或者不带后缀的股票代码
            startDate: 所需要数据的开始时间
            endDate: 所需要数据的结束时间
        @return:
            数据更新时间点（包含startDate, endDate）及该时间点对应的数据序列，形式为
            [(time, data), ...]
        '''
        code = psd.drop_suffix(code)
        startDate = psd.date_processing(startDate)
        endDate = psd.date_processing(endDate)
        statement = dateSQL.format(code=code, firstTD=startDate, lastTD=endDate)
        data = db.fetchall(statement)
        data = list(set(x[0] for x in data))
        resData = list()
        tmpData = handle(db, code, startDate)
        resData.append((startDate, tmpData))    # startDate不一定是更新时间点
        for date in data:
            tmpData = handle(db, code, date)
            resData.append((date, tmpData))
        return resData
    return inner


def get_TS_data(dataSQL, dateSQL, colName, db, code, days, dataType=1, seasonNum=4, colIdx=0,
                rawDataHandle=None, **kwargs):
    '''
    从财务报表数据库中获取所需要的数据，并将其转化为给定日期的值
    @param:
        dataSQL: SQL语句，从数据库中获取所需要指标的语句，要求语句中需要包含{code}, {startDate},
            {endDate}这几项，便于匹配获取数据，对于获取时点数据，要求需要的数据在返回数据的第一列
        dateSQL: 获取数据更新日期的SQL语句，要求内部包含{code}, {firstTD}, {lastTD}
        colName: 获取的数据的列名
        db: 数据库对象
        code: 需要获取的数据对应的证券代码，可以为带后缀或者不带后缀的股票代码
        days: 需要知道数据值的日期序列
        dataType: 所获取的数据的类型，时点数据（1）或者流量数据（2），默认为时点数据（1）
        seasonNum: 在获取流量数据时需要给定获取数据的期数（季度数），默认为4
        colIdx: 在获取存量数据时，需要给定获取的数据所在的列，默认为0
        rawDataHandle: 对原始数据进行处理的函数，形式要求为rawDataHandle(data, **kwargs)
        kwargs: 特殊处理的函数的其他参数
    @return:
        所要求的时间序列数据，形式为pd.DataFrame
    '''
    assert dataType in (1, 2), 'dataType parameter error!'
    if dataType == 1:
        get_raw_data = get_point_data_factory(dataSQL, colIdx, rawDataHandle, **kwargs)
    else:
        get_raw_data = get_flows_data_factory(dataSQL, [colName, 'publDate', 'endDate'], seasonNum,
                                              rawDataHandle, **kwargs)
    get_data = get_data_TS_factory(dateSQL, get_raw_data)
    startDate = days[0]
    endDate = days[-1]
    data = get_data(db, code, startDate, endDate)
    data = psd.map_data(data, days, cols={'value': colName, 'time': 'time'},
                        fromNowOn=False)
    return data

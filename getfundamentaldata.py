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

__version__ = 2.0
修改日期：2017年1月18日
修改内容：

'''
import processingdata as psd
import getfinancialdata as gfd
import processingdata as psd

import datetime as dt
import pandas as pd
import numpy as np


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
    for year in range(date.year-num//4-1, date.year+1):
        for rptd in rptDates:
            optionalRes.append(dt.datetime.strptime(str(year)+rptd, '%Y%m%d'))
    optionalRes = [x for x in optionalRes if x <= date]
    optionalRes = sorted(optionalRes, reverse=reverse)
    return optionalRes[:num]


def get_publdate_data_factory(dataSQL, cols, retrospectPeriodNum=1):
    '''
    生成获取在更新日当天的数据详情的函数，数据报告时间往前回溯N期（由参数retrospectPeriodNum决定）
    @param:
        dataSQL: 获取数据的SQL语句，要求最后两列依次为更新日期、报告期，且SQL中应包含{code},
            {rtpDate}, {publDate}分别表示股票代码、最早报告期、最晚更新日期
        cols: 数据列的列名，要求与SQL返回的结果顺序相对应，最后两列依次为更新日期、报告期的列名
        retrospectPeriodNum: 最早往前追溯的期数，默认为1，即只需要当前最新一期
    '''
    def inner(db, code, startDate, endDate):
        '''
        获取每个报告更新期的数据，最早可以追溯{rspn}期，取数据SQL如下：
        {SQL}
        @param:
            db: 数据库名，为SQLserver对象
            code: 股票代码
            startDate: 数据开始日期
            endDate: 数据结束日期（即最新更新日期在该日期之前）
        @return-> pd.DataFrame
            从startDate算起，往前数{rspn}个报告期开始，到更新日期在endDate之前为止的期间内的每个更新
            日的对应的数据，以pd.DataFrame的格式返回
        注：对于取出数据的结果，若为空，则直接返回的DataFrame中数值为空值，更新日期为startDate，报告
            期为startDate最近的报告期
        '''.format(rspn=retrospectPeriodNum, SQL=dataSQL)
        code = psd.drop_suffix(code)
        startDate = psd.date_processing(startDate)
        endDate = psd.date_processing(startDate)
        # 获取在该日期之前最近的retrospectPeriodNum个报告期
        rptDateStart = get_latest_report_dates(startDate, retrospectPeriodNum)[-1]
        sqlStatement = dataSQL.format(code=code, rptDate=rptDateStart, publDate=endDate)
        data = db.fetchall(sqlStatement)
        # 缺失值处理
        # 若此时没有任何值返回，直接返会na值，日期项分别为更新日期、对应报告期
        if data is None:
            latestRptDate = get_latest_report_dates(startDate, 1)
            data = dict(zip(cols, [[np.nan]]*(len(cols)-2)+[[startDate], [latestRptDate]]))
            data = pd.DataFrame(data)
            return data
        # 数据转换，当前假设所取出的数据有时间、str、数值和None几种类型
        newData = list()
        for row in data:
            tmpRow = list()
            for item in row:
                if item is None:
                    tmpRow.append(np.nan)
                elif isinstance(item, (str, dt.datetime)):
                    tmpRow.append(item)
                else:   # 除NA、字符、日期以外均按照数值处理
                    tmpRow.append(float(item))
            tmpRow = dict(cols, tmpRow)
            newData.append(tmpRow)
        res = pd.DataFrame(newData)
        return res
    return inner


def data_handle_factory(retrospectPeriodNum, valueCol, handle=None
                        rptDateCol='rptDate', publDateCol='publDate'):
    '''
    生成用于计算指标值的函数
    @param:
        retrospectPeriodNum: 计算需要数据的回溯期数
        handle: 用于计算指标的函数，要求形式为handle(pd.Series)->float，pd.Series已经按照更新日期的
            升序排列，默认为None，但是此时要求retrospecPeriodNum=1
        valueCol: 指标计算原始数值所在的列名
        rptDateCol: 报告期列的列名，默认为rptDate
        publDateCol: 更新日期列的列名，默认为publDate
    @return:
        用于计算指标的函数
    '''
    def inner(data):
    '''
        用于根据数据计算指标值
        @param:
            data: pd.DataFrame格式，需要包含{valueCol}, {rptDateCol}, {publDateCol}这三列
        @return:
            pd.Series，长度与data一致，由函数{func}计算的指标值
        注：若数据不够，则返回NA
    '''.format(valueCol=valueCol, rptDateCol=rptDateCol, publDateCol=publDateCol,
               func=handle.__name__)
        if handle is None:
            assert retrospectPeriodNum == 1, 'retrospect period number is wrong!'
        data = data.sort_values(publDateCol).reset_index(drop=True)
        startIdx = retrospectPeriodNum-1
        if startIdx >= len(data):
            return pd.Series([np.nan]*len(data))
        res = list()
        while True:     # 将下标移动到能够计算出值的地方，之前的数据按照NA处理
            rptDates = set(data.iloc[:startIdx+1][rptDateCol].tolist())
            if len(rptDates) >= retrospectPeriodNum:    # 表明从此处开始，报告期够用
                break
            startIdx += 1
            res.append(np.nan)
        while startIdx < len(data):
            currentRptDate = data.iloc[startIdx][rptDateCol]
            currentPublDate = data.iloc[startIdx][publDateCol]
            startRptDate = get_latest_report_dates(currentRptDate, retrospectPeriodNum)
            tmpData = data.loc[((data[rptDateCol] >= startRptDate) &
                                (data[rptDateCol] <= currentRptDate) &
                                (data[publDateCol] <= currentPublDate)), :]
            inputDatas = tmpData.groupby(rptDateCol).apply(lambda x: x.iloc[-1][valueCol])
            if len(inputDatas) < retrospectPeriodNum:    # 表明此时没有足够的数量的值
                tmpRes = np.nan
            else:
                if handle is not None:
                    tmpRes = handle(inputDatas)    # tmpData按照publDate的升序顺序排列的
                else:
                    tmpRes = inputDatas.iloc[0]
            res.append(tmpRes)
            startIdx += 1
        return pd.Series(res)
    return inner


def get_TS_data_factory(dataSQLs, cols, retrospectPeriodNum, funcs=None):
    '''
    生成获取时间序列数据的函数
    @param:
        dataSQLs: 用于获取数据的SQL，可迭代形式，长度可以为1或者2，为1表示无论什么行业都只有一张报表
            可取，为2表示有两种报表可取（例如，金融类和非金融类企业），此时需要将两种报表的数据合并,
            暂时假设两种报表的数据不可能冲突
        cols: 从报表中取出的数据的列名，顺序要与按照SQL取出数据的列对应
        retrosepectPeriodNum: 回溯期
        funcs: 处理某些列的函数，形式为{col: func, ...}，不需要处理的列可以不用提供，默认为None，即
            没有列需要处理
    @return:
        取数据，并将数据映射到交易日的函数
    '''
    def inner(db, code, startDate, endDate):
        '''
        从数据库中取出数据，并将取出的数据映射到交易日
        @param:
            db: 数据库对象
            code: 股票代码
            startDate: 数据的开始日期
            endDate: 数据的结束日期
        @return:
            pd.DataFrame格式，包含列为{cols}
        '''.format(cols=cols+['time'])
        dataCols = cols + ['publDate', 'rptDate']
        assert len(dataSQLs) <= 2, 'Erroe, SQL number exceeds!'
        publDatas = list()
        for dataSQL in dataSQLs:
            get_publdate_data = get_publdate_data_factory(dataSQL, dataCols, retrospectPeriodNum)
            tmpPublData = get_publdate_data(db, code, startDate, endDate)
            publDatas.append(tmpPublData)
        if len(publDatas) == 1:
            publDateData = publDatas[0]
        else:
            publDateData = combine_publdata_data(publDatas)
        if funcs is not None:
            assert isinstance(funcs, dict), 'handle function is wrong!'
            for col in funcs:
                func = funcs[col]
                publDateData[col] = func(publDateData)
        tds = gfd.get_tds(startDate, endDate)
        data = publDateData.loc[:, cols]
        prepredData = list()
        for i in range(len(data)):
            tmpData = list()
            tmpData.append(publDateData.iloc[i]['publDate'])
            tmpData.append(data.iloc[i].to_dict())
            prepredData.append(tmpData)
        res = psd.map_data(prepredData, tds)
        return res
    return inner


def combine_publdata_data(publDatas, dateCols=('publDate', 'rptDate')):
    '''
    将从不同类型的报表中取出的数据合并
    合并原则为：
        若两个报表中均有数据，则直接连接
        若两个报表中只有一个有非空的数据，则返回该表数据
        若两个报表中均无非空数据，则返回第一个（任意一个）
    @param:
        publDatas: 报表数据列表或者其他可迭代类型
        daateCols: 日期列的列名
    @return:
        按照规则合并后的结果
    '''
    NADataIdx = list()
    dataCols = set(publDatas[0].columns).difference(set(dateCols))
    for idx in range(len(publDatas)):
        tmpData = publDatas[idx]
        if len(tmpData) == 1:
            if np.all(pd.isnull(tmpData.loc[:, list(dataCols)])):
                NADataIdx.append(idx)
    if len(NADataIdx) == 0:  # 表明两个报表中取出的数据均有数值，则合并
        res = pd.concat(publDatas)
        return res
    elif len(NADataIdx) == 1:   # 表明其中一个报表上有非空的数值，则返回该报表
        validDataIdx = 1-NAData[0]
        return publDatas[validDataIdx]
    else:   # 两个报表均为空值，返回第一个（任意一个）
        return publDatas[0]

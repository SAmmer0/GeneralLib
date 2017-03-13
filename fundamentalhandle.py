#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-02-05 21:58:31
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

from decimal import Decimal
import datetime as dt
import pandas as pd
import numpy as np


import dateshandle
import processingdata as psd


class FDHBaseError(Exception):
    pass


class InvalidParameterError(FDHBaseError):
    pass


class DiscontinuousReportDateError(FDHBaseError):
    pass


class EmptyDataError(FDHBaseError):
    pass


class InvalidColumnError(FDHBaseError):
    pass


def clean_data(raw_data, col_names, replacer=None):
    '''
    用于将数据库中的数据转换为python中常用的数据格式
    转换规则如下：
        Decimal格式的数据转换为float，None转换为np.nan
    @param:
        raw_data: 从数据库中取出的原始数据
        col_names: 与数据相对应的各个列的列名
        replacer: 数据转换规则，默认为None，即使用上述规则，也可以自行提供，要求为输入为单个原始数据，
            输出为转换后的数据
    @return:
        转换后的数据，格式为pd.DataFrame
    注：若raw_data为空，则返回空的DataFrame
    '''
    def inner_replacer(data):
        if isinstance(data, Decimal):
            return float(data)
        elif data is None:
            return np.nan
        else:
            return data
    res = list()
    if replacer is None:
        replacer = inner_replacer
    for col in raw_data:
        cleaned_data = [replacer(d) for d in col]
        res.append(dict(zip(col_names, cleaned_data)))
    return pd.DataFrame(res)


def _get_latest(df, date, col):
    '''
    内部函数，用来从分组后的df中获取某个日期能观测到的最新数据
    @param:
        df: 按照报告期进行分组后的数据
        date: 给定的观测日
        col: 观测日所在的列
    @return:
        给定观测日的最新数据，Series形式，如果没有符合条件的数据，则对应的列都返回np.nan
    '''
    data = df.loc[df[col] <= date]
    if len(data) < 1:
        return gen_nan_series(df.columns)
    return data.iloc[-1]


def gen_nan_series(cols):
    return pd.Series(dict(zip(cols, [np.nan] * len(cols))))


def col_filter(raw_cols, tobe_filtered):
    '''
    去除一些不需要被包含的列
    @param:
        raw_cols: 原始列，可迭代类型
        tobe_filtered: 需要被去除的列，可迭代类型（注：此处并不会检查这些需要被去除的列是否全在
            原始列之中）
    @return:
        过滤之后的列
    '''
    res = [x for x in raw_cols if x not in tobe_filtered]
    return res


def cal_newest_data(data, period=1, rpt_col='rpt_date', update_col='update_time', pre_handle=None,
                    agg_handle=np.sum, **kwargs):
    '''
    用于计算给定数据在每个更新日期对应的最新数据
    @param:
        data: 原始数据，为df格式，要求必须包含rpt_col和update_col提供的列
        rpt_col: 报告期对应的列
        update_col: 更新日期对应的列
        pre_handle: 对数据进行预处理的函数，为字典类型，形式为{col: func}，默认为None，即不做预处理
        agg_handle: 对当前更新时间对应的最新数据做处理的函数
        kwargs: 用于向agg_handle中添加其他参数
    @return:
        返回df形式数据，包含数据列和update_col列，按照update_col列升序排列
    '''
    df = data.sort_values(update_col).reset_index(drop=True)    # 获取一个拷贝，避免修改原数据
    if pre_handle is not None:
        for col, fun in pre_handle:
            df[col] = fun(df[col])
    data_cols = col_filter(df.columns, [rpt_col, update_col])
    by_rdt = df.groupby(rpt_col)
    udt_list = list()
    data_list = list()
    for udt in sorted(df[update_col].unique()):
        udt_list.append(udt)
        tmp = by_rdt.apply(_get_latest, date=udt, col=update_col)   # 获取当前更新日最新数据
        tmp = tmp.dropna(subset=data_cols, how='all')
        tmp = tmp.iloc[-period:]
        if (len(tmp) < period or not dateshandle.is_continuous_rptd(tmp[rpt_col])):
            cal_res = gen_nan_series(data_cols)
        else:
            cal_data = tmp.loc[:, data_cols]
            cal_res = agg_handle(cal_data, **kwargs)
        data_list.append(cal_res)
    res = pd.DataFrame(data_list)
    res[update_col] = udt_list
    res = res.sort_values(update_col).reset_index(drop=True)
    return res


class Processor(object):

    '''
    原始数据由外部传入，然后计算更新日的最新数据列，然后根据要求返回所需要的数据
    '''

    def __init__(self, data, rpt_col='rpt_date', update_col='update_time', funcs=None):
        '''
        @param:
            data: 原始数据，要求为pd.DataFrame格式，且需要有rpt_col以及update_col这些参数提供的列
            rpt_col: 报告期所在列的列名
            update_col: 报告更新期所在列的列名
            funcs: 用于对原始数据进行进一步加工的函数，默认为None，即不需要加工，可提供的格式为字典，
                内容为{col: handle}
        '''
        self.rpt_col = rpt_col
        self.update_col = update_col
        self.data = data
        self.parameter_checker()
        self.convertor(funcs)
        self.newest_datas = None
        self._grouped = self.data.groupby(self.rpt_col)
        self._nperiod_data = None
        self._data_cols = [col for col in self.data.columns
                           if col not in [self.update_col, self.rpt_col]]

    def parameter_checker(self):
        '''
        检查参数是否合法，当传入的数据为空时，会引起EmptyDataError，其他都为InvalidParameterError
        '''
        if not isinstance(self.data, pd.DataFrame):
            raise InvalidParameterError('only pd.DataFrame is valid')
        if len(self.data) == 0:
            raise EmptyDataError('Input data is empty')
        for col_name in (self.rpt_col, self.update_col):
            if col_name not in self.data.columns:
                raise InvalidParameterError('%s not in data columns' % col_name)
        self.data = self.data.sort_values(self.update_col).reset_index(drop=True)

    def convertor(self, funcs):
        '''
        按照给定的方式转换数据
        @param:
            funcs: 字典形式，{col: handle}
        '''
        if funcs is None:
            return
        for col, func in funcs.items():
            self.data[col] = self.data[col].apply(func)

    def _cal_newest_data(self, date):
        valid_data = self.data.loc[self.data[self.update_col] <= date]
        grouped_valid_data = valid_data.groupby(self.rpt_col)
        res = grouped_valid_data.apply(lambda x: x.iloc[-1])
        return res

    def get_newest_data(self):
        '''
        计算所有更新日期对应的过去的最新的数据
        '''
        update_time = self.data[self.update_col]
        if self.newest_datas is None:
            self.newest_datas = dict()
            for udt in update_time:
                tmp = self._grouped.apply(_get_latest, date=udt, col=self.update_col)
                tmp = tmp.dropna(subset=self._data_cols, how='all')
                self.newest_datas[udt] = tmp

    def _get_nperiod_data(self, period_num=1):
        '''
        获取最新的给定期数的数据，以(update_time, newest_data)形式返回
        每次调用该函数后都会重新计算最新的值
        @param:
            period_num: 给定的期数
        '''
        self._nperiod_data = dict()
        for udt in sorted(self.newest_datas):
            self._nperiod_data[udt] = self.newest_datas[udt].iloc[-period_num:]
            # 此处将reset_index删除，因为在操作过程中并没有用到索引或者需要对齐
            # tmp = tmp.reset_index(drop=True)

    def output_data(self, period_num=1, func=np.sum):
        '''
        返回使用给定期数的数据计算的数据
        @param:
            period_num: 给定期数
            func: 使用给定期数计算出所需要数据的函数，默认为np.sum，要求函数形式为func(DataFrame)->
                Series
        @return:
            每个更新期对应的计算结果，其中以更新日期作为index
        注：
            如果报告期不连续，则那一期的数据结果为np.nan
        '''
        self.get_newest_data()
        self._get_nperiod_data(period_num=period_num)
        res = list()
        times = list()
        for udt, data in self._nperiod_data.items():
            # 数据不够，或者报告期不连续
            if (len(data) < period_num or
                    not dateshandle.is_continuous_rptd(data[self.rpt_col].tolist())):
                cal_res = gen_nan_series(self._data_cols)
            else:
                data = data.loc[:, self._data_cols]
                cal_res = func(data)
            res.append(cal_res)
            times.append(udt)
        res = pd.DataFrame(res)
        # res.index = [npd[0] for npd in self._nperiod_data]
        res['time'] = times
        res = res.sort_values('time').reset_index(drop=True)
        return res


class Config(object):

    '''
    用于设置获取数据库基本面数据的配置，目前只支持获取单个SQL的数据
    要求sql有类似于示例中的格式：
        SELECT C.FinancialExpense, C.IncomeTaxCost, C.TotalProfit, C.InfoPublDate, C.EndDate
        FROM SecuMain M, LC_QIncomeStatementNew C
        WHERE M.CompanyCode = C.CompanyCode AND
            M.SecuCode = \'{code}\' AND
            M.SecuMarket in (83, 90) AND
            M.SecuCategory = 1 AND
            C.EndDate >= CAST(\'{start_time}\' AS datetime) AND
            C.InfoPublDate <= CAST(\'{end_time}\' AS datetime)
    '''

    def __init__(self, sql, cols, period=1, handle=None):
        self.sql = sql
        self.cols = cols
        self.period = period
        self.handle = handle

    def is_valid_col(self, col):
        if col not in self.cols:
            raise InvalidColumnError('{c} is not in {cols}'.format(c=col, cols=self.cols))

    def add_handle(self, col, func):
        '''
        用于添加处理函数，也可以用于覆盖当前的函数
        '''
        if self.handle is None:
            self.handle = dict()
        self.is_valid_col(col)
        self.handle[col] = func

    def format_sql(self, code, start_time, end_time):
        if not isinstance(start_time, str):
            start_time = start_time.strftime('%Y-%m-%d')
        if not isinstance(end_time, str):
            end_time = end_time.strftime('%Y-%m-%d')
        code = psd.drop_suffix(code)
        return self.sql.format(code=code, start_time=start_time, end_time=end_time)

    def change_period(self, period):
        self.period = period


class SQLWrapper(object):

    '''
    直接与SQL进行连接和获取数据，目前只支持获取单个SQL的数据
    '''

    def __init__(self, cursor):
        self.cursor = cursor

    def get_data(self, config, code, start_time, end_time):
        sql = config.format_sql(code, start_time, end_time)
        data = self.cursor.fetchall(sql)
        data_cleaned = clean_data(data, config.cols)
        return data_cleaned


def test():
    cols = ['data', 'rpt_date', 'update_time']
    rpt_dates = dateshandle.get_latest_report_dates('2016-12-31', 20)

    def gen_dates(r_dates):
        rpt_res = list()
        update_res = list()
        repeat_num = 2
        for date in r_dates:
            for rn in range(repeat_num):
                rpt_res.append(date)
                if rn == 0:
                    update_res.append(date + dt.timedelta(60))
                else:
                    update_res.append(date + dt.timedelta(360))
        return rpt_res, update_res
    rpt_dates, update_dates = gen_dates(rpt_dates)
    data = np.random.randint(len(rpt_dates), size=(len(rpt_dates),))
    df = dict(zip(cols, [data, rpt_dates, update_dates]))
    df = pd.DataFrame(df)
    process_obj = Processor(df, rpt_col='rpt_date', update_col='update_time')
    res = process_obj.output_data()
    res2 = cal_newest_data(df, rpt_col='rpt_date', update_col='update_time')
    return res, res2


if __name__ == '__main__':
    res, res2 = test()

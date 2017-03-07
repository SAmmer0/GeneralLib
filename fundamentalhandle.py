#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-02-05 21:58:31
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

from decimal import Decimal
import pandas as pd
import numpy as np

import dateshandle


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


def get_config(sql, cols, period, handle, code, start_time, end_time):
    '''
    用于构造参数的辅助函数
    @param:
        sql: 符合Config类标准的SQL语句
        cols: 数据列列名
        period: 数据的期限
        handle: {col: func}的形式或者None
        code: 证券代码
        start_time: 获取数据的报告期开始时间
        end_time: 最新的更新时间
    @return:
        字典形式的配置好的参数，可直接用于SQLWrapper的get_data函数
    '''
    config = Config(sql, cols, period, handle)
    return {'config': config, 'code': code, 'start_time': start_time, 'end_time': end_time}


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
        self.newest_datas = None
        self.convertor(funcs)
        self._nperiod_data = None
        self._data_cols = set(self.data.columns).difference(set([self.update_col, self.rpt_col]))

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
            self.newest_datas[udt] = self._cal_newest_data(udt)

    def _get_nperiod_data(self, period_num=1):
        '''
        获取最新的给定期数的数据，以(update_time, newest_data)形式返回
        @param:
            period_num: 给定的期数
        '''
        if self._nperiod_data is None:
            self._nperiod_data = list()
            for udt in sorted(self.newest_datas):
                tmp = self.newest_datas[udt].iloc[-period_num:]
                tmp = tmp.reset_index(drop=True)
                self._nperiod_data.append((udt, tmp))

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
        for udt, data in self._nperiod_data:
            # 数据不够，或者报告期不连续
            if (len(data) < period_num or
                    not dateshandle.is_continuous_rptd(data[self.rpt_col].tolist())):
                cal_res = pd.Series(dict(zip(self._data_cols, [np.nan]*len(self._data_cols))))
            else:
                data = data.loc[:, list(self._data_cols)]
                cal_res = func(data)
            res.append(cal_res)
        res = pd.DataFrame(res)
        res.index = [npd[0] for npd in self._nperiod_data]
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
        用于添加处理函数
        '''
        if self.handle is None:
            self.handle = dict()
        self.is_valid_col(col)
        self.handle[col] = func

    def format_sql(self, code, start_time, end_time):
        return self.sql.format(code=code, start_time=start_time, end_time=end_time)


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
        return data

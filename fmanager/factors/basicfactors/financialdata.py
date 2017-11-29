#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-18 14:34:55
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
财务数据
__version__ = 1.0.0
修改日期：2017-07-25
修改内容：
    初始化
'''
import pdb
import datatoolkits
import dateshandle
import fdgetter
import fdmutils
import pandas as pd
from fmanager.factors.utils import (Factor, check_indexorder, check_duplicate_factorname,
                                    checkdata_completeness)
# from ..query import query

# --------------------------------------------------------------------------------------------------
# 常量和功能函数
NAME = 'financialdata'


def get_factor_dict():
    res = dict()
    for f in factor_list:
        res[f.name] = {'rel_path': NAME + '\\' + f.name, 'factor': f}
    return res


factor_list = []


def _handle_dbdata(data, start_time, end_time, func, **kwargs):
    '''
    将从数据库中获取的数据映射到交易日中

    Parameter
    ---------
    data: pd.DataFrame
        从数据库中获取的数据，要求列名为['update_time', 'rpt_date', 'code', 'data']
    start_time: str or other type that can be transfered by pd.to_datetime
        获取的数据的起始时间
    end_time: str or other type that can be transfered by pd.to_datetime
        获取数据的结束之间
    func: function
        要求function接受的第一个参数为pd.DataFrame，为经过代码和观察日分组的数据，且返回的结果
        为一个值或者只包含一个值得pd.Series
    **kwargs: dictionary like parameter
        为需要传入func的其他参数

    Return
    ------
    out: pd.DataFrame
        经过特定计算后的结果，columns为股票代码，index为日期
    '''
    by_code = data.groupby('code')
    data = by_code.apply(fdmutils.get_observabel_data).reset_index(drop=True)
    by_cno = data.groupby(['code', 'obs_time'])
    data = by_cno.apply(func, **kwargs)
    data = data.reset_index().rename(columns={'obs_time': 'time'})
    tds = dateshandle.get_tds(start_time, end_time)
    data = data.groupby('code').apply(datatoolkits.map_data, days=tds,
                                      fillna={'code': lambda x: x.code.iloc[0]})
    data = data.reset_index(drop=True)
    data = data.pivot_table('data', index='time', columns='code')
    return data
# --------------------------------------------------------------------------------------------------
# TTM数据


def get_TTM(fd_type, sql_type):
    '''
    母函数，用于生成获取一些指标TTM的函数

    Parameter
    ---------
    fd_type: str
        需要计算TTM的因子类型

    sheet_type: str
        因子数据所在的表的类型，只支持{'季度利润表': 'IS', '季度现金流量表': 'CFS'}
    '''
    sql_type_dict = {'IS': 'LC_QIncomeStatementNew', 'CFS': 'LC_QCashFlowStatementNew'}
    sql_type = sql_type_dict[sql_type]
    sql_template = '''
    SELECT S.InfoPublDate, S.EndDate, M.SecuCode, S.data
    FROM sql_type S, SecuMain M
    WHERE
        M.CompanyCode = S.CompanyCode AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.BulletinType != 10 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime) AND
        S.EndDate >= (SELECT TOP(1) S2.CHANGEDATE
                      FROM LC_ListStatus S2
                      WHERE
                          S2.INNERCODE = M.INNERCODE AND
                          S2.ChangeType = 1)
    ORDER BY M.SecuCode ASC, S.InfoPublDate ASC
        '''.replace('data', fd_type).replace('sql_type', sql_type)

    def _inner(universe, start_time, end_time):
        new_start = pd.to_datetime(start_time) - pd.Timedelta('540 day')
        data = fdgetter.get_db_data(sql_template, start_time=new_start, end_time=end_time,
                                    cols=('update_time', 'rpt_date', 'code', 'data'),
                                    add_stockcode=False)
        data['code'] = data.code.apply(datatoolkits.add_suffix)
        by_code = data.groupby('code')
        data = by_code.apply(fdmutils.get_observabel_data).reset_index(drop=True)
        by_cno = data.groupby(['code', 'obs_time'])
        data = by_cno.apply(fdmutils.cal_ttm, col_name='data').reset_index()\
            .rename(columns={'obs_time': 'time'})

        tds = dateshandle.get_tds(start_time, end_time)
        # pdb.set_trace()
        data = data.groupby('code').apply(datatoolkits.map_data, days=tds,
                                          fillna={'code': lambda x: x.code.iloc[0]})
        data = data.reset_index(drop=True)
        data = data.pivot_table('data', index='time', columns='code')
        data = data.loc[:, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return _inner


# 净利润TTM
factor_list.append(Factor('NI_TTM', get_TTM('NPFromParentCompanyOwners', 'IS'), pd.to_datetime('2017-07-25'),
                          desc='净利润TTM'))
# 营业收入TTM
factor_list.append(Factor('OPREV_TTM', get_TTM('OperatingRevenue', 'IS'), pd.to_datetime('2017-07-25'),
                          desc='营业收入TTM'))
# 营业利润TTM
factor_list.append(Factor('OPPROFIT_TTM', get_TTM('OperatingProfit', 'IS'),
                          pd.to_datetime('2017-07-25'), desc='营业利润TTM'))
# 销售费用TTM
factor_list.append(Factor('OPEXP_TTM', get_TTM('OperatingExpense', 'IS'), pd.to_datetime('2017-07-25'),
                          desc='销售费用TTM'))
# 管理费用TTM
factor_list.append(Factor('ADMINEXP_TTM', get_TTM('AdministrationExpense', 'IS'),
                          pd.to_datetime('2017-07-25'), desc='管理费用TTM'))
# 财务费用TTM
factor_list.append(Factor('FIEXP_TTM', get_TTM('FinancialExpense', 'IS'), pd.to_datetime('2017-07-25'),
                          desc='财务费用TTM'))
# 营业成本TTM
factor_list.append(Factor('OPCOST_TTM', get_TTM('OperatingCost', 'IS'), pd.to_datetime('2017-07-28'),
                          desc='营业成本'))
# 经营活动中的现金流净额
factor_list.append(Factor('OPNETCF_TTM', get_TTM('NetOperateCashFlow', 'CFS'),
                          pd.to_datetime('2017-07-25'), desc='经营活动中的现金流净额'))
# --------------------------------------------------------------------------------------------------
# 资产负债表最新数据


def get_BS_latest(fd_type):
    '''
    母函数，用于生成获取最新资产负债表数据的函数
    '''
    sql_template = '''
    SELECT S.InfoPublDate, S.EndDate, M.SecuCode, S.data_type
    FROM SecuMain M, LC_BalanceSheetAll S
    WHERE M.CompanyCode = S.CompanyCode AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.BulletinType != 10 AND
        S.IfMerged = 1 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime) AND
        S.EndDate >= (SELECT TOP(1) S2.CHANGEDATE
                      FROM LC_ListStatus S2
                      WHERE
                          S2.INNERCODE = M.INNERCODE AND
                          S2.ChangeType = 1)
    ORDER BY M.SecuCode ASC, S.InfoPublDate ASC
    '''.replace('data_type', fd_type)

    def _inner(universe, start_time, end_time):
        new_start = pd.to_datetime(start_time) - pd.Timedelta('250 day')
        data = fdgetter.get_db_data(sql_template, start_time=new_start, end_time=end_time,
                                    cols=('update_time', 'rpt_date', 'code', 'data'),
                                    add_stockcode=False)
        data['code'] = data.code.apply(datatoolkits.add_suffix)
        by_code = data.groupby('code')
        data = by_code.apply(fdmutils.get_observabel_data).reset_index(drop=True)
        by_cno = data.groupby(['code', 'obs_time'])
        data = by_cno.apply(lambda x: x.loc[:, 'data'].iloc[-1]).rename('data')
        data = data.reset_index().rename(columns={'obs_time': 'time'})
        tds = dateshandle.get_tds(start_time, end_time)
        data = data.groupby('code').apply(datatoolkits.map_data, days=tds,
                                          fillna={'code': lambda x: x.code.iloc[0]})
        data = data.reset_index(drop=True)
        data = data.pivot_table('data', index='time', columns='code')
        data = data.loc[:, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return _inner

# 注：从数据库中取出来的数据，有可能对应的位置会是NA值，有两种情况，一种是股票还未上市或者已经退市
# 第二种情况是数据库中的数据本身为Null，如果将NA值直接与其他的因子值相加会导致NA，因此在因子相加时
# 考虑将这些数据弄到一个DataFrame中，然后使用sum方法，忽略其中的NA值
# 总资产


factor_list.append(Factor('TA', get_BS_latest('TotalAssets'), pd.to_datetime('2017-07-25'),
                          desc='总资产'))
# 非流动性负债
factor_list.append(Factor('TNCL', get_BS_latest('TotalNonCurrentLiability'), pd.to_datetime('2017-07-25'),
                          desc='非流动性负债'))
# 流动性资产
factor_list.append(Factor('TCA', get_BS_latest('TotalCurrentAssets'), pd.to_datetime('2017-07-25'),
                          desc='流动性资产'))
# 流动负债
factor_list.append(Factor('TCL', get_BS_latest('TotalCurrentLiability'), pd.to_datetime('2017-07-25'),
                          desc='流动负债'))
# 归属母公司权益
factor_list.append(Factor('EQUITY', get_BS_latest('SEWithoutMI'), pd.to_datetime('2017-07-25'),
                          desc='归属母公司权益'))
# 现金
factor_list.append(Factor('CASH', get_BS_latest('CashEquivalents'), pd.to_datetime('2017-07-25'),
                          desc='现金'))
# 优先股
factor_list.append(Factor('PREFER_STOCK', get_BS_latest('EPreferStock'), pd.to_datetime('2017-10-27'),
                          desc='优先股'))
# --------------------------------------------------------------------------------------------------
# 特定时间季度数据（例如最新季度数据，往前推三个季度的数据等等）


def get_season_nshift(data_type, sql_type, n):
    '''
    母函数，用于生成获取对应数据的函数（对应的数据是指当前最新季度往前推n个季度的数据）

    Parameter
    ---------
    data_type: str
        从数据库字典中获取的对应数据的数据代码
    sql_type: str
        目前仅支持{'季度利润表': 'IS', '季度现金流量表': 'CFS'}
    n: int
        往前推的期数，例如，n=1表明最近一个季度，n=2表明上个季度（相对最近季度而言）
    '''
    sql_type_dict = {'IS': 'LC_QIncomeStatementNew', 'CFS': 'LC_QCashFlowStatementNew'}
    sql_type = sql_type_dict[sql_type]
    sql_template = '''
    SELECT S.InfoPublDate, S.EndDate, M.SecuCode, S.data
    FROM sql_type S, SecuMain M
    WHERE
        M.CompanyCode = S.CompanyCode AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.BulletinType != 10 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime) AND
        S.EndDate >= (SELECT TOP(1) S2.CHANGEDATE
                      FROM LC_ListStatus S2
                      WHERE
                          S2.INNERCODE = M.INNERCODE AND
                          S2.ChangeType = 1)
    ORDER BY M.SecuCode ASC, S.InfoPublDate ASC
    '''
    sql = sql_template.replace('data', data_type).replace('sql_type', sql_type)

    def _inner(universe, start_time, end_time):
        offset = 100 * n + 250
        new_start = pd.to_datetime(start_time) - pd.Timedelta('%d day' % offset)
        data = fdgetter.get_db_data(sql, start_time=new_start, end_time=end_time,
                                    cols=('update_time', 'rpt_date', 'code', 'data'),
                                    add_stockcode=False)
        data['code'] = data.code.apply(datatoolkits.add_suffix)
        # 该部分可以单独做成一个函数
        by_code = data.groupby('code')
        data = by_code.apply(fdmutils.get_observabel_data).reset_index(drop=True)
        by_cno = data.groupby(['code', 'obs_time'])
        data = by_cno.apply(fdmutils.cal_season, col_name='data', offset=n)
        data = data.reset_index().rename(columns={'obs_time': 'time'})
        tds = dateshandle.get_tds(start_time, end_time)
        data = data.groupby('code').apply(datatoolkits.map_data, days=tds,
                                          fillna={'code': lambda x: x.code.iloc[0]})
        data = data.reset_index(drop=True)
        data = data.pivot_table('data', index='time', columns='code')
        data = data.loc[:, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return _inner


# 最近季度归属母公司净利润
factor_list.append(Factor('NI_1S', get_season_nshift('NPFromParentCompanyOwners', 'IS', 1),
                          pd.to_datetime('2017-07-26'), desc='最近一个季度归属母公司净利润'))
# 往前推4个季度的归属母公司的净利润（即如果最近季度为一季度，那么该因子指的是上个年度的一季度数据）
factor_list.append(Factor('NI_5S', get_season_nshift('NPFromParentCompanyOwners', 'IS', 5),
                          pd.to_datetime('2017-07-26'), desc='往前推四个季度归属母公司净利润'))
# 最近季度营业收入
factor_list.append(Factor('OPREV_1S', get_season_nshift('OperatingRevenue', 'IS', 1),
                          pd.to_datetime('2017-07-26'), desc='最近一个季度营业收入'))
# 往前推4个季度的营业收入
factor_list.append(Factor('OPREV_5S', get_season_nshift('OperatingRevenue', 'IS', 5),
                          pd.to_datetime('2017-07-26'), desc='往前推四个季度营业收入'))
# --------------------------------------------------------------------------------------------------
# 特定时间的年度数据（例如最新财年数据，上个财年的数据）


def get_year_nshift(data_type, sql_type, n):
    '''
    母函数，用于生成用于生成获取对应数据的函数（对应的数据是指当前最新财年往前推n个财年的数据）

    Parameter
    ---------
    data_type: str
        从数据库字典中获取的对应数据的数据代码
    sql_type: str
        目前仅支持{'利润表': 'IS', '现金流量表': 'CFS'}
    n: int
        往前推的期数，例如，n=1表明最近一个财年，n=2表明上个财年（相对最近财年而言），以此类推
    '''
    sql_type_dict = {'IS': 'LC_IncomeStatementAll', 'CFS': 'LC_CashFlowStatementAll'}
    sql_type = sql_type_dict[sql_type]
    sql_template = '''
    SELECT S.InfoPublDate, S.EndDate, M.SecuCode, S.data
    FROM sql_type S, SecuMain M
    WHERE
        M.CompanyCode = S.CompanyCode AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.AccountingStandards = 1 AND
        S.IfAdjusted not in (4, 5) AND
        S.BulletinType != 10 AND
        S.IfMerged = 1 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime) AND
        S.EndDate >= (SELECT TOP(1) S2.CHANGEDATE
                      FROM LC_ListStatus S2
                      WHERE
                          S2.INNERCODE = M.INNERCODE AND
                          S2.ChangeType = 1)
    ORDER BY M.SecuCode ASC, S.InfoPublDate ASC
    '''
    sql = sql_template.replace('data', data_type).replace('sql_type', sql_type)

    def _inner(universe, start_time, end_time):
        offset = 366 * n + 180
        # pdb.set_trace()
        new_start = pd.to_datetime(start_time) - pd.Timedelta('%d day' % offset)
        data = fdgetter.get_db_data(sql, start_time=new_start, end_time=end_time,
                                    cols=('update_time', 'rpt_date', 'code', 'data'),
                                    add_stockcode=False)
        data['code'] = data.code.apply(datatoolkits.add_suffix)
        data = _handle_dbdata(data, start_time, end_time, fdmutils.cal_yr, col_name='data',
                              offset=n)
        # pdb.set_trace()
        data = data.loc[:, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return _inner


# 最近财年的归属母公司的净利润
factor_list.append(Factor('NI_1Y', get_year_nshift('NPParentCompanyOwners', 'IS', 1),
                          pd.to_datetime('2017-07-26'), desc='最近财年归属母公司的净利润'))
# 往前推2个财年（上财年）的归属母公司净利润
factor_list.append(Factor('NI_2Y', get_year_nshift('NPParentCompanyOwners', 'IS', 2),
                          pd.to_datetime('2017-07-26'), desc='往前推2个财年的归属母公司净利润'))
# 往前推3个财年的归属母公司净利润
factor_list.append(Factor('NI_3Y', get_year_nshift('NPParentCompanyOwners', 'IS', 3),
                          pd.to_datetime('2017-07-26'), desc='往前推3个财年的归属母公司净利润'))
# 往前推4个财年的归属母公司净利润
factor_list.append(Factor('NI_4Y', get_year_nshift('NPParentCompanyOwners', 'IS', 4),
                          pd.to_datetime('2017-07-26'), desc='往前推4个财年的归属母公司净利润'))
# 往前推5个财年的归属母公司净利润
factor_list.append(Factor('NI_5Y', get_year_nshift('NPParentCompanyOwners', 'IS', 5),
                          pd.to_datetime('2017-07-26'), desc='往前推5个财年的归属母公司净利润'))

# 最近财年的营业收入
factor_list.append(Factor('OPREV_1Y', get_year_nshift('OperatingRevenue', 'IS', 1),
                          pd.to_datetime('2017-07-26'), desc='最近财年营业收入'))
# 往前推2个财年（上财年）的营业收入
factor_list.append(Factor('OPREV_2Y', get_year_nshift('OperatingRevenue', 'IS', 2),
                          pd.to_datetime('2017-07-26'), desc='往前推2个财年的营业收入'))
# 往前推3个财年的营业收入
factor_list.append(Factor('OPREV_3Y', get_year_nshift('OperatingRevenue', 'IS', 3),
                          pd.to_datetime('2017-07-26'), desc='往前推3个财年的营业收入'))
# 往前推4个财年的营业收入
factor_list.append(Factor('OPREV_4Y', get_year_nshift('OperatingRevenue', 'IS', 4),
                          pd.to_datetime('2017-07-26'), desc='往前推4个财年的营业收入'))
# 往前推5个财年的营业收入
factor_list.append(Factor('OPREV_5Y', get_year_nshift('OperatingRevenue', 'IS', 5),
                          pd.to_datetime('2017-07-26'), desc='往前推5个财年的营业收入'))

# --------------------------------------------------------------------------------------------------


check_duplicate_factorname(factor_list, __name__)

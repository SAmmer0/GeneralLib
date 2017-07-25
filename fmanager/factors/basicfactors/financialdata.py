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
from ..utils import Factor, check_indexorder
# from ..query import query

# --------------------------------------------------------------------------------------------------
# 常量和功能函数
NAME = 'financialdata'


def get_factor_dict():
    res = dict()
    for f in factor_list:
        res[f.name] = {'rel_path': NAME + '\\' + f.name, 'factor': f}
    return res
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
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime)
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
        return data
    return _inner


# 净利润TTM
ni_ttm = Factor('NI_TTM', get_TTM('NPFromParentCompanyOwners', 'IS'), pd.to_datetime('2017-07-25'),
                desc='净利润TTM')
# 营业收入TTM
oprev_ttm = Factor('OPREV_TTM', get_TTM('OperatingRevenue', 'IS'), pd.to_datetime('2017-07-25'),
                   desc='营业收入TTM')
# 营业利润TTM
opprofit_ttm = Factor('OPPROFIT_TTM', get_TTM('OperatingProfit', 'IS'),
                      pd.to_datetime('2017-07-25'), desc='营业利润TTM')
# 销售费用TTM
opexp_ttm = Factor('OPEXP_TTM', get_TTM('OperatingExpense', 'IS'), pd.to_datetime('2017-07-25'),
                   desc='销售费用TTM')
# 管理费用TTM
adminexp_ttm = Factor('ADMINEXP_TTM', get_TTM('AdministrationExpense', 'IS'),
                      pd.to_datetime('2017-07-25'), desc='管理费用TTM')
# 财务费用TTM
fiexp_ttm = Factor('FIEXP_TTM', get_TTM('FinancialExpense', 'IS'), pd.to_datetime('2017-07-25'),
                   desc='财务费用TTM')
# 经营活动中的现金流净额
opnetcf_ttm = Factor('OPNETCF_TTM', get_TTM('NetOperateCashFlow', 'CFS'),
                     pd.to_datetime('2017-07-25'), desc='经营活动中的现金流净额')
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
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime)
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
        return data
    return _inner


# 总资产
TA = Factor('TA', get_BS_latest('TotalAssets'), pd.to_datetime('2017-07-25'),
            desc='总资产')
# 非流动性负债
TNCL = Factor('TNCL', get_BS_latest('TotalNonCurrentLiability'), pd.to_datetime('2017-07-25'),
              desc='非流动性负债')
# 流动性资产
TCA = Factor('TCA', get_BS_latest('TotalCurrentAssets'), pd.to_datetime('2017-07-25'),
             desc='流动性资产')
# 流动负债
TCL = Factor('TCL', get_BS_latest('TotalCurrentLiability'), pd.to_datetime('2017-07-25'),
             desc='流动负债')
# 归属母公司权益
equity = Factor('EQUITY', get_BS_latest('SEWithoutMI'), pd.to_datetime('2017-07-25'),
                desc='归属母公司权益')
# 现金
cash = Factor('CASH', get_BS_latest('CashEquivalents'), pd.to_datetime('2017-07-25'),
              desc='现金')
# --------------------------------------------------------------------------------------------------

factor_list = [ni_ttm, oprev_ttm, opprofit_ttm, opexp_ttm, adminexp_ttm, fiexp_ttm, opnetcf_ttm,
               TA, TNCL, TCA, TCL, equity, cash]

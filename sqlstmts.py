#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-05-15 16:53:32
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
本文件是专门用于存储获取数据的SQL
__version__ = 1.1
修改日期：2017-05-15
修改内容：初始化

__version__ = 1.2.0
修改日期：2017-05-31
修改内容：
    修改了获取股本数据的方式，以免很多数据都没有办法计算市值

__version__ = 1.2.1
修改日期：2017-06-01
修改内容：
    添加了获取指数行情的SQL

__version__ = 1.2.2
修改日期：2017-06-07
修改内容：
    在获取股本中添加总股本数据
'''

__version__ = '1.2.1'
# --------------------------------------------------------------------------------------------------
# 年度财务报表
# 年度利润表（利润分配表）
YIS_SQL = '''
    SELECT %s
    FROM LC_IncomeStatementAll S, SecuMain M
    WHERE
        M.CompanyCode = S.CompanyCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.AccountingStandards = 1 AND
        S.IfAdjusted not in (4, 5) AND
        S.BulletinType != 10 AND
        S.IfMerged = 1 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime)
    '''

# 资产负债表
BSS_SQL = '''
    SELECT %s
    FROM SecuMain M, LC_BalanceSheetAll S
    WHERE M.CompanyCode = S.CompanyCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.BulletinType != 10 AND
        S.IfMerged = 1 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime)
    '''

# 年度现金流量表
YCFS_SQL = '''
    SELECT %s
    FROM LC_CashFlowStatementAll S, SecuMain M
    WHERE
        M.CompanyCode = S.CompanyCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.AccountingStandards = 1 AND
        S.IfAdjusted not in (4, 5) AND
        S.BulletinType != 10 AND
        S.IfMerged = 1 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime)
    '''
# --------------------------------------------------------------------------------------------------
# 季度财务报表
# 季度利润表
QIS_SQL = '''
    SELECT %s
    FROM LC_QIncomeStatementNew S, SecuMain M
    WHERE
        M.CompanyCode = S.CompanyCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.BulletinType != 10 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime)
    '''

# 季度现金流量表
QCFS_SQL = '''
    SELECT %s
    FROM LC_QCashFlowStatementNew S, SecuMain M
    WHERE
        M.CompanyCode = S.CompanyCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.BulletinType != 10 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime)
    '''

# --------------------------------------------------------------------------------------------------
# 其他财务数据表
# 股本数据，获取总股本和流通股本
SN_SQL = '''
    SELECT S.TotalShares, S.NonResiSharesJY, S.EndDate
    FROM SecuMain M, LC_ShareStru S
    WHERE M.CompanyCode = S.CompanyCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1
    '''

# 分红进度表
DIV_SQL = '''
    SELECT %s
    FROM SecuMain M, LC_DividendProgress S
    WHERE
        M.InnerCode = S.InnerCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.InfoPubType = 40 AND
        S.Process = 3131 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfopubDate <= CAST(\'{end_time}\' AS datetime)
    '''

# --------------------------------------------------------------------------------------------------
# 指数相关数据表
# 指数成分
INDEX_SQL = '''
    SELECT  M2.SecuCode, S.EndDate
    FROM SecuMain M, LC_IndexComponentsWeight S, SecuMain M2
    WHERE
        M.InnerCode = S.IndexCode AND
        M2.InnerCode = S.InnerCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuCategory = 4 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.EndDate <= CAST(\'{end_time}\' AS datetime)
    '''

# 获取所有A股成份
AUNIVERSE_SQL = '''
    SELECT SecuCode
    FROM SecuMain
    WHERE
        SecuCategory = 1 AND
        SecuMarket in (83, 90) AND
        ListedState != 9
    '''


# 获取中信行业分类
# 中信行业的更新和对新股进行分类都有一些时间上的滞后
ZXIND_SQL = '''
    SELECT S.FirstIndustryName, S.InfoPublDate, M.SecuCode
    FROM LC_exgIndustry S, SecuMain M
    WHERE S.CompanyCOde = M.CompanyCode AND
        S.Standard = 3 AND
        M.SecuCategory = 1 AND
        M.SecuMarket in (90, 83)
    ORDER BY M.Secucode, S.Standard, S.InfoPublDate ASC
    '''


# --------------------------------------------------------------------------------------------------
# 股票行情相关表
# 获取股票行情
QUOTE_SQL = '''
    SELECT %s
    FROM QT_DailyQuote S, SecuMain M
    WHERE
        S.InnerCode = M.InnerCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90) AND
        S.TradingDay <= CAST(\'{end_time}\' as datetime) AND
        S.TradingDay >= CAST(\'{start_time}\' as datetime) AND
        M.SecuCategory = 1
    ORDER BY S.TradingDay ASC
    '''

# 获取复权因子
ADJFACTOR_SQL = '''
    SELECT A.ExDiviDate, A.RatioAdjustingFactor
    FROM QT_AdjustingFactor A, SecuMain M
    WHERE
        A.InnerCode = M.InnerCode AND
        M.SecuCode = \'{code}\' AND
        M.secuMarket in (83, 90)
    ORDER BY A.ExDiviDate ASC
    '''

# 获取股票戴帽摘帽情况
ST_SQL = '''
    SELECT S.InfoPublDate, S.SecurityAbbr
    FROM LC_SpecialTrade S, SecuMain M
    WHERE
        S.InnerCode = M.InnerCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90)
    '''

# 获取指数行情数据
INDEX_QUOTE_SQL = '''
    SELECT %s
    FROM QT_IndexQuote S, SecuMain M
    WHERE S.InnerCode = M.InnerCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuCategory = 4 AND
        S.TradingDay >= \'{start_time}\' AND
        S.TradingDay <= \'{end_time}\'
    ORDER BY S.TradingDay ASC
'''
# --------------------------------------------------------------------------------------------------
# 研究报告相关数据
# 一致预期目标价
TARGET_PRICE_SQL = '''
    SELECT TARGET_PRICE, CON_DATE
    FROM CON_FORECAST_SCHEDULE
    WHERE
        CON_DATE >= CAST(\'{start_time}\' AS datetime) AND
        CON_DATE <= CAST(\'{end_time}\' AS datetime) AND
        STOCK_CODE = \'{code}\'
'''

# --------------------------------------------------------------------------------------------------
# 集合现有的所有基础SQL
BASIC_SQLs = {'QIS': QIS_SQL, 'YIS': YIS_SQL, 'QCFS': QCFS_SQL, 'YCFS': YCFS_SQL,
              'BSS': BSS_SQL, 'SN': SN_SQL, 'INDEX_CONSTITUENTS': INDEX_SQL, 'DIV': DIV_SQL,
              'QUOTE': QUOTE_SQL, 'ADJ_FACTOR': ADJFACTOR_SQL, 'A_UNIVERSE': AUNIVERSE_SQL,
              'ST_TAG': ST_SQL, 'ZX_IND': ZXIND_SQL, 'TARGET_PRICE': TARGET_PRICE_SQL,
              'INDEX_QUOTE': INDEX_QUOTE_SQL}

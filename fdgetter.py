#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-29 16:04:27
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
该模块主要用于从数据库中获取数据，进行简单的加工，并转化为DataFrame的格式
__version__ = 1.0
修改日期：2017-03-29
修改内容：
    初始化

__version__ = 1.1
修改日期：2017-04-09
修改内容：
    1. 添加获取日级别行情数据和复权因子的SQL
    2. 为format_sql添加默认起始时间值
    3. 添加连接数据库的异常处理

__version__ = 1.2
修改日期：2017-04-12
修改内容：
    1. 添加了获取A股所有成分的SQL
    2. 为get_db_data和format_sql添加新的默认参数

__version__ = 1.2.1
修改日期：2017-05-05
修改内容：
    修改了获取股票股份的方式，将其获取的数据固定下来，获取流通股份
    将gen_sql_cols的返回的结果设置为TempRes(namedTuple类型)
    添加获取中信行业分类的SQL
'''
__version__ = '1.2.1'

from collections import namedtuple
import datatoolkits
from decimal import Decimal
import functools
import numpy as np
import pandas as pd
import SQLserver

# --------------------------------------------------------------------------------------------------
# 设置数据库常量
try:
    jydb = SQLserver.SQLserver(DATABASE='jydb', SERVER='128.6.5.18', UID='jydb', PWD='jydb')
except Exception as e:  # 当前环境下没有连接数据库
    jydb = None
    print(e)

# --------------------------------------------------------------------------------------------------
# 设置常用的sql
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
# 股本数据，获取流通股本
SN_SQL = '''
    SELECT S.NonRestrictedShares, S.EndDate
    FROM SecuMain M, LC_ShareStru S
    WHERE M.CompanyCode = S.CompanyCode AND
        M.SecuCode = \'{code}\' AND
        M.SecuMarket in (83, 90) AND
        M.SecuCategory = 1 AND
        S.EndDate >= CAST(\'{start_time}\' AS datetime) AND
        S.InfoPublDate <= CAST(\'{end_time}\' AS datetime)
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
# 获取所有A股成份
AUNIVERSE_SQL = '''
    SELECT SecuCode
    FROM SecuMain
    WHERE
        SecuCategory = 1 AND
        SecuMarket in (83, 90) AND
        ListedState != 9
    '''

# 获取股票戴帽摘帽情况
ST_SQL = '''
    SELECT S.InfoPublDate, S.SecurityAbbr
    FROM LC_SpecialTrade S, SecuMain M
    WHERE
        S.InnerCode = M.InnerCode AND
        M.SecuCode = {code} AND
        M.SecuMarket in (83, 90)
    '''

# 获取中信行业分类
ZXIND_SQL = '''
    SELECT S.FirstIndustryName, S.InfoPublDate, M.SecuCode
    FROM LC_exgIndustry S, SecuMain M
    WHERE S.CompanyCOde = M.CompanyCode AND
        S.Standard = 3 AND
        M.SecuCategory = 1 AND
        M.SecuMarket in (90, 83)
    ORDER BY M.Secucode, S.Standard, S.InfoPublDate ASC
    '''

# 集合现有的所有基础SQL
BASIC_SQLs = {'QIS': QIS_SQL, 'YIS': YIS_SQL, 'QCFS': QCFS_SQL, 'YCFS': YCFS_SQL,
              'BSS': BSS_SQL, 'SN': SN_SQL, 'INDEX_CONSTITUENTS': INDEX_SQL, 'DIV': DIV_SQL,
              'QUOTE': QUOTE_SQL, 'ADJ_FACTOR': ADJFACTOR_SQL, 'A_UNIVERSE': AUNIVERSE_SQL,
              'ST_TAG': ST_SQL, 'ZX_IND': ZXIND_SQL}
# 添加SQL模板的其他操作
SQLFILE_PATH = r"F:\GeneralLib\CONST_DATAS\sql_templates.pickle"
# 获取当前的SQL模板


def get_sql_template(path=SQLFILE_PATH):
    try:
        res = datatoolkits.load_pickle(path)
    except FileNotFoundError:
        res = BASIC_SQLs
        datatoolkits.dump_pickle(res, path)
    return res

# 向模板中添加其他模板，并存储到文件中


def add_template(templates, name, sql, path=SQLFILE_PATH):
    '''
    向模板中添加内容，并存储
    '''
    templates[name] = sql
    datatoolkits.dump_pickle(templates, path)
    return templates


def del_template(templates, name, path=SQLFILE_PATH):
    '''
    删除模板中的内容，并存储
    注：删除操作不会对原数据造成影响
    '''
    assert name in templates, 'Error, valid sql names are {}'.format(list(templates.keys()))
    templates_cp = templates.copy()
    del templates_cp[name]
    return templates_cp


def modify_template(templates, name, sql, path=SQLFILE_PATH):
    '''
    修改模板中已有的模板，并存储
    '''
    assert name in templates, 'Error, valid sql names are {}'.format(list(templates.keys()))
    templates[name] = sql
    datatoolkits.dump_pickle(templates, path)
    return templates


def reset_templates(path=SQLFILE_PATH):
    '''
    将模板重置为文件中的初始SQL模板
    '''
    datatoolkits.dump_pickle(BASIC_SQLs, path)
    # return BASIC_SQLs


# 设置基础常量
SQLs = get_sql_template()
# --------------------------------------------------------------------------------------------------
# 其他函数


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
    if len(raw_data) == 0:
        return datatoolkits.gen_df(col_names)

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


def gen_sql_cols(cols, sql_type):
    '''
    利用当前有的sql模板生成带上数据列的sql模板
    @param:
        cols: 字典类型{sql_col: df_col}，其中sql_col为数据库中每列数据对应的列列名，需要通过查询
            数据库的字典，df_col为最终df对应的数据列的列名
        sql_type: 使用的sql的类型，必须为SQLs中包含的类型
    @return:
        sql_template: 生成的sql模板
        cols: 用于传入clean_data的数据列的列名
    '''
    sql_cols, df_cols = zip(*cols.items())
    suffix = 'S.'
    sql_cols = ','.join([suffix + s for s in sql_cols])
    assert sql_type in SQLs, 'Error, valid sql types are {}'.format(list(SQLs.keys()))
    sql = SQLs[sql_type]
    sql = sql % sql_cols
    TempRes = namedtuple('TempRes', 'sql cols')
    res = TempRes(sql=sql, cols=df_cols)
    return res


def format_sql(sql, code='', start_time=pd.to_datetime('1990-01-01'),
               end_time=pd.to_datetime('1990-01-01')):
    '''
    对应的股票和时间区间生成SQL
    @param:
        sql: SQL模板
        code: 股票代码
        start_time: 时间区间起点，可以为datetime或者str格式，添加默认值，考虑到有些SQL不需要提供
            起始的时间，而str.format对于字符串中没有提供的参数不会报错，添加默认值可以提高通用性
        end_time: 时间区间终点，格式同上，原因同上
    @return:
        代入参数的sql
    '''
    if not isinstance(start_time, str):
        start_time = start_time.strftime('%Y-%m-%d')
    if not isinstance(end_time, str):
        end_time = end_time.strftime('%Y-%m-%d')
    code = datatoolkits.drop_suffix(code)
    return sql.format(code=code, start_time=start_time, end_time=end_time)


def get_db_data(sql, code='', start_time=pd.to_datetime('1990-01-01'),
                end_time=pd.to_datetime('1990-01-01'), cols=('',), db=jydb, add_stockcode=True):
    '''
    从数据库中取出数据
    @param:
        sql: SQL模板，即数据列已经填充完毕后的SQL模板或者其他自定义的sql
        code: 股票代码
        start_time: 时间区间起点，可以为datetime或者str格式
        end_time: 时间区间终点，格式同上
        cols: 数据列对应的列名，应当为gen_sql_cols返回的参数
        db: 使用的数据库源，默认为jydb（模块内部提供）
        add_stockcode: 是否在数据结果中添加股票代码，默认为添加（True）
    @return:
        从数据库中取出经过基本处理的DataFrame数据
    '''
    sql = format_sql(sql, code, start_time, end_time)
    data = db.fetchall(sql)
    data_cleaned = clean_data(data, cols)
    if add_stockcode:
        data_cleaned['code'] = len(data_cleaned) * [code]
    return data_cleaned


def combine(datas, on, how='outer'):
    '''
    将同一支股票的其他数据合并到一个DataFrame中
    @param:
        datas: 需要合并的数据，要求为可迭代类型，内容为DataFrame，且均包含on参数中提供的列
        on: 合并的主键
        how: 合并方式，默认为'outer'，其他参数与pd.merge相同
    @return:
        合并后的数据，DataFrame形式
    '''
    res = functools.reduce(lambda x, y: pd.merge(x, y, on=on, how=how), datas)
    return res


if __name__ == '__main__':
    qis_sql = gen_sql_cols({'NetProfit': 'ni', 'EndDate': 'rpt_date',
                            'InfoPublDate': 'update_time'}, 'QIS')
    bss_sql = gen_sql_cols({'TotalAssets': 'total_asset', 'ShortTermLoan': 'ST_loan',
                            'EndDate': 'rpt_date', 'InfoPublDate': 'update_time'}, 'BSS')
    code = '000001.SZ'
    data = list()
    for sql, col in (qis_sql, bss_sql):
        tmp_data = get_db_data(sql, code, '2009-01-01', '2016-01-01', col)
        data.append(tmp_data)
    data = combine(data, on=('code', 'rpt_date', 'update_time'))

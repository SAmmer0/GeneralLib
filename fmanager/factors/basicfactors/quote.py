#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-18 14:35:34
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
行情类因子
__version__: 1.0.0
修改日期：2017-07-20
修改内容：
    初始化，添加基本因子
'''
__version__ = '1.0.0'

import datatoolkits
import dateshandle
import fdgetter
import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
import pdb
from fmanager.const import START_TIME
from fmanager.factors.utils import (check_indexorder, Factor, check_duplicate_factorname,
                                    checkdata_completeness)
from fmanager.factors.query import query
# --------------------------------------------------------------------------------------------------
# 常量和功能函数
NAME = 'quote'

# --------------------------------------------------------------------------------------------------
# 功能函数


def get_factor_dict():
    res = dict()
    for f in factor_list:
        res[f.name] = {'rel_path': NAME + '\\' + f.name, 'factor': f}
    return res


factor_list = []
# --------------------------------------------------------------------------------------------------
# 获取行情相关数据


def get_quote(data_type):
    '''
    母函数，用于生成获取给定行情数据的函数
    '''
    sql = '''
        SELECT S.TradingDay, data_type, M.Secucode
        FROM QT_DailyQuote S, SecuMain M
        WHERE
            S.InnerCode = M.InnerCode AND
            M.SecuMarket in (83, 90) AND
            S.TradingDay <= CAST(\'{end_time}\' as datetime) AND
            S.TradingDay >= CAST(\'{start_time}\' as datetime) AND
            M.SecuCategory = 1
        ORDER BY S.TradingDay ASC, M.Secucode ASC
        '''
    price_filter = ['openprice', 'highprice', 'lowprice']
    if data_type.lower() not in price_filter:
        transed_sql = sql.replace('data_type', 'S.' + data_type)
        cols = ('time', 'data', 'code')
    else:
        transed_sql = sql.replace('data_type', 'S.PrevClosePrice, S.' + data_type)
        cols = ('time', 'prevclose', 'data', 'code')

    def _inner(universe, start_time, end_time):
        data = fdgetter.get_db_data(transed_sql, cols=cols, start_time=start_time,
                                    end_time=end_time, add_stockcode=False)
        data['code'] = data.code.apply(datatoolkits.add_suffix)
        if len(data.columns) == 4:
            data.loc[data.data == 0, 'data'] = data['prevclose']
            data.drop('prevclose', inplace=True, axis=1)
        data = data.pivot_table('data', index='time', columns='code')
        data = data.loc[:, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return _inner


# 收盘价
factor_list.append(Factor('CLOSE', get_quote('ClosePrice'), pd.to_datetime('2017-07-20')))
# 开盘价
factor_list.append(Factor('OPEN', get_quote('OpenPrice'), pd.to_datetime('2017-07-20')))
# 最高价
factor_list.append(Factor('HIGH', get_quote('HighPrice'), pd.to_datetime('2017-07-20')))
# 最低价
factor_list.append(Factor('LOW', get_quote('LowPrice'), pd.to_datetime('2017-07-20')))
# 成交量
factor_list.append(Factor('TO_VOLUME', get_quote('TurnoverVolume'), pd.to_datetime('2017-07-20'),
                          desc='单位为股'))
# 成交额
factor_list.append(Factor('TO_VALUE', get_quote('TurnoverValue'), pd.to_datetime('2017-07-20'),
                          desc='单位为元'))

factor_list.append(Factor('PREV_CLOSE', get_quote('PrevClosePrice'), pd.to_datetime('2017-10-19')))

# --------------------------------------------------------------------------------------------------
# 复权因子


def get_adjfactor(universe, start_time, end_time):
    '''
    股票的复权因子
    '''
    sql = '''
        SELECT A.ExDiviDate, A.RatioAdjustingFactor, M.SecuCode
        FROM QT_AdjustingFactor A, SecuMain M
        WHERE
            A.InnerCode = M.InnerCode AND
            M.secuMarket in (83, 90) AND
            M.SECUCATEGORY = 1
        ORDER BY M.SecuCode ASC, A.ExDiviDate ASC
        '''
    data = fdgetter.get_db_data(sql, cols=('time', 'data', 'code'), add_stockcode=False)
    data['code'] = data.code.apply(datatoolkits.add_suffix)
    by_code = data.groupby('code')
    tds = dateshandle.get_tds(start_time, end_time)
    data = by_code.apply(datatoolkits.map_data, days=tds, fromNowOn=True,
                         fillna={'code': lambda x: x.code.iloc[0], 'data': lambda x: 1})
    data = data.reset_index(drop=True)
    data = data.pivot_table('data', index='time', columns='code')
    data = data.loc[:, sorted(universe)].fillna(1)    # 因为新股大多数情况下没有分红记录
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('ADJ_FACTOR', get_adjfactor, pd.to_datetime('2017-07-21')))
# --------------------------------------------------------------------------------------------------
# 股本


def get_shares(share_type):
    '''
    母函数，用于生成获取给定类型股本的函数
    '''
    sql = '''
        SELECT S.share_type, S.EndDate, S.InfoPublDate, M.SecuCode
        FROM SecuMain M, LC_ShareStru S
        WHERE M.CompanyCode = S.CompanyCode AND
            M.SecuMarket in (83, 90) AND
            M.SecuCategory = 1  AND
            S.InfoPublDate >= (SELECT TOP(1) S2.CHANGEDATE
                          FROM LC_ListStatus S2
                          WHERE
                              S2.INNERCODE = M.INNERCODE AND
                              S2.ChangeType = 1)
        '''
    transed_sql = sql.replace('share_type', share_type)

    def _inner(universe, start_time, end_time):
        data = fdgetter.get_db_data(transed_sql, cols=('data', 'end_time', 'publ_time', 'code'),
                                    add_stockcode=False)
        data['code'] = data.code.apply(datatoolkits.add_suffix)
        data['time'] = data.publ_time.fillna(data.end_time)
        data = data.drop(['end_time', 'publ_time'], axis=1).drop_duplicates(['code', 'time']).\
            sort_values(['code', 'time'])   # 此处假设若时间相同则股本数量相同
        by_code = data.groupby('code')
        tds = dateshandle.get_tds(start_time, end_time)
        data = by_code.apply(datatoolkits.map_data, days=tds, fromNowOn=True,
                             fillna={'code': lambda x: x.code.iloc[0]})
        data = data.reset_index(drop=True)
        data = data.pivot_table('data', index='time', columns='code')
        data = data.loc[:, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return _inner


# 流通股本
factor_list.append(Factor('FLOAT_SHARE', get_shares('NonResiSharesJY'),
                          pd.to_datetime('2017-07-21')))
# 总股本
factor_list.append(Factor('TOTAL_SHARE', get_shares('TotalShares'), pd.to_datetime('2017-07-21')))
# --------------------------------------------------------------------------------------------------
# 股票市值：包含总市值和流通市值


def get_mktvalue(share_factor_name):
    '''
    母函数，用于生成计算市值因子的函数
    '''
    def _inner(universe, start_time, end_time):
        share_data = query(share_factor_name, (start_time, end_time))
        close_data = query('CLOSE', (start_time, end_time))
        assert len(share_data) == len(close_data), "Error, basic data length does not  match! " + \
            "share data = {sd_len}, while close data = {cd_len}".format(sd_len=len(share_data),
                                                                        cd_len=len(close_data))
        res = share_data * close_data
        res = res.loc[:, sorted(universe)]
        assert checkdata_completeness(res, start_time, end_time), "Error, data missed!"
        return res
    return _inner


factor_list.append(Factor('TOTAL_MKTVALUE', get_mktvalue('TOTAL_SHARE'), pd.to_datetime('2017-07-24'),
                          desc="使用收盘价计算", dependency=['TOTAL_SHARE', 'CLOSE']))
factor_list.append(Factor('FLOAT_MKTVALUE', get_mktvalue('FLOAT_SHARE'), pd.to_datetime('2017-07-24'),
                          desc="使用收盘价计算", dependency=['FLOAT_SHARE', 'CLOSE']))

# --------------------------------------------------------------------------------------------------
# 后复权价格


def gen_adjprice(price_type):
    '''
    用于生成计算后复权数据的函数

    Parameter
    ---------
    price_type: string
        使用的价格类型，要求为原始价格因子类型，目前支持['OPEN', 'CLOSE', 'HIGH', 'LOW']
    '''
    valids = ['OPEN', 'CLOSE', 'HIGH', 'LOW']
    assert price_type in valids,\
        ValueError('Invalid price type, valid types are {vld}, you provide {yp}'.
                   format(vld=valids, yp=price_type))

    def inner(universe, start_time, end_time):
        adj_factor = query('ADJ_FACTOR', (start_time, end_time))
        data = query(price_type, (start_time, end_time))
        assert len(adj_factor) == len(data), "Error, basic data length does not  match! " + \
            "adj_factor data = {sd_len}, while close data = {cd_len}".format(sd_len=len(adj_factor),
                                                                             cd_len=len(data))
        res = adj_factor * data
        res = res.loc[:, sorted(universe)]
        assert checkdata_completeness(res, start_time, end_time), "Error, data missed!"
        return res
    return inner


factor_list.append(Factor('ADJ_CLOSE', gen_adjprice('CLOSE'), pd.to_datetime('2017-07-24'),
                          dependency=['CLOSE', 'ADJ_FACTOR']))

# --------------------------------------------------------------------------------------------------
# 日收益率


def get_dailyret(universe, start_time, end_time):
    '''
    获取日收益率，使用后复权收盘价计算
    '''
    new_start = pd.to_datetime(start_time) - pd.Timedelta('30 day')
    data = query('ADJ_CLOSE', (new_start, end_time))
    data = data.pct_change()
    mask = data.index >= start_time
    data = data.loc[mask, sorted(universe)]
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('DAILY_RET', get_dailyret, pd.to_datetime('2017-07-24'),
                          dependency=['ADJ_CLOSE']))
# --------------------------------------------------------------------------------------------------
# 换手率


def get_torate(universe, start_time, end_time):
    '''
    获取换手率，使用当天交易量/流通股数来计算
    '''
    volume = query('TO_VOLUME', (start_time, end_time))
    float_shares = query('FLOAT_SHARE', (start_time, end_time))
    res = volume / float_shares
    res = res.loc[:, sorted(universe)]
    assert checkdata_completeness(res, start_time, end_time), "Error, data missed!"
    return res


factor_list.append(Factor('TO_RATE', get_torate, pd.to_datetime('2017-07-24'),
                          dependency=['TO_VOLUME', 'FLOAT_SHARE']))


# 过去一个月日均换手率
def get_avgtorate(universe, start_time, end_time):
    '''
    指过去20个交易日平均换手率
    '''
    start_time = pd.to_datetime(start_time)
    new_start = start_time - pd.Timedelta('60 day')
    daily_torate = query('TO_RATE', (new_start, end_time))
    data = daily_torate.rolling(20, min_periods=20).mean().dropna(how='all')
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask, sorted(universe)]
    if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
        assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('TOAVG_1M', get_avgtorate, pd.to_datetime('2017-08-02'),
                          dependency=['TO_RATE'], desc='过去一个月（20交易日）日均换手率'))


def get_sto(category):
    '''
    母函数，用于生成BARRA的几个换手率因子，包括STOM, STOA, STOMQ

    Parameter
    ---------
    category: string
        分类，可选的类别有STOM, STOQ, STOA，分别表示月度、季度和年度

    Return
    ------
    out: function
    '''
    month_days = 21
    month_num_map = {'STOM': 1, 'STOQ': 3, 'STOA': 12}
    month_num = month_num_map[category]
    offset = month_days * month_num

    def inner(universe, start_time, end_time):
        start_time = pd.to_datetime(start_time)
        threshold = 10e-6
        new_start = dateshandle.tds_shift(start_time, offset)
        daily_torate = query('TO_RATE', (new_start, end_time))
        data = daily_torate.rolling(offset, min_periods=offset).sum().dropna(how='all')
        data[data <= threshold] = np.NaN
        data = data / month_num
        data = np.log(data)
        mask = (data.index >= start_time) & (data.index <= end_time)
        data = data.loc[mask, sorted(universe)]
        if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
            assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return inner


factor_list.append(Factor('STOM', get_sto('STOM'), pd.to_datetime('2017-10-27'),
                          dependency=['TO_RATE'], desc='BARRA STOM月换手率因子'))
factor_list.append(Factor('STOQ', get_sto('STOQ'), pd.to_datetime('2017-10-27'),
                          dependency=['TO_RATE'], desc='BARRA STOQ季度换手率因子'))
factor_list.append(Factor('STOA', get_sto('STOA'), pd.to_datetime('2017-10-27'),
                          dependency=['TO_RATE'], desc='BARRA STOA年度换手率因子'))

# --------------------------------------------------------------------------------------------------
# 对数市值


def get_lnfloatmktv(universe, start_time, end_time):
    '''
    对数流通市值
    '''
    fmktv = query('FLOAT_MKTVALUE', (start_time, end_time))
    data = np.log(fmktv)
    data = data.loc[:, sorted(universe)]
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


def get_lntotalmktv(universe, start_time, end_time):
    '''
    对数总市值
    '''
    tmktv = query('TOTAL_MKTVALUE', (start_time, end_time))
    data = np.log(tmktv)
    data = data.loc[:, sorted(universe)]
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


def get_nonlinearmktv(universe, start_time, end_time):
    '''
    非线性市值
    '''
    lnmktv = query('LN_TMKV', (start_time, end_time))

    def get_nlsize(df):
        # 市值3次方然后通过OLS来获取与市值正交的残差
        raw_index = df.index
        df = df.dropna()
        nonlsize = np.power(df, 3)
        mod = OLS(nonlsize, add_constant(df))
        mod_res = mod.fit()
        return mod_res.resid.reindex(raw_index)
    data = lnmktv.apply(get_nlsize, axis=1)
    data = data.loc[:, sorted(universe)]
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('LN_FMKV', get_lnfloatmktv, pd.to_datetime('2017-08-02'),
                          dependency=['FLOAT_MKTVALUE'], desc='对数流通市值'))
factor_list.append(Factor('LN_TMKV', get_lntotalmktv, pd.to_datetime('2017-10-17'),
                          dependency=['TOTAL_MKTVALUE'], desc='对数总市值'))
factor_list.append(Factor('NLSIZE', get_nonlinearmktv, pd.to_datetime('2017-10-19'),
                          dependency=['LN_TMKV'], desc='非线性市值'))
# --------------------------------------------------------------------------------------------------
# 指数行情


def gen_indexquotegetter(index_code, price_type='ClosePrice'):
    '''
    母函数，用于生成获取指数行情数据的函数

    Parameter
    ---------
    index_code: string
        指数代码
    price_type: string
        行情类型
    '''
    sql_tmplate = '''
            SELECT S.price_type, S.TradingDay
            FROM QT_IndexQuote S, secuMain M
            WHERE
                S.innerCode = M.innerCode AND
                M.secuCode = 'code' AND
                M.secuCategory = 4 AND
                S.TradingDay >= \'{start_time}\' AND
                S.TradingDay <= \'{end_time}\'
            ORDER BY S.TradingDay ASC
        '''

    def _inner(universe, start_time, end_time):
        '''
        上证综指
        '''
        sql = sql_tmplate.replace('code', index_code).replace('price_type', price_type)
        data = fdgetter.get_db_data(sql, cols=('close', 'time'), start_time=start_time,
                                    end_time=end_time, add_stockcode=False)
        # pdb.set_trace()
        data = pd.DataFrame(np.repeat([data.close.values], len(universe), axis=0).T,
                            index=data.time, columns=sorted(universe))
        checkdata_completeness(data, start_time, end_time)
        check_indexorder(data)
        return data
    return _inner


# 上证综指
factor_list.append(Factor('SSEC_CLOSE', gen_indexquotegetter('000001'), pd.to_datetime('2017-08-14'),
                          desc='上证综指收盘价'))
factor_list.append(Factor('SSEC_OPEN', gen_indexquotegetter('000001', 'OpenPrice'),
                          pd.to_datetime('2017-11-30'), desc='上证综指开盘价'))
# 上证50
factor_list.append(Factor('SSE50_CLOSE', gen_indexquotegetter('000016'), pd.to_datetime('2017-08-14'),
                          desc='上证50收盘价'))
factor_list.append(Factor('SSE50_OPEN', gen_indexquotegetter('000016', 'OpenPrice'),
                          pd.to_datetime('2017-11-30'), desc='上证50开盘价'))
# 中证500
factor_list.append(Factor('CS500_CLOSE', gen_indexquotegetter('000905'), pd.to_datetime('2017-08-14'),
                          desc='中证500收盘价'))
factor_list.append(Factor('CS500_OPEN', gen_indexquotegetter('000905', 'OpenPrice'),
                          pd.to_datetime('2017-11-30'), desc='中证500开盘价'))
# 沪深300
factor_list.append(Factor('SSZ300_CLOSE', gen_indexquotegetter('000300'), pd.to_datetime('2017-08-14'),
                          desc='沪深300收盘价'))
factor_list.append(Factor('SSZ300_OPEN', gen_indexquotegetter('000300', 'OpenPrice'),
                          pd.to_datetime('2017-11-30'), desc='沪深300开盘价'))
# 中证全指
factor_list.append(Factor('CSI985_CLOSE', gen_indexquotegetter('000985'), pd.to_datetime('2017-08-14'),
                          desc='中证全指收盘价'))
factor_list.append(Factor('CSI985_OPEN', gen_indexquotegetter('000985', 'OpenPrice'),
                          pd.to_datetime('2017-11-30'), desc='中证全指开盘价'))
# --------------------------------------------------------------------------------------------------


check_duplicate_factorname(factor_list, __name__)

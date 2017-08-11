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
from ...const import START_TIME
from ..utils import check_indexorder, Factor, check_duplicate_factorname, checkdata_completeness
from ..query import query
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
close_price = Factor('CLOSE', get_quote('ClosePrice'), pd.to_datetime('2017-07-20'))
# 开盘价
open_price = Factor('OPEN', get_quote('OpenPrice'), pd.to_datetime('2017-07-20'))
# 最高价
high_price = Factor('HIGH', get_quote('HighPrice'), pd.to_datetime('2017-07-20'))
# 最低价
low_price = Factor('LOW', get_quote('LowPrice'), pd.to_datetime('2017-07-20'))
# 成交量
to_volume = Factor('TO_VOLUME', get_quote('TurnoverVolume'), pd.to_datetime('2017-07-20'),
                   desc='单位为股')
# 成交额
to_value = Factor('TO_VALUE', get_quote('TurnoverValue'), pd.to_datetime('2017-07-20'),
                  desc='单位为元')

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
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


adj_factor = Factor('ADJ_FACTOR', get_adjfactor, pd.to_datetime('2017-07-21'))
# --------------------------------------------------------------------------------------------------
# 股本


def get_shares(share_type):
    '''
    母函数，用于生成获取给定类型股本的函数
    '''
    sql = '''
        SELECT S.share_type, S.EndDate, M.SecuCode
        FROM SecuMain M, LC_ShareStru S
        WHERE M.CompanyCode = S.CompanyCode AND
            M.SecuMarket in (83, 90) AND
            M.SecuCategory = 1
        '''
    transed_sql = sql.replace('share_type', share_type)

    def _inner(universe, start_time, end_time):
        data = fdgetter.get_db_data(transed_sql, cols=('data', 'time', 'code'), add_stockcode=False)
        data['code'] = data.code.apply(datatoolkits.add_suffix)
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
float_shares = Factor('FLOAT_SHARE', get_shares('NonResiSharesJY'), pd.to_datetime('2017-07-21'))
# 总股本
total_shares = Factor('TOTAL_SHARE', get_shares('TotalShares'), pd.to_datetime('2017-07-21'))
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


total_mktvalue = Factor('TOTAL_MKTVALUE', get_mktvalue('TOTAL_SHARE'), pd.to_datetime('2017-07-24'),
                        desc="使用收盘价计算", dependency=['TOTAL_SHARE', 'CLOSE'])
float_mktvalue = Factor('FLOAT_MKTVALUE', get_mktvalue('FLOAT_SHARE'), pd.to_datetime('2017-07-24'),
                        desc="使用收盘价计算", dependency=['FLOAT_SHARE', 'CLOSE'])

# --------------------------------------------------------------------------------------------------
# 后复权价格


def get_adjclose(universe, start_time, end_time):
    '''
    获取后复权收盘价
    '''
    adj_factor = query('ADJ_FACTOR', (start_time, end_time))
    close_data = query('CLOSE', (start_time, end_time))
    assert len(adj_factor) == len(close_data), "Error, basic data length does not  match! " + \
        "adj_factor data = {sd_len}, while close data = {cd_len}".format(sd_len=len(adj_factor),
                                                                         cd_len=len(close_data))
    res = adj_factor * close_data
    res = res.loc[:, sorted(universe)]
    assert checkdata_completeness(res, start_time, end_time), "Error, data missed!"
    return res


adj_close = Factor('ADJ_CLOSE', get_adjclose, pd.to_datetime('2017-07-24'),
                   dependency=['CLOSE', 'ADJ_FACTOR'])

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


daily_ret = Factor('DAILY_RET', get_dailyret, pd.to_datetime('2017-07-24'),
                   dependency=['ADJ_CLOSE'])
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


to_rate = Factor('TO_RATE', get_torate, pd.to_datetime('2017-07-24'),
                 dependency=['TO_VOLUME', 'FLOAT_SHARE'])


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


avg_torate = Factor('TOAVG_1M', get_avgtorate, pd.to_datetime('2017-08-02'),
                    dependency=['TO_RATE'], desc='过去一个月（20交易日）日均换手率')

# --------------------------------------------------------------------------------------------------
# 对数市值


def get_lnfloatmktv(universe, start_time, end_time):
    '''
    对数市值
    '''
    fmktv = query('FLOAT_MKTVALUE', (start_time, end_time))
    data = np.log(fmktv)
    data = data.loc[:, sorted(universe)]
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


ln_flmv = Factor('LN_FMKV', get_lnfloatmktv, pd.to_datetime('2017-08-02'),
                 dependency=['FLOAT_MKTVALUE'], desc='对数市值')
# --------------------------------------------------------------------------------------------------

factor_list = [close_price, open_price, high_price, low_price, to_value, to_volume, adj_factor,
               float_shares, total_shares, total_mktvalue, float_mktvalue, adj_close,
               daily_ret, to_rate, ln_flmv, avg_torate]
check_duplicate_factorname(factor_list, __name__)

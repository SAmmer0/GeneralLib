#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-13 14:47:20
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
一般性因子，当前初始化，要纳入中信行业、上市状态、特殊处理、交易状态、上证50成份、沪深300成份、
中证500成份

修改日期：2017-07-13
修改内容：
    初始化
'''
import datatoolkits
import dateshandle
import fdgetter
import fdmutils
import pandas as pd
import pdb

from fmanager.database import NaS
from fmanager.factors.utils import (Factor, ZXIND_TRANS_DICT, check_duplicate_factorname,
                                    checkdata_completeness)
from fmanager.const import START_TIME

# --------------------------------------------------------------------------------------------------
# 常量和功能函数
NAME = 'general'


def get_factor_dict():
    res = dict()
    for f in factor_list:
        res[f.name] = {'rel_path': NAME + '\\' + f.name, 'factor': f}
    return res
# --------------------------------------------------------------------------------------------------
# 中信行业因子


def get_zxind(universe, start_time, end_time):
    '''
    获取中信行业的数据，并映射到每个交易日中
    '''
    ind_data = fdgetter.get_db_data(fdgetter.BASIC_SQLs['ZX_IND'], cols=('ind', 'time', 'code'),
                                    add_stockcode=False)
    ind_data['ind'] = ind_data.ind.map(ZXIND_TRANS_DICT)
    ind_data['code'] = ind_data.code.apply(datatoolkits.add_suffix)
    # pdb.set_trace()
    tds = dateshandle.get_tds(start_time, end_time)
    by_code = ind_data.groupby('code')
    ind_data = by_code.apply(datatoolkits.map_data, days=tds,
                             fillna={'ind': lambda x: NaS, 'code': lambda x: x.code.iloc[0]})
    ind_data = ind_data.reset_index(drop=True).set_index(['time', 'code'])
    ind_data = ind_data.loc[:, 'ind'].unstack()
    ind_data = ind_data.loc[:, sorted(universe)].dropna(axis=0, how='all').fillna(NaS)
    assert checkdata_completeness(ind_data, start_time, end_time), "Error, data missed!"
    return ind_data


zx_ind_factor = Factor('ZX_IND', get_zxind, pd.to_datetime('2017-07-13'), data_type='S30')

# --------------------------------------------------------------------------------------------------
# 上市状态因子，1表示正常上市，2表示暂停上市，3表示退市整理，4表示终止上市


def get_liststatus(universe, start_time, end_time):
    '''
    获取股票的上市状态
    '''
    ls_data = fdgetter.get_db_data(fdgetter.BASIC_SQLs['LIST_STATUS'],
                                   cols=('code', 'list_status', 'time'), add_stockcode=False)
    ls_map = {1: 1, 2: 2, 3: 1, 4: 4, 6: 3}   # 原数据库中1表示上市，2表示暂停上市，3表示恢复上市，4表示退市,6表示退市整理
    ls_data['list_status'] = ls_data.list_status.map(ls_map)
    ls_data['code'] = ls_data.code.apply(datatoolkits.add_suffix)
    by_code = ls_data.groupby('code')
    tds = dateshandle.get_tds(start_time, end_time)
    ls_data = by_code.apply(datatoolkits.map_data, days=tds,
                            fillna={'code': lambda x: x.code.iloc[0]},
                            fromNowOn=True)
    # ls_data = ls_data.reset_index(drop=True).set_index(['time', 'code'])
    # ls_data = ls_data.loc[:, 'list_status'].unstack()
    ls_data = ls_data.reset_index(drop=True)
    ls_data = ls_data.pivot_table('list_status', index='time',
                                  columns='code').dropna(axis=0, how='all')
    ls_data = ls_data.loc[:, sorted(universe)]
    assert checkdata_completeness(ls_data, start_time, end_time), "Error, data missed!"
    return ls_data


ls_factor = Factor('LIST_STATUS', get_liststatus, pd.to_datetime('2017-07-14'),
                   desc='1表示正常上市，2表示暂停上市，3表示退市整理，4表示终止上市')
# --------------------------------------------------------------------------------------------------
# 特殊处理，0表示正常，1表示ST，2表示*ST，3表示退市整理，4表示高风险警示，5表示PT


def get_st(universe, start_time, end_time):
    '''
    获取股票特殊处理的情况
    '''
    st_data = fdgetter.get_db_data(fdgetter.BASIC_SQLs['ST_TAG'],
                                   cols=('time', 'abbr', 'ms', 'code'),
                                   add_stockcode=False)

    def _assign_st(row):
        map_dict = {'ST': 1, 'PT': 5, '撤销ST': 0, '*ST': 2, '撤消*ST并实行ST': 1,
                    '从ST变为*ST': 2, '撤销*ST': 0, '退市整理期': 3, '高风险警示': 4}
        if row.ms in map_dict:
            return map_dict[row.ms]
        else:
            assert row.ms == '撤销PT', "Error, cannot handle tag '{tag}'".format(tag=row.ms)
            if 'ST' in row.abbr:
                return 1
            elif '*ST' in row.abbr:
                return 2
            else:
                return 0
    st_data = st_data.assign(tag=lambda x: x.apply(_assign_st, axis=1))
    st_data['code'] = st_data.code.apply(datatoolkits.add_suffix)
    # 剔除日期重复项，因为数字越大表示越风险越高，因而只保留数字大的
    # pdb.set_trace()
    st_data = st_data.sort_values(['code', 'time', 'tag'])
    by_codentime = st_data.groupby(['code', 'time'])
    st_data = by_codentime.apply(lambda x: x.tail(1).iloc[0])
    st_data = st_data.reset_index(drop=True)
    tds = dateshandle.get_tds(start_time, end_time)
    # pdb.set_trace()
    by_code = st_data.groupby('code')
    st_data = by_code.apply(datatoolkits.map_data, days=tds,
                            fillna={'code': lambda x: x.code.iloc[0]},
                            fromNowOn=True)
    st_data = st_data.reset_index(drop=True)
    # st_data = st_data.reset_index(drop=True).set_index(['time', 'code'])
    # st_data = st_data.loc[:, 'tag'].unstack()
    st_data = st_data.pivot_table('tag', index='time', columns='code').dropna(axis=0, how='all')
    st_data = st_data.loc[:, sorted(universe)].fillna(0)
    assert checkdata_completeness(st_data, start_time, end_time), "Error, data missed!"
    return st_data


st_factor = Factor('ST_TAG', get_st, pd.to_datetime('2017-07-15'),
                   desc='0表示正常，1表示ST，2表示*ST，3表示退市整理，4表示高风险警示，5表示PT')
# --------------------------------------------------------------------------------------------------
# 交易状态，区分股票是否能够交易（目前考虑的不能交易的状况为停牌，且停牌的时间超过1个完整的
# 交易日，对于日内停牌不超过一个交易日的，视为正常交易日），实现中采用成交量来衡量是否可以交易


def get_tradeable(universe, start_time, end_time):
    '''
    获取股票的交易状态
    Notes
    -----
    将成交量为0或者最高价等于最低价视为不能交易，返回值为1表示正常交易，0表示不能交易，NA表示未上市而不能交易
    '''
    sql = '''
    SELECT S.TradingDay, S.TurnoverVolume, S.HighPrice, S.LowPrice, M.Secucode
    FROM QT_DailyQuote S, SecuMain M
    WHERE
        S.InnerCode = M.InnerCode AND
        M.SecuMarket in (83, 90) AND
        S.TradingDay <= CAST(\'{end_time}\' as datetime) AND
        S.TradingDay >= CAST(\'{start_time}\' as datetime) AND
        M.SecuCategory = 1
    ORDER BY S.TradingDay ASC, M.Secucode ASC
    '''
    data = fdgetter.get_db_data(sql, cols=('time', 'vol', 'high', 'low', 'code'),
                                start_time=start_time, end_time=end_time, add_stockcode=False)
    # pdb.set_trace()
    data['code'] = data.code.apply(datatoolkits.add_suffix)
    data.loc[data.vol > 0, 'vol'] = 1
    data.loc[(data.vol > 0) & (data.high == data.low), 'vol'] = 0
    data = data.pivot_table('vol', index='time', columns='code')
    data = data.loc[:, sorted(universe)]
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


tradeable_factor = Factor('TRADEABLE', get_tradeable, pd.to_datetime('2017-7-17'),
                          desc='0视为不能交易，1表示正常交易，NA表示为上市或者退市')
# --------------------------------------------------------------------------------------------------
# 获取指数成份股，1表示是成份，NA表示不是成分


def get_iconstituents(index_code):
    '''
    依据提供的指数代码获取指数成份，因为指数的成分基本一个月一次更新，因此，为了避免获取数据时，无法
    获取指定时间内的数据，需要将给定的开始日期玩前推一段时间（两个月）
    '''
    def _inner(universe, start_time, end_time):
        start_time = pd.to_datetime(start_time)
        nstart_time = start_time - pd.Timedelta('60 days')
        data = fdgetter.get_db_data(fdgetter.BASIC_SQLs['INDEX_CONSTITUENTS'], code=index_code,
                                    cols=('code', 'time'), add_stockcode=False,
                                    start_time=nstart_time, end_time=end_time)
        data = data.assign(is_constituent=1)
        data['code'] = data.code.apply(datatoolkits.add_suffix)
        data = data.pivot_table('is_constituent', index='time', columns='code')
        tds = dateshandle.get_tds(nstart_time, end_time)
        data = data.loc[:, sorted(universe)].reset_index()
        data = datatoolkits.map_data(data, days=tds, fromNowOn=True)
        data = data.set_index('time').dropna(axis=0, how='all')
        data = data.loc[(data.index >= start_time) & (data.index <= end_time)]
        # pdb.set_trace()
        if start_time > pd.to_datetime(START_TIME):
            assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return _inner


# 上证50成份
get_IH_constituents = get_iconstituents('000016')
# 中证500
get_IC_constituents = get_iconstituents('000905')
# 沪深300
get_IF_constituents = get_iconstituents('000300')

IH_constituents = Factor('IH_CONS', get_IH_constituents, pd.to_datetime('2017-07-18'))
IC_constituents = Factor('IC_CONS', get_IC_constituents, pd.to_datetime('2017-07-18'))
IF_constituents = Factor('IF_CONS', get_IF_constituents, pd.to_datetime('2017-07-18'))
# --------------------------------------------------------------------------------------------------
factor_list = [zx_ind_factor, ls_factor, st_factor, tradeable_factor, IH_constituents,
               IC_constituents, IF_constituents]
check_duplicate_factorname(factor_list, __name__)

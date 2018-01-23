#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-22 15:41:02
# @Author  : Hao Li (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
经过异常数据处理、填充以及标准化处理后的BARRA因子数据，包含相关的处理函数
有效的数据包括：上市时间超过半年、被分配了有效的中信行业
相比于原始数据，本模块中的数据仅进行极端值检测、异常值拉回、缺失值填充、标准化
所有因子必须以BARRA_开头，名称与CNE5的名称一致
'''

import pdb

import pandas as pd
import numpy as np
import statsmodels.robust as robust_mad
from statsmodels.stats.stattools import medcouple

from fmanager.factors.query import query
from fmanager.factors.utils import (check_indexorder, checkdata_completeness, Factor,
                                    check_duplicate_factorname, convert_data)
from dateshandle import tds_shift
from fmanager.database.const import NaS


NAME = 'barra'


def get_factor_dict():
    res = dict()
    for f in factor_list:
        res[f.name] = {'rel_path': NAME + '\\' + f.name, 'factor': f}
    return res


factor_list = []
# --------------------------------------------------------------------------------------------------
# 工具函数


def error_detection_mad(ts, threshold=3):
    '''
    使用MAD方法检测序列中的异常值

    Parameter
    ---------
    ts: pd.Series
        需要检测异常值的序列
    threshold: int, default 3
        异常值的确定标准，默认为3，意义见Notes

    Return
    ------
    out: pd.Series
        boolean序列，异常值会被标记为True，其他值会被标记为False

    Notes
    -----
    MAD方法是将median +(-) 1.483 * threshold * mad作为上（下）界
    '''
    med = np.median(ts.dropna())
    mad = robust_mad(ts.dropna())
    bound = 1.483 * threshold * mad
    lower = med - bound
    upper = med + bound
    out = (ts > upper) | (ts < lower) | pd.isnull(ts)
    # print('upper=', upper, ' lower=', lower)
    return out


def error_detection_mbp(ts):
    '''
    使用经过偏度调整过后的boxplot方法剔除异常值，参考东方证券《选股因子数据的异常值处理和正态转换》

    Parameter
    ---------
    ts: pd.Series
        需要检测异常值的序列

    Return
    ------
    out: pd.Series
        boolean序列，异常值会被标记为True，其他值则为False

    Notes
    -----
    md = median(ts)
    mc = median(((x_i - md) - (md - x_j)) / (x_i - x_j), x_i >= md and x_j < md)
    L = Q_1 - 1.5 * exp(-3.5 * mc) * IQR if mc >= 0
    L = Q_1 - 1.5 * exp(-4 * mc) * IQR if mc < 0
    U = Q_3 + 1.5 * exp(4 * mc) * IQR if mc >= 0
    U = Q_3 + 1.5 * exp(3.5 * mc) * IQR if mc < 0
    异常数据为(-inf, L)|(U, inf)
    '''
    q1, q3 = ts.quantile([0.25, 0.75])
    iqr = q3 - q1
    mc = float(medcouple(ts.dropna()))
    if mc >= 0:
        multiple_l = -3.5
        multiple_u = 4
    else:
        multiple_l = -4
        multiple_u = 3.5
    upper = q3 + 1.5 * np.exp(multiple_u * mc) * iqr
    lower = q1 - 1.5 * np.exp(multiple_l * mc) * iqr
    out = (ts < lower) | (ts > upper) | pd.isnull(ts)
    return out


def winsorize(ts, nan_mask=None, multiple=3):
    '''
    使用标准差方法对数据进行Winsorize处理，即对于数值在mean+(-)multiple*std的数据拉回到边界上，
    在使用winsorize之前，最好对数据进行异常值剔除处理，避免异常值对标准差的影响过大

    Parameter
    ---------
    ts: pd.Series
        需要进行处理的数据序列
    nan_mask: pd.Series, default None
        是否为NA值的布尔序列
    multiple: float, default 3
        确定上下界的波动率乘数

    Return
    ------
    out: pd.Series
        经过winsorize处理后的结果，index与原数据相同
    '''
    if nan_mask is None:
        nan_mask = pd.Series(True, index=ts.index)
    mean = np.mean(ts.loc[nan_mask])
    std = np.std(ts.loc[nan_mask])
    upper = mean + multiple * std
    lower = mean - multiple * std
    out = ts.copy()
    out.loc[out > upper] = upper
    out.loc[out < lower] = lower
    return out


# --------------------------------------------------------------------------------------------------
# 有效数据因子：指上市时间超过半年(125个交易日)，当前未退市且有有效中信行业的股票
# BARRA_VSF(valid stock flag)
def get_vsf(universe, start_time, end_time):
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    offset = 125
    forward_start = tds_shift(start_time, offset)
    ls_status = query('LIST_STATUS', (forward_start, end_time))
    ls_status_shift = ls_status.shift(offset)
    ls_status_shift = ls_status_shift.loc[ls_status_shift.index >= start_time]
    valid_stock = (ls_status_shift == 1) & (ls_status == 1)
    ind_data = query('ZX_IND', (start_time, end_time))
    ind_flag = ind_data != NaS
    valid_stock = valid_stock & ind_flag
    mask = (valid_stock.index >= start_time) & (valid_stock.index <= end_time)
    out = valid_stock.loc[mask, sorted(universe)]
    assert checkdata_completeness(out, start_time, end_time), "Error, data missed!"
    assert check_indexorder(out), 'Error, data order is mixed!'
    return out


factor_list.append(Factor('BARRA_VSF', get_vsf, pd.to_datetime('2018-01-22'),
                          dependency=['LIST_STATUS', 'ZX_IND'],
                          desc='有效数据因子，指上市时间超过125个交易日且有有效中信行业的股票'))


# 计算模板，工厂类函数
def barradata_factory(factor_name):
    '''
    工厂函数，用于返回计算BARRA因子的函数

    Parameter
    ---------
    factor_name: string
        原始因子数据的名称

    Return
    ------
    func: function(universe, start_time, end_time)
        计算因子值的函数
    '''
    def inner(universe, start_time, end_time):
        universe = sorted(universe)
        factor_data = query(factor_name, (start_time, end_time))
        vf_data = query('BARRA_VSF', (start_time, end_time))
        ind_data = query('ZX_IND', (start_time, end_time))
        mktv_data = query('TOTAL_MKTVALUE', (start_time, end_time))
        mktv_data = mktv_data.T.transform(lambda x: x / x.sum()).T
        data = convert_data([factor_data, vf_data, ind_data, mktv_data],
                            ['factor', 'valid_stock', 'industry', 'mktv'])
        # pdb.set_trace()

        def calcbydate(df):
            '''
            每个交易日对数据进行相关处理，返回pd.Series(data, index=sorted(universe))
            '''
            df = df.reset_index(level=0, drop=True).T\
                .astype({'factor': np.float64, 'valid_stock': np.bool, 'mktv': np.float64})
            # df = df.T.astype({'factor': np.float64, 'valid_stock': np.bool, 'mktv': np.float64})
            df = df.loc[df['valid_stock']]
            # 极端值处理
            err_flag = error_detection_mbp(df['factor'])
            df.loc[err_flag, 'factor'] = np.nan
            # 异常值处理
            df['factor'] = winsorize(df['factor'], pd.isnull(df['factor']))
            # pdb.set_trace()
            ind_mean = df.groupby('industry')['factor'].transform('mean')
            out = df['factor'].fillna(ind_mean)
            # pdb.set_trace()
            wmean = out.dot(df.loc[out.index, 'mktv'])
            std = np.std(out)
            out = (out - wmean) / std
            out = out.reindex(universe)
            return out
        out = data.groupby(level=0, group_keys=False).apply(calcbydate)
        # test = data.xs('2017-12-25', level=0)
        # out = calcbydate(test)
        return out
    return inner


check_duplicate_factorname(factor_list, __name__)

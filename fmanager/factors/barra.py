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

import pandas as pd
import numpy as np
import statsmodels.robust as robust_mad
from statsmodels.stats.stattools import medcouple

from fmanager.factors.query import query
from fmanager.factors.utils import (check_indexorder, checkdata_completeness, Factor,
                                    check_duplicate_factorname)
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


def winsorize(ts, multiple=3):
    '''
    使用标准差方法对数据进行Winsorize处理，即对于数值在mean+(-)multiple*std的数据拉回到边界上，
    在使用winsorize之前，最好对数据进行异常值剔除处理，避免异常值对标准差的影响过大

    Parameter
    ---------
    ts: pd.Series
        需要进行处理的数据序列
    multiple: float, default 3
        确定上下界的波动率乘数

    Return
    ------
    out: pd.Series
        经过winsorize处理后的结果，index与原数据相同
    '''
    mean = ts.mean()
    std = ts.std()
    upper = mean + multiple * std
    lower = mean - multiple * std
    out = ts.copy()
    out.loc[out > upper] = upper
    out.loc[out < lower] = lower
    return out


# --------------------------------------------------------------------------------------------------
# 有效数据因子：指上市时间超过半年(125个交易日)且有有效中信行业的股票
# BARRA_VSF(valid stock flag)
def get_vsf(universe, start_time, end_time):
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    offset = 125
    forward_start = tds_shift(start_time, offset)
    ls_status = query('LIST_STATUS', (forward_start, end_time)).shift(offset)
    ls_status = ls_status.loc[ls_status.index >= start_time]
    valid_stock = ls_status == 1
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

check_duplicate_factorname(factor_list, __name__)

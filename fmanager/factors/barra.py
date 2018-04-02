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
from statsmodels.regression.linear_model import WLS
from statsmodels.tools import add_constant
from tqdm import tqdm

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
    valid_stock = (valid_stock & ind_flag).astype(np.float64)
    mask = (valid_stock.index >= start_time) & (valid_stock.index <= end_time)
    out = valid_stock.loc[mask, sorted(universe)]
    assert checkdata_completeness(out, start_time, end_time), "Error, data missed!"
    assert check_indexorder(out), 'Error, data order is mixed!'
    return out


factor_list.append(Factor('BARRA_VSF', get_vsf, pd.to_datetime('2018-01-22'),
                          dependency=['LIST_STATUS', 'ZX_IND'],
                          desc='BARRA有效数据因子，指上市时间超过125个交易日且有有效中信行业的股票，其中1.0表示有效，0.表示无效，NA表示无数据'))


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
        # pdb.set_trace()
        try:
            data = convert_data([factor_data, vf_data, ind_data, mktv_data],
                                ['factor', 'valid_stock', 'industry', 'mktv'])
        except AssertionError:
            factor_data = factor_data.reindex(index=vf_data.index)
            data = convert_data([factor_data, vf_data, ind_data, mktv_data],
                                ['factor', 'valid_stock', 'industry', 'mktv'])
        # pdb.set_trace()

        def calcbydate(df):
            '''
            每个交易日对数据进行相关处理，返回pd.Series(data, index=sorted(universe))
            '''
            df = df.reset_index(level=0, drop=True).T\
                .astype({'factor': np.float64, 'valid_stock': np.float64, 'mktv': np.float64})
            # df = df.T.astype({'factor': np.float64, 'valid_stock': np.bool, 'mktv': np.float64})
            df = df.loc[df['valid_stock'] == 1.0]
            if len(df) == 0 or pd.isnull(df['factor']).sum() / len(df) > 0.995:  # 剔除NA数据占比过高的情况
                return pd.Series(np.nan, index=universe)
            # 极端值处理
            # pdb.set_trace()
            err_flag = error_detection_mbp(df['factor'])
            df.loc[err_flag, 'factor'] = np.nan
            # 异常值处理
            df['factor'] = winsorize(df['factor'], pd.isnull(df['factor']))
            # pdb.set_trace()
            ind_mean = df.groupby('industry')['factor'].transform('mean')
            if pd.isnull(ind_mean).sum() / len(ind_mean) > 0.1:     # 缺失行业均值的股票数量超过10%
                return pd.Series(np.nan, index=universe)
            else:   # 行业均值缺失情况在阈值之下，使用市场均值进行填充
                ind_mean = ind_mean.fillna(df['factor'].mean())
            out = df['factor'].fillna(ind_mean)
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


def generate_factor(raw_name, barra_name, add_time):
    '''
    辅助函数，用于自动化生成因子

    Parameter
    ---------
    raw_name: string
        原始因子的名称
    barra_name: string
        BARRA因子的名称
    add_time: datetime like
        因子添加时间
    '''
    dep = ['BARRA_VSF', 'ZX_IND', 'TOTAL_MKTVALUE', raw_name]
    desc = '{name}因子'.format(name=barra_name.replace('_', ' '))
    factor = Factor(barra_name, barradata_factory(raw_name), pd.to_datetime(add_time),
                    dependency=dep, desc=desc)
    return factor


# --------------------------------------------------------------------------------------------------
# BARRA风险因子，即使用描述因子合成风险因子


def barra_rf_factory(weights, tobe_orthogonalized=None):
    '''
    母函数，用于生成计算风险因子的函数

    Parameter
    ---------
    weights: dict
        格式为{descriptor_name: weight}
    tobe_orthogonalized: list
        正交化对象风险因子，None表示当前不需要正交化

    Return
    ------
    func: function(universe, start_time, end_time)->pandas.DataFrame
    '''
    def inner(universe, start_time, end_time):
        descriptor = sorted(weights.keys())
        if tobe_orthogonalized is not None:
            descriptor += tobe_orthogonalized
        descriptor += ['BARRA_VSF', 'TOTAL_MKTVALUE']    # 用于识别当前理论上应当有效的数据量
        # pdb.set_trace()
        datas = [query(desc, (start_time, end_time)) for desc in descriptor]
        datas = convert_data(datas, descriptor)
        desc_weights = pd.Series(weights)
        desc_weights = pd.DataFrame(np.repeat(desc_weights.values.reshape((-1, 1)), len(datas.columns),
                                              axis=1), index=desc_weights.index, columns=datas.columns)

        def calc_rf_bydate(df):
            # 计算每个时间点的风险因子数据
            # 正交化处理也必须放在每个交易日数据计算中，因为涉及到NA值的处理
            df = df.reset_index(level=0, drop=True)
            descrpitor_data = df.loc[desc_weights.index, :]
            na_mask = pd.isnull(descrpitor_data)
            # 采用smart weight，即对于NA值，给予的权重为0，并且将其他权重重新计算
            if len(weights) > 1:
                tmp_weight = desc_weights.copy()
                tmp_weight[na_mask] = np.nan
                tmp_weight = tmp_weight.div(tmp_weight.sum(axis=0), axis=1)
                res = (tmp_weight * descrpitor_data).sum(axis=0)
                res[np.all(na_mask, axis=0)] = np.nan   # 所有数据都为NA时，避免pandas将其计算为0
            else:
                res = descrpitor_data.iloc[0, :]
            if (df.loc['BARRA_VSF'].sum() == 0 or
                    res[df.loc['BARRA_VSF'] == 1].count() / df.loc['BARRA_VSF'].sum() < 0.5):  # 有效数据占VSF的比例不足50%，则数据无效，设置为NA
                res.loc[:] = np.nan
                return res

            valid_mask = df.loc['BARRA_VSF'] == 1
            if np.any(pd.isnull(res.loc[valid_mask])):
                res.loc[:] = np.nan
                return res
            if tobe_orthogonalized is not None:  # 需要进行正交化操作
                orth_data = df.loc[tobe_orthogonalized, valid_mask].T
                if np.any(np.any(pd.isnull(orth_data), axis=0)):    # 当前正交数据中有缺失
                    res.loc[:] = np.nan
                    return res
                res = res.loc[valid_mask]
                ols_weight = np.sqrt(df.loc['TOTAL_MKTVALUE', valid_mask])
                ols_weight = ols_weight / ols_weight.sum()
                res = WLS(res, add_constant(orth_data), weights=ols_weight).fit().resid
            else:
                res = res.loc[valid_mask]
            mkt_weight = df.loc['TOTAL_MKTVALUE', res.index]
            mkt_weight = mkt_weight / mkt_weight.sum()
            res = (res - (res * mkt_weight).sum()) / np.std(res)
            return res.reindex(df.columns)
        # tqdm.pandas()
        # rf_data = datas.groupby(level=0).progress_apply(calc_rf_bydate)
        rf_data = datas.groupby(level=0).apply(calc_rf_bydate)
        return rf_data
    return inner


def generate_rf_factor(factor_name, weights, tobe_orthogonalized, add_time):
    '''
    辅助函数，用于自动化生成风险因子

    Parameter
    ---------
    factor_name: string
        新因子的名称
    weights: dict
        描述子的权重
    tobe_orghogonalized: list
        正交对象风险因子列表，若没有需要正交的对象，该参数应该为None
    add_time: datetime like
        因子添加时间

    Return
    ------
    factor: Factor
        风险因子对象
    '''
    calcu_method = barra_rf_factory(weights, tobe_orthogonalized)
    dep = ['BARRA_VSF', 'TOTAL_MKTVALUE'] + list(weights.keys())
    if tobe_orthogonalized is not None:
        dep += tobe_orthogonalized
    desc = '{}风险因子'.format(factor_name)
    factor = Factor(factor_name, calcu_method, pd.to_datetime(add_time), dependency=dep, desc=desc)
    return factor

# --------------------------------------------------------------------------------------------------


BARRA_FACTORS = [('LN_TMKV', 'BARRA_LNCAP', '2018-01-24'), ('BETA', 'BARRA_BETA', '2018-01-24'),
                 ('RSTR', 'BARRA_RSTR', '2018-01-24'), ('DSTD', 'BARRA_DASTD', '2018-01-24'),
                 ('CMRA', 'BARRA_CMRA', '2018-01-24'), ('SPECIAL_VOL', 'BARRA_HSIGMA', '2018-01-24'),
                 ('NLSIZE', 'BARRA_NLSIZE', '2018-01-24'), ('BP', 'BARRA_BTOP', '2018-01-24'),
                 ('STOM', 'BARRA_STOM', '2018-01-24'), ('STOQ', 'BARRA_STOQ', '2018-01-24'),
                 ('STOA', 'BARRA_STOA', '2018-01-24'), ('CFP_TTM', 'BARRA_CETOP', '2018-01-24'),
                 ('EP_TTM', 'BARRA_ETOP', '2018-01-24'), ('NI_5YG', 'BARRA_EGRO', '2018-01-24'),
                 ('OPREV_5YG', 'BARRA_SGRO', '2018-01-24'), ('MLEV', 'BARRA_MLEV', '2018-01-24'),
                 ('BLEV', 'BARRA_BLEV', '2018-01-24'), ('DTOA', 'BARRA_DTOA', '2018-01-24')]

for factor_cfg in BARRA_FACTORS:
    factor_list.append(generate_factor(*factor_cfg))


BARRA_RF_FACTORS = [('BARRA_RF_SIZE', {'BARRA_LNCAP': 1}, None, '2018-03-31'),
                    ('BARRA_RF_BETA', {'BARRA_BETA': 1}, None, '2018-03-31'),
                    ('BARRA_RF_MOM', {'BARRA_RSTR': 1}, None, '2018-03-31'),
                    ('BARRA_RF_RESIDUAL_VOL', {'BARRA_DASTD': 0.74, 'BARRA_CMRA': 0.16,
                                               'BARRA_HSIGMA': 0.1},
                     ['BARRA_RF_BETA', 'BARRA_RF_SIZE'], '2018-03-31'),
                    ('BARRA_RF_NLSIZE', {'BARRA_NLSIZE': 1}, ['BARRA_RF_SIZE'], '2018-03-31'),
                    ('BARRA_RF_BP', {'BARRA_BTOP': 1}, None, '2018-03-31'),
                    ('BARRA_RF_LIQUIDITY', {'BARRA_STOA': 0.3, 'BARRA_STOM': 0.35,
                                            'BARRA_STOQ': 0.35}, ['BARRA_RF_SIZE'], '2018-03-31'),
                    ('BARRA_RF_EARNINGS_YIELD', {'BARRA_CETOP': 0.67, 'BARRA_ETOP': 0.33}, None,
                        '2018-03-31'),
                    ('BARRA_RF_GROWTH', {'BARRA_EGRO': 0.33, 'BARRA_SGRO': 0.67}, None,
                        '2018-03-31'),
                    ('BARRA_RF_LEVERAGE', {'BARRA_MLEV': 0.38, 'BARRA_DTOA': 0.35,
                                           'BARRA_BLEV': 0.27}, None, '2018-03-31')]
for rf_cfg in BARRA_RF_FACTORS:
    factor_list.append(generate_rf_factor(*rf_cfg))
check_duplicate_factorname(factor_list, __name__)

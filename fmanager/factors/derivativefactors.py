#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-18 14:37:17
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
由基础数据计算而来的衍生因子

__version__ = 1.0.0
修改日期：2017-07-27
修改内容：
    初始化
'''

# import datatoolkits
import dateshandle
import numpy as np
from numpy.linalg import linalg, LinAlgError
import pandas as pd
import pdb
import warnings
# from functools import wraps

from scipy.stats import skew, kurtosis
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
from tqdm import tqdm
from fmanager.const import START_TIME
from fmanager.factors.utils import (Factor, check_indexorder, check_duplicate_factorname,
                                    convert_data, checkdata_completeness, drop_delist_data)
from fmanager.factors.query import query
import fdgetter
import datatoolkits
from fmanager.factors.utils import convert_data
from tdtools import get_calendar, trans_date
from datatoolkits import rolling_apply

# from statsmodels.api import add_constant

# --------------------------------------------------------------------------------------------------
# 常量和功能函数
NAME = 'derivativefactors'


def get_factor_dict():
    res = dict()
    for f in factor_list:
        res[f.name] = {'rel_path': NAME + '\\' + f.name, 'factor': f}
    return res


factor_list = []
# --------------------------------------------------------------------------------------------------
# 价值类因子
# EP_TTM


@drop_delist_data
def get_ep(universe, start_time, end_time):
    '''
    EP为净利润与总市值的比
    '''
    ni_data = query('NI_TTM', (start_time, end_time))
    tmktv_data = query('TOTAL_MKTVALUE', (start_time, end_time))
    ep = ni_data / tmktv_data
    ep = ep.loc[:, sorted(universe)]
    assert check_indexorder(ep), 'Error, data order is mixed!'
    assert checkdata_completeness(ep, start_time, end_time), "Error, data missed!"
    return ep


factor_list.append(Factor('EP_TTM', get_ep, pd.to_datetime('2017-07-27'),
                          dependency=['NI_TTM', 'TOTAL_MKTVALUE'], desc='净利润/总市值计算得到'))

# BP_TTM


@drop_delist_data
def get_bp(universe, start_time, end_time):
    '''
    BP为归属母公司权益/总市值
    '''
    bv_data = query('EQUITY', (start_time, end_time))
    tmktv_data = query('TOTAL_MKTVALUE', (start_time, end_time))
    bp = bv_data / tmktv_data
    bp = bp.loc[:, sorted(universe)]
    assert check_indexorder(bp), 'Error, data order is mixed!'
    assert checkdata_completeness(bp, start_time, end_time), "Error, data missed!"
    return bp


factor_list.append(Factor('BP', get_bp, pd.to_datetime('2017-07-27'),
                          dependency=['EQUITY', 'TOTAL_MKTVALUE'], desc='最新的归属母公司权益/总市值'))

# SP_TTM


@drop_delist_data
def get_sp(universe, start_time, end_time):
    '''
    SP为营业收入与总市值的比
    '''
    sale_data = query('OPREV_TTM', (start_time, end_time))
    tmktv_data = query('TOTAL_MKTVALUE', (start_time, end_time))
    sp = sale_data / tmktv_data
    sp = sp.loc[:, sorted(universe)]
    assert check_indexorder(sp), 'Error, data order is mixed!'
    assert checkdata_completeness(sp, start_time, end_time), "Error, data missed!"
    return sp


factor_list.append(Factor('SP_TTM', get_sp, pd.to_datetime('2017-07-27'),
                          dependency=['OPREV_TTM', 'TOTAL_MKTVALUE'], desc='营业收入/总市值'))

# CFP_TTM


@drop_delist_data
def get_cfp(universe, start_time, end_time):
    '''
    CFP为经营活动产生的现金流量净额/总市值
    '''
    cf_data = query('OPNETCF_TTM', (start_time, end_time))
    tmktv_data = query('TOTAL_MKTVALUE', (start_time, end_time))
    cfp = cf_data / tmktv_data
    cfp = cfp.loc[:, sorted(universe)]
    assert check_indexorder(cfp), 'Error, data order is mixed!'
    assert checkdata_completeness(cfp, start_time, end_time), "Error, data missed!"
    return cfp


factor_list.append(Factor('CFP_TTM', get_cfp, pd.to_datetime('2017-07-27'),
                          dependency=['OPNETCF_TTM', 'TOTAL_MKTVALUE'], desc='经营活动中现金流净额/总市值'))

# SALE2EV


@drop_delist_data
def get_sale2ev(universe, start_time, end_time):
    '''
    SALE2EV = 营业收入/(总市值+非流动负债合计-货币资金)
    '''
    sale_data = query('OPREV_TTM', (start_time, end_time), fillna=0)
    tmktv_data = query('TOTAL_MKTVALUE', (start_time, end_time), fillna=0)
    ncdebt_data = query('TNCL', (start_time, end_time), fillna=0)
    cash_data = query('CASH', (start_time, end_time), fillna=0)
    data = sale_data / (tmktv_data + ncdebt_data - cash_data)
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('SALE2EV', get_sale2ev, pd.to_datetime('2017-07-27'),
                          dependency=['OPREV_TTM', 'TOTAL_MKTVALUE', 'TNCL', 'CASH'],
                          desc='营业收入/(总市值+非流动负债合计-货币资金)'))
# --------------------------------------------------------------------------------------------------
# 成长类因子
# 单季度营业收入同比增长


@drop_delist_data
def get_oprev_yoy(universe, start_time, end_time):
    '''
    OPREV_YOY = (本季度营业收入-上年同季度营业收入)/abs(上年同季度营业收入)
    '''
    oprev_lq = query('OPREV_1S', (start_time, end_time))
    oprev_lyq = query('OPREV_5S', (start_time, end_time))
    data = (oprev_lq - oprev_lyq) / np.abs(oprev_lyq) - 1   # 这个地方有错误，不应该减1，下同
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('OPREV_YOY', get_oprev_yoy, pd.to_datetime('2017-07-27'),
                          dependency=['OPREV_1S', 'OPREV_5S'],
                          desc='(本季度营业收入-上年同季度营业收入)/abs(上年同季度营业收入)'))
# 单季度净利润同比增长


@drop_delist_data
def get_ni_yoy(universe, start_time, end_time):
    '''
    NI_YOY = (本季度净利润-上年同季度净利润)/abs(上年同季度净利润)
    '''
    ni_lq = query('NI_1S', (start_time, end_time))
    ni_lyq = query('NI_5S', (start_time, end_time))
    data = (ni_lq - ni_lyq) / np.abs(ni_lyq) - 1
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('NI_YOY', get_ni_yoy, pd.to_datetime('2017-07-27'),
                          dependency=['NI_1S', 'NI_5S'],
                          desc='(本季度净利润-上年同季度净利润)/abs(上年同季度净利润)'))

# 过去5年增长率


def get_p5ygrowth(factor_type):
    '''
    母函数，用于生成计算过去五年平均增长率的函数

    Parameter
    ---------
    factor_type: str
        因子类型，目前只支持['NI', 'OPREV']

    Notes
    -----
    采用将对应数值对时间做回归（时间由远到近依次为1到5），然后除以平均值的绝对值
    '''
    def calc_growth(df):
        # 假设数据按照升序排列，即由上到下依次为1-5
        t = np.arange(5, 0, -1)
        df_mean = df.mean()
        df_demean = df - df_mean
        res = np.dot(t, df_demean.values) / 10
        res = pd.Series(res, index=df.columns)
        res = res / np.abs(df_mean)
        return res

    @drop_delist_data
    def _inner(universe, start_time, end_time):
        datas = list()
        for i in range(1, 6):
            tmp_data = query(factor_type + '_%dY' % i, (start_time, end_time))
            datas.append(tmp_data)
        data = convert_data(datas, range(1, 6))  # 1（最近年度）-5（最远年度）依次表示到现在的时间间隔越来越远
        data = data.sort_index()
        by_date = data.groupby(level=0)
        data = by_date.apply(calc_growth)
        data = data.loc[:, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return _inner


# 净利润过去5年增长率
factor_list.append(Factor('NI_5YG', get_p5ygrowth('NI'), pd.to_datetime('2017-07-28'),
                          dependency=['NI_%dY' % i for i in range(1, 6)]))
# 营业收入过去5年增长率
factor_list.append(Factor('OPREV_5YG', get_p5ygrowth('OPREV'), pd.to_datetime('2017-07-28'),
                          dependency=['OPREV_%dY' % i for i in range(1, 6)]))

# --------------------------------------------------------------------------------------------------
# 质量类因子
# ROE


@drop_delist_data
def get_roe(universe, start_time, end_time):
    '''
    ROE = 净利润TTM / 归属母公司权益
    '''
    ni_data = query('NI_TTM', (start_time, end_time))
    equity_data = query('EQUITY', (start_time, end_time))
    data = ni_data / equity_data
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('ROE', get_roe, pd.to_datetime('2017-07-28'),
                          dependency=['NI_TTM', 'EQUITY'], desc='净利润TTM/归属母公司权益'))
# ROA


@drop_delist_data
def get_roa(universe, start_time, end_time):
    '''
    ROA = 净利润TTM / 总资产
    '''
    ni_data = query('NI_TTM', (start_time, end_time))
    ta_data = query('TA', (start_time, end_time))
    data = ni_data / ta_data
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('ROA', get_roa, pd.to_datetime('2017-07-28'),
                          dependency=['NI_TTM', 'TA'], desc='净利润TTM/总资产'))

# 营业利润率


@drop_delist_data
def get_opprofitmargin(universe, start_time, end_time):
    '''
    营业利润率 = (营业收入-营业成本-销售费用-管理费用-财务费用) / abs(营业收入)
    '''
    oprev = query('OPREV_TTM', (start_time, end_time))
    opcost = query('OPCOST_TTM', (start_time, end_time))
    opsale = query('OPEXP_TTM', (start_time, end_time))
    adminexp = query('ADMINEXP_TTM', (start_time, end_time))
    fiexp = query('FIEXP_TTM', (start_time, end_time))
    data = (oprev - opcost - opsale - adminexp - fiexp) / np.abs(oprev)
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('OPPROFIT_MARGIN', get_opprofitmargin, pd.to_datetime('2017-07-28'),
                          dependency=['OPREV_TTM', 'OPCOST_TTM', 'OPEXP_TTM', 'ADMINEXP_TTM',
                                      'FIEXP_TTM'],
                          desc='营业利润率 = (营业收入-营业成本-销售费用-管理费用-财务费用) / abs(营业收入)'))

# 毛利率


@drop_delist_data
def get_grossmargin(universe, start_time, end_time):
    '''
    毛利率 = (营业收入 - 营业成本) / abs(营业收入)
    '''
    oprev = query('OPREV_TTM', (start_time, end_time))
    opcost = query('OPCOST_TTM', (start_time, end_time))
    data = (oprev - opcost) / np.abs(oprev)
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('GROSS_MARGIN', get_grossmargin, pd.to_datetime('2017-07-31'),
                          dependency=['OPREV_TTM', 'OPCOST_TTM'],
                          desc='毛利率 = (营业收入 - 营业成本) / abs(营业收入)'))

# 资产周转率


@drop_delist_data
def get_tato(universe, start_time, end_time):
    '''
    资产周转率 = 营业收入TTM / 最新总资产
    '''
    oprev = query('OPREV_TTM', (start_time, end_time))
    ta = query('TA', (start_time, end_time))
    data = oprev / ta
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('TATO', get_tato, pd.to_datetime('2017-07-31'),
                          dependency=['OPREV_TTM', 'TA'], desc='营业收入TTM / 最新总资产'))

# 流动比率


@drop_delist_data
def get_currentratio(universe, start_time, end_time):
    '''
    流动比率 = 流动资产 / 流动负债
    '''
    ca = query('TCA', (start_time, end_time))
    cl = query('TCL', (start_time, end_time))
    data = ca / cl
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('CURRENT_RATIO', get_currentratio, pd.to_datetime('2017-07-31'),
                          dependency=['TCA', 'TCL'], desc='流动比率 = 流动资产 / 流动负债'))

# 现金流净额与营业利润比


@drop_delist_data
def get_nopcf2opprofit(universe, start_time, end_time):
    '''
    ratio = 经营活动中产生的现金流净额TTM / 营业利润TTM
    '''
    cf = query('OPNETCF_TTM', (start_time, end_time))
    opprofit = query('OPPROFIT_TTM', (start_time, end_time))
    data = cf / opprofit
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('OPNETCF2OPPROFIT', get_nopcf2opprofit, pd.to_datetime('2017-07-31'),
                          dependency=['OPNETCF_TTM', 'OPPROFIT_TTM'],
                          desc='经营活动中产生的现金流净额TTM / 营业利润TTM'))

# 三费（财务费、管理费、销售费用）占销售比例


@drop_delist_data
def get_3fee2sale(universe, start_time, end_time):
    '''
    ratio = (销售费用TTM+管理费用TTM+财务费用TTM) / abs(营业收入)
    '''
    oprev = query('OPREV_TTM', (start_time, end_time))
    opexp = query('OPEXP_TTM', (start_time, end_time))
    adexp = query('ADMINEXP_TTM', (start_time, end_time))
    fiexp = query('FIEXP_TTM', (start_time, end_time))
    data = (opexp + adexp + fiexp) / np.abs(oprev)
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('FEE2SALE', get_3fee2sale, pd.to_datetime('2017-07-31'),
                          dependency=['OPREV_TTM', 'OPEXP_TTM', 'ADMINEXP_TTM', 'FIEXP_TTM'],
                          desc='(销售费用TTM+管理费用TTM+财务费用TTM) / abs(营业收入)'))
# --------------------------------------------------------------------------------------------------
# 动量因子


def get_momentum(days):
    '''
    母函数，用于生成计算动量的函数

    Parameter
    ---------
    days: int
        计算动量的交易日间隔
    '''
    @drop_delist_data
    def _inner(universe, start_time, end_time):
        start_time = pd.to_datetime(start_time)
        shift_days = int(days / 20 * 31)
        new_start = start_time - pd.Timedelta('30 day') - pd.Timedelta('%d day' % shift_days)
        quote = query('ADJ_CLOSE', (new_start, end_time))
        # pdb.set_trace()
        data = quote.pct_change(days).dropna(how='all')
        mask = (data.index >= start_time) & (data.index <= end_time)
        data = data.loc[mask, sorted(universe)]
        if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
            assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return _inner


# 1月动量，假设一个月有20个交易日
factor_list.append(Factor('MOM_1M', get_momentum(20), pd.to_datetime('2017-07-31'),
                          dependency=['ADJ_CLOSE']))
# 3月动量
factor_list.append(Factor('MOM_3M', get_momentum(60), pd.to_datetime('2017-07-31'),
                          dependency=['ADJ_CLOSE']))
factor_list.append(Factor('MOM_6M', get_momentum(120), pd.to_datetime('2018-03-15'),
                          dependency=['ADJ_CLOSE']))
# 60个月动量
factor_list.append(Factor('MOM_60M', get_momentum(1200), pd.to_datetime('2017-07-31'),
                          dependency=['ADJ_CLOSE']))
# 12个月动量
factor_list.append(Factor('MOM_12M', get_momentum(250), pd.to_datetime('2017-12-13'),
                          dependency=['ADJ_CLOSE'], desc='12个月动量'))

# --------------------------------------------------------------------------------------------------
# 市值中性化后因子


def get_mkvneu_factor(factor_name):
    '''
    母函数，用于生成计算市值中性化(相对LN_TMKV)后的动量的函数

    Parameter
    ---------
    factor_name: string
        需要中性化的因子的名称
    '''
    @drop_delist_data
    def _inner(universe, start_time, end_time):
        factor_data = query(factor_name, (start_time, end_time))
        # 市值因子肯定有数据，因此以其他因子的时间为准
        ln_cap = query('LN_TMKV', (start_time, end_time)).reindex(factor_data.index)
        # pdb.set_trace()
        data = convert_data([factor_data, ln_cap], ['factor', 'size'])

        def ols_neutralize(df):
            df = df.reset_index(0, drop=True).T
            raw_index = df.index
            mask = (~pd.isnull(df['factor'])) & (~pd.isnull(df['size']))
            if mask.sum() <= 200:   # 有效的数据量过少，直接返回nan
                return pd.Series(np.NaN, index=raw_index)
            y_data = df.loc[mask, 'factor']
            x_data = add_constant(df.loc[mask, 'size'])
            result = OLS(y_data, x_data).fit().resid
            return pd.Series(result, index=raw_index)

        by_date = data.groupby(level=0)
        result = by_date.apply(ols_neutralize)
        result = result.loc[:, sorted(universe)]
        # pdb.set_trace()
        return result
    return _inner


factor_list.append(Factor('MOM_1M_MKVNEU', get_mkvneu_factor('MOM_1M'), pd.to_datetime('2018-03-09'),
                          dependency=['MOM_1M', 'LN_TMKV'], desc='(对数)市值中性化后的动量因子'))
# --------------------------------------------------------------------------------------------------
# 偏度峰度因子
# 偏度


def gen_skfunc(days, func_name):
    '''
    母函数，用于生成计算偏度或者峰度的函数

    Parameter
    ---------
    days: int
        计算相关数据的交易日间隔
    func_name: str
        计算的数据结果类型，只支持skew、kurt和std
    '''
    func_category = {'skew': skew, 'kurt': kurtosis, 'std': np.std}
    func = func_category[func_name]

    @drop_delist_data
    def _inner(universe, start_time, end_time):
        start_time = pd.to_datetime(start_time)
        shift_days = int(days / 20 * 31)
        new_start = start_time - pd.Timedelta('30 day') - pd.Timedelta('%d day' % shift_days)
        data = query('DAILY_RET', (new_start, end_time))
        rolling = data.rolling(days, min_periods=days)
        data = rolling.apply(func)
        # data = data.dropna(how='all')
        mask = (data.index >= start_time) & (data.index <= end_time)
        data = data.loc[mask, sorted(universe)]
        if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
            checkdata_completeness(data, start_time, end_time)
        return data
    return _inner


factor_list.append(Factor('SKEW_1M', gen_skfunc(20, 'skew'), pd.to_datetime('2017-08-02'),
                          dependency=['DAILY_RET'], desc='过去20个交易日收益率的skew'))
factor_list.append(Factor('KURTOSIS_1M', gen_skfunc(20, 'kurt'), pd.to_datetime('2017-08-02'),
                          dependency=['DAILY_RET'], desc='过去20个交易日收益率的kurtosis'))
factor_list.append(Factor('VOL_1M', gen_skfunc(20, 'std'), pd.to_datetime('2017-11-30'),
                          dependency=['DAILY_RET'], desc='过去20个交易日收益率的标准差'))
# --------------------------------------------------------------------------------------------------
# 一致预期价格距离因子


@drop_delist_data
def get_conexpprice(universe, start_time, end_time):
    '''
    一致预期价格距离因子 = 一致预期目标价（在other因子模块中） / close - 1
    '''
    conprice = query('TARGET_PRICE', (start_time, end_time))
    close = query('CLOSE', (start_time, end_time))
    data = conprice / close - 1
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('CONEXP_DIS', get_conexpprice, pd.to_datetime('2017-08-04'),
                          dependency=['TARGET_PRICE', 'CLOSE'],
                          desc="一致预期价格距离因子 = 一致预期目标价（在other因子模块中） / close - 1"))
# --------------------------------------------------------------------------------------------------
# 前景理论因子


@drop_delist_data
def get_prospectfactor1w(universe, start_time, end_time):
    '''
    前景理论因子
    '''
    frequency = 5
    period = 60 * frequency  # 因子使用的行情数据的回溯期
    start_time = pd.to_datetime(start_time)
    new_start = dateshandle.tds_shift(start_time, period)
    # index_data = query('SSEC_CLOSE', (new_start, end_time))
    data = query('ADJ_CLOSE', (new_start, end_time))
    # 计算对应超额收益率
    # index_data = index_data.loc[:, data.columns]
    # index_data = index_data.pct_change(5).dropna(how='all')
    data = data.pct_change(frequency).dropna(how='all')
    # data = data - index_data
    data = data.dropna(how='all', axis=1)

    # 计算因子值
    def ts_ptvalue(ts_arr):
        '''
        计算单个股票的前景理论值
        '''

        def value_func(x):
            alpha = 0.88
            if x >= 0:
                lambda_ = 1
            else:
                lambda_ = -2.25
                x = -x
            return lambda_ * pow(x, alpha)

        def weight_func(p, delta):
            '''
            通用权重计算函数
            '''
            if p == 0:
                return 0
            else:
                return pow(p, delta) / pow(pow(p, delta) + pow(1 - p, delta), 1 / delta)

        # 正值加权函数，参数delta为0.61
        def weight_plus(p):
            return weight_func(p, 0.61)

        # 负值加权函数，参数delta为0.69
        def weight_minus(p):
            return weight_func(p, 0.69)

        # 计算每个收益点的加权后的前景值
        def tki_raw(r_k, total_len):
            '''
            r_k为tuple(idx, ret)
            '''
            # pdb.set_trace()
            if r_k[1] >= 0:
                w_func = weight_plus
                start_idx = total_len - r_k[0]
            else:
                w_func = weight_minus
                start_idx = r_k[0] + 1
            tmp = w_func(start_idx / total_len) - w_func((start_idx - 1) / total_len)
            return value_func(r_k[1]) * tmp

        # 计算一只股票当前的前景值
        def cal_tk(arr):
            # pdb.set_trace()
            arr = arr[frequency - 1::frequency]
            arr = list(enumerate(sorted(arr)))
            res = [tki_raw(r_k, len(arr)) for r_k in arr]
            # pdb.set_trace()
            return np.sum(res)
        return ts_arr.rolling(period, min_periods=period).apply(cal_tk)
    data = data.apply(ts_ptvalue)
    data = data.dropna(how='all')
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask, sorted(universe)]
    if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
        checkdata_completeness(data, start_time, end_time)
    return data


@drop_delist_data
def get_prospectfactor1wer(universe, start_time, end_time):
    '''
    前景理论因子
    '''
    frequency = 5
    period = 60 * frequency  # 因子使用的行情数据的回溯期
    start_time = pd.to_datetime(start_time)
    new_start = dateshandle.tds_shift(start_time, period)
    index_data = query('SSEC_CLOSE', (new_start, end_time))
    data = query('ADJ_CLOSE', (new_start, end_time))
    # 计算对应超额收益率
    index_data = index_data.loc[:, data.columns]
    index_data = index_data.pct_change(frequency).dropna(how='all')
    data = data.pct_change(frequency).dropna(how='all')
    data = data - index_data
    data = data.dropna(how='all', axis=1)

    # 计算因子值
    def ts_ptvalue(ts_arr):
        '''
        计算单个股票的前景理论值
        '''

        def value_func(x):
            alpha = 0.88
            if x >= 0:
                lambda_ = 1
            else:
                lambda_ = -2.25
                x = -x
            return lambda_ * pow(x, alpha)

        def weight_func(p, delta):
            '''
            通用权重计算函数
            '''
            if p == 0:
                return 0
            else:
                return pow(p, delta) / pow(pow(p, delta) + pow(1 - p, delta), 1 / delta)

        # 正值加权函数，参数delta为0.61
        def weight_plus(p):
            return weight_func(p, 0.61)

        # 负值加权函数，参数delta为0.69
        def weight_minus(p):
            return weight_func(p, 0.69)

        # 计算每个收益点的加权后的前景值
        def tki_raw(r_k, total_len):
            '''
            r_k为tuple(idx, ret)
            '''
            # pdb.set_trace()
            if r_k[1] >= 0:
                w_func = weight_plus
                start_idx = total_len - r_k[0]
            else:
                w_func = weight_minus
                start_idx = r_k[0] + 1
            tmp = w_func(start_idx / total_len) - w_func((start_idx - 1) / total_len)
            return value_func(r_k[1]) * tmp

        # 计算一只股票当前的前景值
        def cal_tk(arr):
            # pdb.set_trace()
            arr = arr[frequency - 1::frequency]
            arr = list(enumerate(sorted(arr)))
            res = [tki_raw(r_k, len(arr)) for r_k in arr]
            # pdb.set_trace()
            return np.sum(res)
        return ts_arr.rolling(period, min_periods=period).apply(cal_tk)
    data = data.apply(ts_ptvalue)
    data = data.dropna(how='all')
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask, sorted(universe)]
    if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
        checkdata_completeness(data, start_time, end_time)
    return data


factor_list.append(Factor('PT_VALUE_1W', get_prospectfactor1w, pd.to_datetime('2017-08-16'),
                          dependency=['ADJ_CLOSE'],
                          desc='前景理论因子（周频数据计算）'))
# ptvalue1weeker = Factor('PT_VALUE_1WER', get_prospectfactor1wer, pd.to_datetime('2017-10-09'),
#                         dependency=['ADJ_CLOSE', 'SSEC_CLOSE'], desc='前景理论因子（周频超额收益计算）')
# --------------------------------------------------------------------------------------------------
# 生成与CAPM相关的因子


def gen_capm_factor(handler):
    '''
    母函数，用于生成与滚动的CAPM模型相关的一些指标，比如说滚动的beta，滚动的特异波动率等

    Parameter
    ---------
    handler: function
        用于在每次滚动回归的时候，计算所需要的数据。函数的格式为handler(y, x, beta)-> float
        其中y为应变量是n*1的向量（n为滚动窗口的长度），x为自变量为n*2的矩阵，beta为2*1的向量
        （0的位置为截距项）
    '''
    @drop_delist_data
    def inner(universe, start_time, end_time):
        def moving_ols(y, x, window):
            '''
            滚动快速计算OLS
            '''
            # 添加截距项
            x = pd.DataFrame({'constant': [1] * len(x), 'x': x}, columns=['constant', 'x'])
            # pdb.set_trace()
            # 计算累计的xx和xy
            K = len(x.columns)
            N = len(x)
            last_xx = np.zeros((K, K))
            last_xy = np.zeros(K)
            cum_xx = []
            cum_xy = []
            for i in range(N):
                data_x = x.values[i: i + 1]
                data_y = y.values[i: i + 1]
                xy = np.dot(data_x.T, data_y)
                xx = np.dot(data_x.T, data_x)
                # 如果只有部分数据是NA，则也会导致后面无法算出beat从而产生BUG
                # 但是考虑到x一般是基准指数，不太会在中途出现NA值，而y是当前股票收益，也不太会中途出现NA值
                # 而且如果y为NA值，则xy全部都为NA值，因而不需要担心上述可能的BUG
                if np.all(np.isnan(last_xx)) and not np.all(np.isnan(xx)):  # 识别第一个有效X'X，并用其重置last_xx
                    last_xx = xx
                else:
                    last_xx = last_xx + np.dot(data_x.T, data_x)
                if np.all(np.isnan(last_xy)) and not np.all(np.isnan(xx)):
                    last_xy = xy
                else:
                    last_xy = last_xy + np.dot(data_x.T, data_y)
                cum_xy.append(last_xy)
                cum_xx.append(last_xx)
            # pdb.set_trace()
            # 计算滚动beta
            out = np.empty(N, dtype=float)
            out[:] = np.NaN
            for i in range(N):
                if i < window or np.any(pd.isnull(x.iloc[i])):
                    continue
                xx = cum_xx[i] - cum_xx[i - window]
                xy = cum_xy[i] - cum_xy[i - window]
                try:
                    beta = linalg.solve(xx, xy)
                    tmp_x = x.values[i + 1 - window: i + 1]
                    tmp_y = y.values[i + 1 - window: i + 1]
                    out[i] = handler(tmp_y, tmp_x, beta)

                except LinAlgError as e:    # 因为停牌等因素，股价一直都不变，此时的beta没有意义
                    continue
            # pdb.set_trace()
            return pd.Series(out, index=x.index)

        days = 252
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        new_start = dateshandle.tds_shift(start_time, days)
        stock_data = query('ADJ_CLOSE', (new_start, end_time))
        benchmark_data = query('SSEC_CLOSE', (new_start, end_time))
        stock_data = stock_data.pct_change().dropna(how='all').dropna(how='all', axis=1)
        benchmark_data = benchmark_data.iloc[:, 0].pct_change().dropna()
        # pdb.set_trace()
        # tqdm.pandas()
        data = stock_data.apply(lambda x: moving_ols(x, benchmark_data, days))
        mask = (data.index >= start_time) & (data.index <= end_time)
        data = data.loc[mask, sorted(universe)]
        if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
            checkdata_completeness(data, start_time, end_time)
        return data
    return inner
# beta因子


def beta_handler(y, x, beta):
    return beta[1]


factor_list.append(Factor('BETA', gen_capm_factor(beta_handler), pd.to_datetime('2017-09-04'),
                          dependency=['ADJ_CLOSE', 'SSEC_CLOSE'], desc='252交易日滚动beta系数'))


# 特质波动率因子
def idiosyncratic_handler(y, x, beta):
    resid = y - np.dot(x, beta)
    return np.std(resid)


factor_list.append(Factor('SPECIAL_VOL', gen_capm_factor(idiosyncratic_handler),
                          pd.to_datetime('2017-09-05'), dependency=['ADJ_CLOSE', 'SSEC_CLOSE'],
                          desc='特质波动率'))


# 系统风险占比因子（systemic risk ratio）
np.seterr('raise')


def srr_handler(y, x, beta):
    resid = y - np.dot(x, beta)
    try:
        out = 1 - np.var(resid) / np.var(y)  # 停牌的股票长期的波动为0，视为无效数据
    except FloatingPointError:
        out = np.nan
    return out


factor_list.append(Factor('SYSRISK_RATIO', gen_capm_factor(srr_handler),
                          pd.to_datetime('2017-12-21'), dependency=['ADJ_CLOSE', 'SSEC_CLOSE'],
                          desc='系统风险占比因子'))

# --------------------------------------------------------------------------------------------------
# 修正后的与CAPM相关的因子
def gen_capm_factor_ffi(handler, cycle=252):
    '''
    母函数，用于生成与滚动的CAPM模型相关的一些指标，比如说滚动的beta，滚动的特异波动率等

    Parameter
    ---------
    handler: function
        用于在每次滚动回归的时候，计算所需要的数据。函数的格式为handler(y, x, beta)-> float
        其中y为应变量是n*1的向量（n为滚动窗口的长度），x为自变量为n*2的矩阵，beta为2*1的向量
        （0的位置为截距项）
    '''
    @drop_delist_data
    def inner(universe, start_time, end_time):
        def moving_ols(y, x, window):
            '''
            滚动快速计算OLS
            '''
            # 添加截距项
            x = pd.DataFrame({'constant': [1] * len(x), 'x': x}, columns=['constant', 'x'])
            # pdb.set_trace()
            # 计算累计的xx和xy
            K = len(x.columns)
            N = len(x)
            last_xx = np.zeros((K, K))
            last_xy = np.zeros(K)
            cum_xx = []
            cum_xy = []
            for i in range(N):
                data_x = x.values[i: i + 1]
                data_y = y.values[i: i + 1]
                xy = np.dot(data_x.T, data_y)
                xx = np.dot(data_x.T, data_x)
                # 如果只有部分数据是NA，则也会导致后面无法算出beat从而产生BUG
                # 但是考虑到x一般是基准指数，不太会在中途出现NA值，而y是当前股票收益，也不太会中途出现NA值
                # 而且如果y为NA值，则xy全部都为NA值，因而不需要担心上述可能的BUG
                if np.all(np.isnan(last_xx)) and not np.all(np.isnan(xx)):  # 识别第一个有效X'X，并用其重置last_xx
                    last_xx = xx
                else:
                    last_xx = last_xx + np.dot(data_x.T, data_x)
                if np.all(np.isnan(last_xy)) and not np.all(np.isnan(xx)):
                    last_xy = xy
                else:
                    last_xy = last_xy + np.dot(data_x.T, data_y)
                cum_xy.append(last_xy)
                cum_xx.append(last_xx)
            # pdb.set_trace()
            # 计算滚动beta
            out = np.empty(N, dtype=float)
            out[:] = np.NaN
            for i in range(N):
                if i < window or np.any(pd.isnull(x.iloc[i])):
                    continue
                xx = cum_xx[i] - cum_xx[i - window]
                xy = cum_xy[i] - cum_xy[i - window]
                try:
                    beta = linalg.solve(xx, xy)
                    tmp_x = x.values[i + 1 - window: i + 1]
                    tmp_y = y.values[i + 1 - window: i + 1]
                    out[i] = handler(tmp_y, tmp_x, beta)

                except LinAlgError as e:    # 因为停牌等因素，股价一直都不变，此时的beta没有意义
                    continue
            # pdb.set_trace()
            return pd.Series(out, index=x.index)

        days = cycle
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        new_start = dateshandle.tds_shift(start_time, days)
        stock_data = query('ADJ_CLOSE', (new_start, end_time))
        benchmark_data = query('CSIFFI_CLOSE', (new_start, end_time))
        stock_data = stock_data.pct_change().dropna(how='all').dropna(how='all', axis=1)
        benchmark_data = benchmark_data.iloc[:, 0].pct_change().dropna()
        # pdb.set_trace()
        # tqdm.pandas()
        data = stock_data.apply(lambda x: moving_ols(x, benchmark_data, days))
        mask = (data.index >= start_time) & (data.index <= end_time)
        data = data.loc[mask, sorted(universe)]
        if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
            checkdata_completeness(data, start_time, end_time)
        return data
    return inner


factor_list.append(Factor('BETA_FFI', gen_capm_factor_ffi(beta_handler), pd.to_datetime('2018-06-07'),
                          dependency=['ADJ_CLOSE', 'CSIFFI_CLOSE'], desc='252交易日滚动beta系数，使用中证流通指数计算'))


factor_list.append(Factor('SPECIAL_VOL_FFI', gen_capm_factor_ffi(idiosyncratic_handler),
                          pd.to_datetime('2018-06-07'), dependency=['ADJ_CLOSE', 'CSIFFI_CLOSE'],
                          desc='特质波动率，使用中证流通指数计算'))
factor_list.append(Factor('SPECIAL_VOL_FFI_120', gen_capm_factor_ffi(idiosyncratic_handler, 120),
                          pd.to_datetime('2018-06-25'), dependency=['ADJ_CLOSE', 'CSIFFI_CLOSE'],
                          desc='特质波动率，使用中证流通指数计算，计算时间周期为120个交易日'))
factor_list.append(Factor('SPECIAL_VOL_FFI_60', gen_capm_factor_ffi(idiosyncratic_handler, 60),
                          pd.to_datetime('2018-06-25'), dependency=['ADJ_CLOSE', 'CSIFFI_CLOSE'],
                          desc='特质波动率，使用中证流通指数计算，计算时间周期为60个交易日'))
factor_list.append(Factor('SPECIAL_VOL_FFI_30', gen_capm_factor_ffi(idiosyncratic_handler, 30),
                          pd.to_datetime('2018-06-25'), dependency=['ADJ_CLOSE', 'CSIFFI_CLOSE'],
                          desc='特质波动率，使用中证流通指数计算，计算时间周期为30个交易日'))



# --------------------------------------------------------------------------------------------------
# 机构持有比例


def get_institutions_holding(data_category):
    '''
    母函数，用于获取给定类型的机构持有比例

    Parameter
    ---------
    data_category: str
        机构持有比例数据的类型，包含无限售流通A股比例和持有A股比例，输入参数的映射规则如下
        {'unconstrained': InstitutionsHoldProp, 'all': InstitutionsHoldPropA}

    Return
    ------
    out: function
        获取对应数据的函数
    '''
    parameter_map = {'unconstrained': 'InstitutionsHoldProp', 'all': 'InstitutionsHoldPropA'}
    sql = '''
        SELECT M.SECUCODE, S.ENDDATE, S.data_category
        FROM SECUMAIN M, LC_StockHoldingSt S
        WHERE
            M.INNERCODE = S.INNERCODE AND
            M.SecuMarket in (83, 90) AND
            M.SecuCategory = 1 AND
            S.ENDDATE >= \'{start_time}\' AND
            S.ENDDATE <= \'{end_time}\'
        ORDER BY M.SECUCODE ASC, S.ENDDATE ASC
        '''.replace('data_category', parameter_map[data_category])

    @drop_delist_data
    def inner(universe, start_time, end_time):
        new_start = dateshandle.tds_shift(start_time, 120)
        data = fdgetter.get_db_data(sql, start_time=new_start, end_time=end_time,
                                    cols=('code', 'time', 'data'), add_stockcode=False)
        data['code'] = data.code.apply(datatoolkits.add_suffix)
        data['data'] = data.data.fillna(0)
        tds = dateshandle.get_tds(start_time, end_time)
        data = data.groupby('code').apply(datatoolkits.map_data, days=tds, fromNowOn=True,
                                          fillna={'code': lambda x: x.code.iloc[0]})
        data = data.reset_index(drop=True)
        data = data.pivot_table('data', index='time', columns='code')
        data = data.loc[:, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
        return data
    return inner

# 聚源数据的计算方法导致无法计算限售占比
# def get_constrained_ihr(universe, start_time, end_time):
#     '''
#     获取机构持有的限售A股比例
#     '''
#     all_data = query('ALL_INSTIHOLDING_RATIO', (start_time, end_time))
#     uncons_data = query('UNCONS_INSTIHOLDING_RATIO', (start_time, end_time))
#     data = all_data - uncons_data
#     assert not np.any(np.any(data < 0, axis=0)), 'Error, negative holding ratio!'
#     data = data.loc[:, sorted(universe)]
#     assert check_indexorder(data), 'Error, data order is mixed!'
#     assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
#     return data


factor_list.append(Factor('UNCONS_INSTIHOLDING_RATIO', get_institutions_holding('unconstrained'),
                          pd.to_datetime('2017-10-13'), desc='机构非限售流通A股机构持有比例'))
factor_list.append(Factor('ALL_INSTIHOLDING_RATIO', get_institutions_holding('all'),
                          pd.to_datetime('2017-10-13'), desc='机构持有的A股比例'))
# cons_instiholdingratio = Factor('CONS_INSTIHOLDING_RATIO', get_constrained_ihr,
#                                 pd.to_datetime('2017-10-13'),
#                                 dependency=['UNCONS_INSTIHOLDING_RATIO', 'ALL_INSTIHOLDING_RATIO'],
#                                 desc='机构持有限售A股比例')
# --------------------------------------------------------------------------------------------------
# 相对强度


@drop_delist_data
def get_rstr(universe, start_time, end_time):
    '''
    BARRA RSTR因子
    无风险利率假定为0，暂时无法从数据库中获取无风险收益率相关数据
    '''
    start_time = pd.to_datetime(start_time)
    lag = 21
    period = 504
    half_life = 126
    decay_rate = 0.5**(1 / half_life)
    weight = np.array([decay_rate**i if i >= lag else 0
                       for i in range(period + lag, 0, -1)])
    weight = weight / np.sum(weight)
    new_start = dateshandle.tds_shift(start_time, period + lag)
    ret_data = query('DAILY_RET', (new_start, end_time))

    def calc_rollingrstr(ts):
        # 滚动计算单股票的rstr
        ts = np.log(1 + ts)
        out = ts.rolling(lag + period).apply(lambda x: x.dot(weight))
        return out
    data = ret_data.apply(calc_rollingrstr)
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask, sorted(universe)]
    if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
        checkdata_completeness(data, start_time, end_time)
    return data


factor_list.append(Factor('RSTR', get_rstr, pd.to_datetime('2017-10-17'), dependency=['DAILY_RET'],
                          desc='BARRA RSTR因子'))
# --------------------------------------------------------------------------------------------------
# 日波动率


@drop_delist_data
def get_dstd(universe, start_time, end_time):
    '''
    BARRA DSTD
    '''
    start_time = pd.to_datetime(start_time)
    period = 252
    half_life = 42
    decay_rate = 0.5**(1 / half_life)
    weight = np.array([decay_rate**i for i in range(period, 0, -1)])
    weight = weight / np.sum(weight)

    def calc_dstd(ts):
        return ts.rolling(period).apply(lambda x: np.sqrt(np.dot(np.power(x - np.mean(x), 2),
                                                                 weight)))
    new_start = dateshandle.tds_shift(start_time, period)
    ret_data = query('DAILY_RET', (new_start, end_time))
    data = ret_data.apply(calc_dstd)
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask, sorted(universe)]
    if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
        checkdata_completeness(data, start_time, end_time)
    return data


factor_list.append(Factor('DSTD', get_dstd, pd.to_datetime('2017-10-17'), dependency=['DAILY_RET'],
                          desc='BARRA DSTD因子'))
# --------------------------------------------------------------------------------------------------
# BARRA CMRA


@drop_delist_data
def get_cmra(universe, start_time, end_time):
    '''
    BARRA CMRA因子（累计波动幅度）
    '''
    monthly_td = 21
    month_cnt = 12

    start_time = pd.to_datetime(start_time)
    new_start = dateshandle.tds_shift(start_time, month_cnt * monthly_td)
    quote_data = query('ADJ_CLOSE', (new_start, end_time))
    ret_data = quote_data.pct_change(monthly_td)
    idx_slice = slice(-1, -month_cnt * monthly_td, -monthly_td)

    def get_single_cmra(df):
        # 计算单股票的滚动CMRA，使用修改后的算法，原报告中的算法会导致股票大跌后出现NA值
        def single_period_cmra(ts):
            # pdb.set_trace()
            valid_data = ts[idx_slice]
            cum_ret = np.cumprod(1 + valid_data) - 1
            rng = np.log(1 + np.max(cum_ret)) - np.log(1 + np.min(cum_ret))
            # if pd.isnull(rng):
            #     pdb.set_trace()
            return rng
        # pdb.set_trace()
        res = df.rolling(monthly_td * month_cnt, min_periods=monthly_td * month_cnt).\
            apply(single_period_cmra)
        # pdb.set_trace()
        return res

    data = ret_data.apply(get_single_cmra)
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask, sorted(universe)]
    if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
        checkdata_completeness(data, start_time, end_time)
    return data


factor_list.append(Factor('CMRA', get_cmra, pd.to_datetime('2017-10-19'),
                          dependency=['ADJ_CLOSE'], desc='BARRA CMRA因子'))
# --------------------------------------------------------------------------------------------------
# BARRA LEVERAGE


@drop_delist_data
def get_mlev(universe, start_time, end_time):
    '''
    BARRA MLEV因子，不考虑优先股，因为金融行业TNCL是NA值，因此同样金融行业该因子也是NA值
    '''
    total_mkv = query('TOTAL_MKTVALUE', (start_time, end_time))
    ldebt = query('TNCL', (start_time, end_time))
    prefer_stock = query('PREFER_STOCK', (start_time, end_time))
    data = convert_data([total_mkv, ldebt, prefer_stock], ['total_mkv', 'ldebt', 'prefer_stock'])
    data = data.groupby(level=0).apply(lambda x: x.sum(axis=0))
    data = data / total_mkv
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('MLEV', get_mlev, pd.to_datetime('2017-10-27'),
                          dependency=['TOTAL_MKTVALUE', 'TNCL', 'PREFER_STOCK'], desc='BARRA MLEV因子'))


@drop_delist_data
def get_dtoa(universe, start_time, end_time):
    '''
    BARRA DTOA因子
    '''
    tasset = query('TA', (start_time, end_time))
    ldebt = query('TNCL', (start_time, end_time))
    sdebt = query('TCL', (start_time, end_time))
    data = convert_data([ldebt, sdebt], ['ldebt', 'sdebt'])
    data = data.groupby(level=0).apply(lambda x: x.sum(axis=0))
    data = data / tasset
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('DTOA', get_dtoa, pd.to_datetime('2017-10-27'),
                          dependency=['TA', 'TNCL', 'TCL'], desc='BARRA DTOA因子'))


@drop_delist_data
def get_blev(universe, start_time, end_time):
    '''
    BARRA BLEV因子
    '''
    equity = query('EQUITY', (start_time, end_time))
    ldebt = query('TNCL', (start_time, end_time))
    prefer_stock = query('PREFER_STOCK', (start_time, end_time))
    data = convert_data([equity, ldebt, prefer_stock], ['equity', 'ldebt', 'prefer_stock'])
    data = data.groupby(level=0).apply(lambda x: x.sum(axis=0))
    data = data / equity
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('BLEV', get_blev, pd.to_datetime('2017-10-27'),
                          dependency=['PREFER_STOCK', 'EQUITY', 'TNCL'], desc='BARRA BLEV因子'))
# --------------------------------------------------------------------------------------------------
# Composite Equity Issues


@drop_delist_data
def get_cei(universe, start_time, end_time):
    '''
    Composite Equity Issues因子，来源于Daniel和Titman(2006)的论文
    相比于原论文有小幅度的改动
    因子值 = (过去250个交易日总市值的变化率) - (过去250个交易日（复权后）收盘价变动)
    '''
    offset = 250
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    new_start = dateshandle.tds_shift(start_time, offset)
    total_mktv = query('TOTAL_MKTVALUE', (new_start, end_time))
    adj_close = query('ADJ_CLOSE', (new_start, end_time))
    chg_tmktv = total_mktv.pct_change(offset)
    chg_ac = adj_close.pct_change(offset)
    data = chg_tmktv - chg_ac
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    checkdata_completeness(data, start_time, end_time)
    return data


factor_list.append(Factor('CEI', get_cei, pd.to_datetime('2017-12-04'),
                          dependency=['TOTAL_MKTVALUE', 'ADJ_CLOSE'],
                          desc='Composite Equity Issues因子'))

# --------------------------------------------------------------------------------------------------
# 总资产增长率


@drop_delist_data
def get_ta_yoy(universe, start_time, end_time):
    '''
    Asset growth因子，来源于Cooper, Gulen和Schill(2008)的论文
    '''
    ta1 = query('TA', (start_time, end_time))
    ta2 = query('TA_2Y', (start_time, end_time))
    data = (ta1 - ta2) / np.abs(ta2)
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    checkdata_completeness(data, start_time, end_time)
    return data


factor_list.append(Factor('TA_YOY', get_ta_yoy, pd.to_datetime('2017-12-04'),
                          dependency=['TA', 'TA_2Y'], desc='总资产年度增长率（YOY）'))
# --------------------------------------------------------------------------------------------------
# 均线因子，用于辅助计算多期限均线因子


def gen_ma(offset):
    '''
    母函数，用于生成给定日期的均线数据
    此处，均线定义为MA普通均线，然后通过当前日期的收盘价正则化

    Parameter
    ---------
    offset: int
        均线的长度
    '''
    @drop_delist_data
    def inner(universe, start_time, end_time):
        new_start = dateshandle.tds_shift(start_time, offset)
        close_data = query('ADJ_CLOSE', (new_start, end_time))
        ma = close_data.rolling(offset, min_periods=offset).mean()
        data = ma / close_data
        mask = (data.index >= pd.to_datetime(start_time)) & (data.index <= pd.to_datetime(end_time))
        data = data.loc[mask, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        checkdata_completeness(data, start_time, end_time)
        return data
    return inner


for offset in [3, 5, 10, 20, 50, 100, 200]:
    factor_list.append(Factor('MA%d' % offset, gen_ma(offset), pd.to_datetime('2017-12-06'),
                              dependency=['ADJ_CLOSE'],
                              desc='经过收盘价正则化后的%d日均线' % offset))


# --------------------------------------------------------------------------------------------------
# ILLIQ因子
def get_illiq(universe, start_time, end_time):
    '''
    获取ILLIQ因子，来源于论文：Time-Varying Liquidity and Momentum Profits
    修改后具体计算方法为：
    1/n * sum(r_close_t/volume_value_t)
    '''
    lag = 22
    new_start = dateshandle.tds_shift(start_time, lag)
    end_time = pd.to_datetime(end_time)
    start_time = pd.to_datetime(start_time)
    close_data = query('ADJ_CLOSE', (new_start, end_time))
    volume_data = query('TO_VALUE', (new_start, end_time))
    multiple = 10 ** 9
    ret_data = np.abs(close_data.pct_change() * multiple)
    data = ret_data / volume_data
    data = data.rolling(lag, min_periods=lag).mean()
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask].reindex(columns=sorted(universe))
    assert check_indexorder(data), 'Error, data order is mixed!'
    checkdata_completeness(data, start_time, end_time)
    return data


factor_list.append(Factor('ILLIQ', get_illiq, pd.to_datetime('2018-03-26'),
                          dependency=['ADJ_CLOSE', 'TO_VALUE'], desc='非流动性因子'))


# --------------------------------------------------------------------------------------------------
# FF因子组合收益率
@drop_delist_data
def get_smb(universe, start_time, end_time):
    '''
    SMB因子收益率，按月换仓
    每个月根据最后一个交易日的数据计算出低20%和高20%的组合，然后分别计算组合的净值，
    利用净值计算组合收益率，SMB因子的收益率为低20%减去高20%，收益以总市值加权计算
    '''
    calendar = get_calendar('stock.sse')
    new_start_time = calendar.shift_tradingdays(start_time, -30)
    adj_close = query('ADJ_CLOSE', (new_start_time, end_time))
    mkv = query('TOTAL_MKTVALUE', (new_start_time, end_time))
    reb_dates = calendar.get_cycle_targets(new_start_time, end_time)
    mkv = mkv.reindex(reb_dates)
    low_rets = []
    high_rets = []
    # pdb.set_trace()
    for i, date in enumerate(reb_dates):
        tmp_mkv = mkv.loc[date].dropna()
        if len(tmp_mkv) == 0:
            continue
        high_mkv = tmp_mkv[tmp_mkv > tmp_mkv.quantile(0.8)]
        low_mkv = tmp_mkv[tmp_mkv < tmp_mkv.quantile(0.2)]
        if i < len(reb_dates) - 1:
            tmp_end = reb_dates[i+1]
        else:
            tmp_end = None
        tmp_close = adj_close[date: tmp_end]
        tmp_close = (tmp_close / tmp_close.iloc[0]).fillna(method='ffill')
        high_mkv = high_mkv / high_mkv.sum()
        low_mkv = low_mkv / low_mkv.sum()
        high_nav = tmp_close.loc[:, high_mkv.index].multiply(high_mkv).sum(axis=1)
        low_nav = tmp_close.loc[:, low_mkv.index].multiply(low_mkv).sum(axis=1)
        high_ret = high_nav.pct_change().dropna()
        low_ret = low_nav.pct_change().dropna()
        low_rets.append(low_ret)
        high_rets.append(high_ret)
    low_rets = pd.concat(low_rets, axis=0).sort_index()
    high_rets = pd.concat(high_rets, axis=0).sort_index()
    rets = low_rets - high_rets
    rets = rets.loc[(rets.index >= pd.to_datetime(start_time)) & (rets.index <= pd.to_datetime(end_time))]
    rets = pd.DataFrame(np.repeat([rets.values], len(universe), axis=0).T,
                        index=rets.index, columns=sorted(universe))
    if pd.to_datetime(start_time) > pd.to_datetime(START_TIME):
        assert check_indexorder(rets), 'Error, data order is mixed!'
        checkdata_completeness(rets, start_time, end_time)
    return rets

factor_list.append(Factor('FF_SMB', get_smb, pd.to_datetime('2018-06-26'), ['ADJ_CLOSE', 'TOTAL_MKTVALUE', 'LIST_STATUS'],
                          'FF三因子模型SMB因子收益'))

@drop_delist_data
def get_hml(universe, start_time, end_time):
    '''
    HML因子收益率，按月换仓
    每个月最后一个交易日，首先根据市值分为高于50%和低于50%两组，在每组内，分别计算出BP高20%和低20%
    组合，分别计算这些组合的净值，然后计算收益率，HML因子的收益率为组内高20%减低20%，然后组间平均
    '''
    calendar = get_calendar('stock.sse')
    new_start_time = calendar.shift_tradingdays(start_time, -30)
    reb_dates = calendar.get_cycle_targets(new_start_time, end_time)
    adj_close = query('ADJ_CLOSE', (new_start_time, end_time))
    bp = query('BP', (new_start_time, end_time))
    mkv = query('TOTAL_MKTVALUE', (new_start_time, end_time))
    bp = bp.reindex(reb_dates)
    mkv = mkv.reindex(reb_dates)
    rets = {'big_high': [], 'small_high': [], 'big_low': [], 'small_low': []}
    group_map = {'big_high': [1, 4], 'small_high': [0, 4], 'big_low': [1, 0], 'small_low': [0, 0]}
    for i, date in enumerate(reb_dates):
        tmp_mkv = mkv.loc[date].dropna()
        tmp_bp = bp.loc[date].dropna()
        data = pd.DataFrame({'mkv': tmp_mkv, 'bp': tmp_bp}).dropna(axis=0)
        if len(data) == 0:
            continue
        if i < len(reb_dates) - 1:
            tmp_end = reb_dates[i+1]
        else:
            tmp_end = None
        data = data.assign(mkv_group=pd.qcut(data.mkv, 2, range(2)))
        data = data.assign(bp_group=data.groupby('mkv_group').bp.transform(lambda x: pd.qcut(x, 5, range(5))))
        for group in group_map:
            pos_map = group_map[group]
            group_mkv = data.loc[(data.mkv_group==pos_map[0])&(data.bp_group==pos_map[1]), 'mkv']
            group_mkv = group_mkv / group_mkv.sum()
            tmp_close = adj_close.loc[date: tmp_end].fillna(method='ffill')
            tmp_close = tmp_close / tmp_close.iloc[0]
            tmp_nav = tmp_close.loc[:, group_mkv.index].multiply(group_mkv).sum(axis=1)
            tmp_ret = tmp_nav.pct_change().dropna()
            rets[group].append(tmp_ret)
    rets = {g: pd.concat(rets[g], axis=0).sort_index() for g in rets}
    data = (rets['big_high'] - rets['big_low']) * 0.5 + (rets['small_high'] - rets['small_low']) * 0.5
    data = data.loc[(data.index >= pd.to_datetime(start_time)) & (data.index <= pd.to_datetime(end_time))]
    data = pd.DataFrame(np.repeat([data.values], len(universe), axis=0).T,
                        index=data.index, columns=sorted(universe))
    if pd.to_datetime(start_time) > pd.to_datetime(START_TIME):
        assert check_indexorder(data), 'Error, data order is mixed!'
        checkdata_completeness(data, start_time, end_time)
    return data

factor_list.append(Factor('FF_HML', get_hml, pd.to_datetime('2018-06-26'), ['BP', 'ADJ_CLOSE', 'TOTAL_MKTVALUE', 'LIST_STATUS'],
                          'FF三因子模型HML因子收益'))

# --------------------------------------------------------------------------------------------------
# FF特异波动率
def get_ff_idio(cycle=252):
    '''
    使用FF三因素模型计算特异波动率
    '''
    @drop_delist_data
    def inner(universe, start_time, end_time):
        def moving_ols(y, x, hml, smb, window):
            '''
            滚动快速计算OLS
            '''
            # 添加截距项
            x = pd.DataFrame({'constant': [1] * len(x), 'x': x, 'hml': hml, 'smb': smb}, columns=['constant', 'x', 'hml', 'smb'])
            # pdb.set_trace()
            # 计算累计的xx和xy
            K = len(x.columns)
            N = len(x)
            last_xx = np.empty((K, K))
            last_xy = np.empty(K)
            last_xx[:] = np.nan
            last_xy[:] = np.nan
            cum_xx = []
            cum_xy = []
            for i in range(N):
                data_x = x.values[i: i + 1]
                data_y = y.values[i: i + 1]
                xy = np.dot(data_x.T, data_y)
                xx = np.dot(data_x.T, data_x)
                # pdb.set_trace()
                # 如果只有部分数据是NA，则也会导致后面无法算出beat从而产生BUG
                # 但是考虑到x一般是基准指数，不太会在中途出现NA值，而y是当前股票收益，也不太会中途出现NA值
                # 而且如果y为NA值，则xy全部都为NA值，因而不需要担心上述可能的BUG
                if (np.any(np.isnan(last_xx)) and not np.any(np.isnan(xx))) or\
                   (np.any(np.isnan(last_xy)) and not np.any(np.isnan(xx))):    # 识别第一个没有任何NA值的xx和xy对
                    last_xx = xx
                    last_xy = xy
                else:
                    last_xx = last_xx + np.dot(data_x.T, data_x)
                    last_xy = last_xy + np.dot(data_x.T, data_y)
                cum_xy.append(last_xy)
                cum_xx.append(last_xx)
            # pdb.set_trace()
            # 计算滚动beta
            out = np.empty(N, dtype=float)
            out[:] = np.NaN
            for i in range(N):
                if i < window or np.any(pd.isnull(x.iloc[i])):
                    continue
                xx = cum_xx[i] - cum_xx[i - window]
                xy = cum_xy[i] - cum_xy[i - window]
                try:
                    beta = linalg.solve(xx, xy)
                    # pdb.set_trace()
                    tmp_x = x.values[i + 1 - window: i + 1]
                    tmp_y = y.values[i + 1 - window: i + 1]
                    if np.all(np.isclose(beta, 0)):    # 因为周期过短会导致计算出的系数为0，因而特异波动率也会为0
                        continue
                    out[i] = np.std(tmp_y - np.dot(tmp_x, beta))
                except LinAlgError as e:    # 因为停牌等因素，股价一直都不变，此时的beta没有意义
                    continue
            # pdb.set_trace()
            return pd.Series(out, index=x.index)

        days = cycle
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        new_start = dateshandle.tds_shift(start_time, days)
        stock_data = query('ADJ_CLOSE', (new_start, end_time))
        benchmark_data = query('CSIFFI_CLOSE', (new_start, end_time))
        stock_data = stock_data.pct_change().dropna(how='all').dropna(how='all', axis=1)
        benchmark_data = benchmark_data.iloc[:, 0].pct_change().dropna()
        hml_data = query('FF_HML', (new_start, end_time)).loc[stock_data.index[0]:, '000001.SZ']
        smb_data = query('FF_SMB', (new_start, end_time)).loc[stock_data.index[0]:, '000001.SZ']
        # pdb.set_trace()
#         tqdm.pandas()
        data = stock_data.apply(lambda x: moving_ols(x, benchmark_data, hml_data, smb_data, days))
#         data = stock_data.progress_apply(lambda x: moving_ols(x, benchmark_data, hml_data, smb_data, days))
        mask = (data.index >= start_time) & (data.index <= end_time)
        data = data.loc[mask, sorted(universe)]
        if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
            checkdata_completeness(data, start_time, end_time)
        return data
    return inner
factor_list.append(Factor('FF_SPECIAL_VOL_30', get_ff_idio(30), pd.to_datetime('2018-06-26'),
                          ['ADJ_CLOSE', 'CSIFFI_CLOSE', 'FF_HML', 'FF_SMB', 'LIST_STATUS'],
                          '使用FF三因子模型计算的特异波动率'))

# --------------------------------------------------------------------------------------------------
# BAC论文，改进的SMAX因子
def gen_smax(lookback_period, ret_freq, max_cnt):
    '''
    母函数，用于生成BAC论文中的SMAX因子，该因子主要用于捕捉股票在剔除波动率后的彩票效应，或者说偏度相关的效应，
    因子值=过去一段时间最大的max_cnt个收益率的平均值/过去一段时间收益的波动率

    Parameter
    ---------
    lookback_period: int
        计算因子所使用的历史数据的长度
    ret_freq: int
        因为国内涨停规则的限制，导致一个交易日最大的收益限制在10%，因此需要多几日的收益来反映该偏度效应，该参数
        即为计算收益所使用的天数
    max_cnt: int
        计算最大收益率的平均值所使用的数据的数量

    Return
    ------
    func: function(universe, start_time, end_time)->pandas.DataFrame

    Notes
    -----
    该因子采用滚动计算的方式，即收益为以ret_freq为周期的滚动，波动为由收益计算而来，因此考虑到重叠带来的相关性，
    这种计算方式会导致波动相对于单日计算出的波动存在膨胀效应
    '''
    def inner(universe, start_time, end_time):
        new_start_time = get_calendar('stock.sse').shift_tradingdays(start_time, -(lookback_period+2*ret_freq))
        adj_close = query('ADJ_CLOSE', (new_start_time, end_time))
        rets = adj_close.pct_change(ret_freq)
        vol = rets.rolling(lookback_period, min_periods=lookback_period).std()
        mask = ~np.isclose(vol, 0, atol=1.e-6)
        vol = vol.where(mask, np.nan)
        avg_nlargest_rets = rets.rolling(lookback_period, min_periods=lookback_period).\
                            apply(lambda x: np.mean(np.sort(x)[-max_cnt:]))
        data = avg_nlargest_rets / vol
        data = data.loc[(data.index >= pd.to_datetime(start_time)) & (data.index <= pd.to_datetime(end_time)), sorted(universe)]
        checkdata_completeness(data, start_time, end_time)
        return data
    return inner

factor_list.append(Factor('SMAX_M', gen_smax(30, 3, 5), pd.to_datetime('2018-06-27'), ['ADJ_CLOSE'], '按月度数据计算的SMAX因子'))
factor_list.append(Factor('SMAX_MD', gen_smax(30, 1, 5), pd.to_datetime('2018-06-27'), ['ADJ_CLOSE'], '按月度数据计算的SMAX因子，使用日收益率数据'))

# --------------------------------------------------------------------------------------------------
#  计算指数成分与指数的相关性
def gen_cons_corr(index_quote_name, window):
    '''
    母函数，用于生成计算股票与某个指数之间的相关性

    Parameter
    ---------
    index_quote_name: string
        指数行情的内部名称
    window: int
        计算相关性的时间窗口的长度
    
    Return
    ------
    func: function(universe, start_time, end_time)->pandas.DataFrame
    '''
    def inner(universe, start_time, end_time):
        start_time, end_time = trans_date(start_time, end_time)
        shifted_start_time = get_calendar('stock.sse').shift_tradingdays(start_time, -window - 1)
        index_ret = query(index_quote_name, (shifted_start_time, end_time)).iloc[:, 0].pct_change().dropna()
        quotes = query('ADJ_CLOSE', (shifted_start_time, end_time))
        rets = quotes.pct_change().dropna(axis=0, how='all')
        def calc_corr(data):
            # 计算当前股票与给定指数收益的相关性
            tmp_data = pd.DataFrame({'data': data, 'index': index_ret}, columns=['data', 'index'])
            # pdb.set_trace()
            def corr_func(x):
                if np.std(x[:, 0]) == 0:
                    return np.nan
                return np.corrcoef(x[:, 0], x[:, 1])[0, 1]
            res = rolling_apply(tmp_data, corr_func, window)
            return res
        # tqdm.pandas()
        # res = rets.progress_apply(calc_corr, axis=0)
        res = rets.apply(calc_corr, axis=0)
        mask = (res.index >= start_time) & (res.index <= end_time)
        res = res.loc[mask, sorted(universe)]
        checkdata_completeness(res, start_time, end_time)
        return res
    return inner

factor_list.append(Factor('SSEC_CORR60', gen_cons_corr('SSEC_CLOSE', 60), pd.to_datetime('2018-09-20'), ['SSEC_CLOSE', 'ADJ_CLOSE'], '股票与上证综指的相关系数'))


# --------------------------------------------------------------------------------------------------

check_duplicate_factorname(factor_list, __name__)


if __name__ == '__main__':
    pass

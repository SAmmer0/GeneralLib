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
from scipy.stats import skew, kurtosis
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
# from tqdm import tqdm
from fmanager.const import START_TIME
from fmanager.factors.utils import (Factor, check_indexorder, check_duplicate_factorname,
                                    convert_data, checkdata_completeness)
from fmanager.factors.query import query
import fdgetter
import datatoolkits

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


def get_oprev_yoy(universe, start_time, end_time):
    '''
    OPREV_YOY = (本季度营业收入-上年同季度营业收入)/abs(上年同季度营业收入)
    '''
    oprev_lq = query('OPREV_1S', (start_time, end_time))
    oprev_lyq = query('OPREV_5S', (start_time, end_time))
    data = (oprev_lq - oprev_lyq) / np.abs(oprev_lyq) - 1
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    assert checkdata_completeness(data, start_time, end_time), "Error, data missed!"
    return data


factor_list.append(Factor('OPREV_YOY', get_oprev_yoy, pd.to_datetime('2017-07-27'),
                          dependency=['OPREV_1S', 'OPREV_5S'],
                          desc='(本季度营业收入-上年同季度营业收入)/abs(上年同季度营业收入)'))
# 单季度净利润同比增长


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
# 60个月动量
factor_list.append(Factor('MOM_60M', get_momentum(1200), pd.to_datetime('2017-07-31'),
                          dependency=['ADJ_CLOSE']))
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
# beta因子


def get_beta(universe, start_time, end_time):
    '''
    计算股票的beta，以上证综指作为市场收益基准
    '''
    def moving_OLS(y, x, window):
        '''
        滚动快速计算beta值
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
        betas = np.empty(x.shape, dtype=float)
        betas[:] = np.NaN
        for i in range(N):
            if i < window or np.any(pd.isnull(x.iloc[i])):
                continue
            xx = cum_xx[i] - cum_xx[i - window]
            xy = cum_xy[i] - cum_xy[i - window]
            try:
                betas[i] = linalg.solve(xx, xy)
            except LinAlgError as e:    # 因为停牌等因素，股价一直都不变，此时的beta没有意义
                continue
        # pdb.set_trace()
        return pd.Series(betas[:, 1], index=x.index)

    days = 252
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    new_start = dateshandle.tds_shift(start_time, days)
    stock_data = query('ADJ_CLOSE', (new_start, end_time))
    benchmark_data = query('SSEC_CLOSE', (new_start, end_time))
    stock_data = stock_data.pct_change().dropna(how='all').dropna(how='all', axis=1)
    benchmark_data = benchmark_data.iloc[:, 0].pct_change().dropna()
    # pdb.set_trace()
    data = stock_data.apply(lambda x: moving_OLS(x, benchmark_data, days))
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask, sorted(universe)]
    if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
        checkdata_completeness(data, start_time, end_time)
    return data


factor_list.append(Factor('BETA', get_beta, pd.to_datetime('2017-09-04'),
                          dependency=['ADJ_CLOSE', 'SSEC_CLOSE'], desc='252交易日滚动beta系数'))


# 特质波动率因子
def get_specialvol(universe, start_time, end_time):
    '''
    特质波动率因子，为计算beta的副产物；同样以上证综指作为市场收益基准
    '''
    def moving_specvol(y, x, window):
        '''
        滚动快速计算特质波动率
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
        # 计算滚动特质波动率
        resids = np.empty(N, dtype=float)
        resids[:] = np.NaN
        for i in range(N):
            if i < window or np.any(pd.isnull(x.iloc[i])):
                continue
            xx = cum_xx[i] - cum_xx[i - window]
            xy = cum_xy[i] - cum_xy[i - window]
            try:
                beta = linalg.solve(xx, xy)
                tmp_x = x.values[i + 1 - window: i + 1]
                tmp_y = y.values[i + 1 - window: i + 1]
                resid = tmp_y - np.dot(tmp_x, beta)
                # pdb.set_trace()
                resids[i] = np.std(resid)
            except LinAlgError as e:    # 因为停牌等因素，股价一直都不变，此时的波动率没有意义
                continue
        # pdb.set_trace()
        return pd.Series(resids, index=x.index)

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
    data = stock_data.apply(lambda x: moving_specvol(x, benchmark_data, days))
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask, sorted(universe)]
    if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
        checkdata_completeness(data, start_time, end_time)
    return data


factor_list.append(Factor('SPECIAL_VOL', get_specialvol, pd.to_datetime('2017-09-05'),
                          dependency=['ADJ_CLOSE', 'SSEC_CLOSE'], desc='特质波动率'))

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
# 多期限趋势因子
# 来源：A trend factor: Any economic gains from using information over investment horizons


def gen_multitrend_factor(trend_offsets):
    '''
    母函数，用于生成多趋势因子的计算方法
    Parameter
    ---------
    trend_offsets: iterable
        因子计算过程中所使用的均线，例如[10, 20, 30]表示在综合趋势的过程中使用10日、20日和30日的均线
        目前可以使用的均线长度包含[3, 5, 10, 20, 30, 60, 90, 120, 180, 240, 270, 300]
    '''
    def inner(universe, start_time, end_time):
        start_time = pd.to_datetime(start_time)
        period_length = 6
        new_start = start_time - pd.Timedelta('%d days' % ((period_length + 2) * 31))
        if new_start < pd.to_datetime(START_TIME):
            new_start = START_TIME
        # 加载数据
        ma_datas = [query('MA%d' % offset, (new_start, end_time)) for offset in trend_offsets]
        ma_datas = convert_data(ma_datas, ['MA%d' % offset for offset in trend_offsets])
        adj_close = query('ADJ_CLOSE', (new_start, end_time))
        month_end = dateshandle.get_period_end(adj_close.index.tolist())
        monthly_ret = adj_close.loc[month_end].pct_change()
        reg_cache = {}
        data = {}
        for idx in range(period_length, len(month_end)):
            reg_coeff = 0
            reg_cnt = 0
            for offset in range(period_length):
                last_td = month_end[idx - offset - 1]
                cur_td = month_end[idx - offset]
                if cur_td in reg_cache:
                    reg_res = reg_cache[cur_td]
                else:
                    # 加载回归使用数据
                    tmp_madata = ma_datas.xs(last_td, level=0).T
                    tmp_ret = monthly_ret.loc[cur_td].reindex(tmp_madata.index)
                    ma_na_mask = np.any(pd.isnull(tmp_madata), axis=1)
                    ret_na_mask = pd.isnull(tmp_ret).values
                    na_mask = ma_na_mask | ret_na_mask
                    tmp_madata = tmp_madata.loc[~na_mask]
                    tmp_ret = tmp_ret.loc[~na_mask]
                    n, m = tmp_madata.shape
                    if n < m + 10:
                        warnings.warn('回归使用的数据量过少，直接跳过', RuntimeWarning)
                        continue
                    reg_mod = OLS(tmp_ret, add_constant(tmp_madata))
                    reg_res = reg_mod.fit().params.drop('const')
                    reg_cache[cur_td] = reg_res
                reg_coeff += reg_res
                reg_cnt += 1
            try:
                reg_coeff = reg_coeff / reg_cnt
            except ZeroDivisionError:
                continue
            td = month_end[idx]
            cur_ma = ma_datas.xs(td, level=0).T
            predict = cur_ma.dot(reg_coeff)
            data[td] = predict
        tds = dateshandle.get_tds(start_time, end_time)
        data = pd.DataFrame(data).T.reindex(tds, method='ffill')
        data = data.loc[:, sorted(universe)]
        assert check_indexorder(data), 'Error, data order is mixed!'
        checkdata_completeness(data, start_time, end_time)
        return data
    return inner


mas = [3, 5, 10, 20, 50, 100, 200]
factor_list.append(Factor('MULTI_TREND', gen_multitrend_factor(mas),
                          pd.to_datetime('2017-12-06'), dependency=['MA%d' % m for m in mas],
                          desc='多期限趋势因子'))

# --------------------------------------------------------------------------------------------------


check_duplicate_factorname(factor_list, __name__)


if __name__ == '__main__':
    pass

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

import datatoolkits
import dateshandle
import numpy as np
import pandas as pd
import pdb
from ..const import START_TIME
from .utils import (Factor, check_indexorder, check_duplicate_factorname, convert_data,
                    checkdata_completeness)
from .query import query

# --------------------------------------------------------------------------------------------------
# 常量和功能函数
NAME = 'derivativefactors'


def get_factor_dict():
    res = dict()
    for f in factor_list:
        res[f.name] = {'rel_path': NAME + '\\' + f.name, 'factor': f}
    return res


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


ep_ttm = Factor('EP_TTM', get_ep, pd.to_datetime('2017-07-27'),
                dependency=['NI_TTM', 'TOTAL_MKTVALUE'], desc='净利润/总市值计算得到')

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


bp = Factor('BP', get_bp, pd.to_datetime('2017-07-27'),
            dependency=['EQUITY', 'TOTAL_MKTVALUE'], desc='最新的归属母公司权益/总市值')

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


sp_ttm = Factor('SP_TTM', get_sp, pd.to_datetime('2017-07-27'),
                dependency=['OPREV_TTM', 'TOTAL_MKTVALUE'], desc='营业收入/总市值')

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


cfp_ttm = Factor('CFP_TTM', get_cfp, pd.to_datetime('2017-07-27'),
                 dependency=['OPNETCF_TTM', 'TOTAL_MKTVALUE'], desc='经营活动中现金流净额/总市值')

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


sale2ev = Factor('SALE2EV', get_sale2ev, pd.to_datetime('2017-07-27'),
                 dependency=['OPREV_TTM', 'TOTAL_MKTVALUE', 'TNCL', 'CASH'],
                 desc='营业收入/(总市值+非流动负债合计-货币资金)')
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


oprev_yoy = Factor('OPREV_YOY', get_oprev_yoy, pd.to_datetime('2017-07-27'),
                   dependency=['OPREV_1S', 'OPREV_5S'],
                   desc='(本季度营业收入-上年同季度营业收入)/abs(上年同季度营业收入)')
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


ni_yoy = Factor('NI_YOY', get_ni_yoy, pd.to_datetime('2017-07-27'),
                dependency=['NI_1S', 'NI_5S'],
                desc='(本季度净利润-上年同季度净利润)/abs(上年同季度净利润)')

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
ni_5yg = Factor('NI_5YG', get_p5ygrowth('NI'), pd.to_datetime('2017-07-28'),
                dependency=['NI_%dY' % i for i in range(1, 6)])
# 营业收入过去5年增长率
oprev_5yg = Factor('OPREV_5YG', get_p5ygrowth('OPREV'), pd.to_datetime('2017-07-28'),
                   dependency=['OPREV_%dY' % i for i in range(1, 6)])

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


roe = Factor('ROE', get_roe, pd.to_datetime('2017-07-28'),
             dependency=['NI_TTM', 'EQUITY'], desc='净利润TTM/归属母公司权益')
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


roa = Factor('ROA', get_roa, pd.to_datetime('2017-07-28'),
             dependency=['NI_TTM', 'TA'], desc='净利润TTM/总资产')

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


opprofit_margin = Factor('OPPROFIT_MARGIN', get_opprofitmargin, pd.to_datetime('2017-07-28'),
                         dependency=['OPREV_TTM', 'OPCOST_TTM', 'OPEXP_TTM', 'ADMINEXP_TTM',
                                     'FIEXP_TTM'],
                         desc='营业利润率 = (营业收入-营业成本-销售费用-管理费用-财务费用) / abs(营业收入)')

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


gross_margin = Factor('GROSS_MARGIN', get_grossmargin, pd.to_datetime('2017-07-31'),
                      dependency=['OPREV_TTM', 'OPCOST_TTM'],
                      desc='毛利率 = (营业收入 - 营业成本) / abs(营业收入)')

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


tato = Factor('TATO', get_tato, pd.to_datetime('2017-07-31'),
              dependency=['OPREV_TTM', 'TA'], desc='营业收入TTM / 最新总资产')

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


current_ratio = Factor('CURRENT_RATIO', get_currentratio, pd.to_datetime('2017-07-31'),
                       dependency=['TCA', 'TCL'], desc='流动比率 = 流动资产 / 流动负债')

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


opnetcf2opprofit = Factor('OPNETCF2OPPROFIT', get_nopcf2opprofit, pd.to_datetime('2017-07-31'),
                          dependency=['OPNETCF_TTM', 'OPPROFIT_TTM'],
                          desc='经营活动中产生的现金流净额TTM / 营业利润TTM')

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


threefee2sale = Factor('FEE2SALE', get_3fee2sale, pd.to_datetime('2017-07-31'),
                       dependency=['OPREV_TTM', 'OPEXP_TTM', 'ADMINEXP_TTM', 'FIEXP_TTM'],
                       desc='(销售费用TTM+管理费用TTM+财务费用TTM) / abs(营业收入)')
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
momentum_1m = Factor('MOM_1M', get_momentum(20), pd.to_datetime('2017-07-31'),
                     dependency=['ADJ_CLOSE'])
# 3月动量
momentum_3m = Factor('MOM_3M', get_momentum(60), pd.to_datetime('2017-07-31'),
                     dependency=['ADJ_CLOSE'])
# 60个月动量
momentum_60m = Factor('MOM_60M', get_momentum(1200), pd.to_datetime('2017-07-31'),
                      dependency=['ADJ_CLOSE'])
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
        计算的数据结果类型，只支持skew和kurt
    '''
    def _inner(universe, start_time, end_time):
        start_time = pd.to_datetime(start_time)
        shift_days = int(days / 20 * 31)
        new_start = start_time - pd.Timedelta('30 day') - pd.Timedelta('%d day' % shift_days)
        data = query('DAILY_RET', (new_start, end_time))
        rolling = data.rolling(days, min_periods=days)
        data = getattr(rolling, func_name)()
        data = data.dropna(how='all')
        mask = (data.index >= start_time) & (data.index <= end_time)
        data = data.loc[mask, sorted(universe)]
        if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
            checkdata_completeness(data, start_time, end_time)
        return data
    return _inner


skew_1m = Factor('SKEW_1M', gen_skfunc(20, 'skew'), pd.to_datetime('2017-08-02'),
                 dependency=['DAILY_RET'], desc='过去20个交易日收益率的skew')
kurtosis_1m = Factor('KURTOSIS_1M', gen_skfunc(20, 'kurt'), pd.to_datetime('2017-08-02'),
                     dependency=['DAILY_RET'], desc='过去20个交易日收益率的kurtosis')
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


conexp_dis = Factor('CONEXP_DIS', get_conexpprice, pd.to_datetime('2017-08-04'),
                    dependency=['TARGET_PRICE', 'CLOSE'],
                    desc="一致预期价格距离因子 = 一致预期目标价（在other因子模块中） / close - 1")
# --------------------------------------------------------------------------------------------------
# 前景理论因子


def get_prospectfactor(universe, start_time, end_time):
    '''
    前景理论因子
    '''
    period = 60  # 因子使用的行情数据的回溯期
    shift_days = int(period / 20 * 31)
    start_time = pd.to_datetime(start_time)
    new_start = start_time - pd.Timedelta('30 day') - pd.Timedelta('%d day' % shift_days)
    index_data = query('SSEC_CLOSE', (new_start, end_time))
    data = query('CLOSE', (new_start, end_time))
    # 计算对应超额收益率
    index_data = index_data.loc[:, data.columns]
    index_data = index_data.pct_change().dropna(how='all')
    data = data.pct_change().dropna(how='all')
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
            arr = list(enumerate(arr))
            res = [tki_raw(r_k, len(arr)) for r_k in arr]
            return sum(res)
        return ts_arr.rolling(60, min_periods=60).apply(cal_tk)
    data = data.apply(ts_ptvalue)
    data = data.dropna(how='all')
    mask = (data.index >= start_time) & (data.index <= end_time)
    data = data.loc[mask, sorted(universe)]
    if start_time > pd.to_datetime(START_TIME):     # 第一次更新从START_TIME开始，必然会有缺失数据
        checkdata_completeness(data, start_time, end_time)
    return data


ptvalue = Factor('PT_VALUE', get_prospectfactor, pd.to_datetime('2017-08-16'),
                 dependency=['CLOSE', 'SSEC_CLOSE'], desc='前景理论因子')
# --------------------------------------------------------------------------------------------------


factor_list = [ep_ttm, bp, sp_ttm, cfp_ttm, sale2ev, oprev_yoy, ni_yoy, ni_5yg, oprev_5yg,
               roe, roa, opprofit_margin, gross_margin, tato, current_ratio, threefee2sale,
               momentum_1m, momentum_3m, momentum_60m, conexp_dis, skew_1m, kurtosis_1m,
               ptvalue]
check_duplicate_factorname(factor_list, __name__)

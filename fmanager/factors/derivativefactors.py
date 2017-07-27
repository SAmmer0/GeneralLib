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
import pdb
import datatoolkits
import dateshandle
import numpy as np
import pandas as pd
from .utils import Factor, check_indexorder, check_duplicate_factorname
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
    return data


ni_yoy = Factor('NI_YOY', get_ni_yoy, pd.to_datetime('2017-07-27'),
                dependency=['NI_1S', 'NI_5S'],
                desc='(本季度净利润-上年同季度净利润)/abs(上年同季度净利润)')
# --------------------------------------------------------------------------------------------------

factor_list = [ep_ttm, bp, sp_ttm, cfp_ttm, sale2ev, oprev_yoy, ni_yoy]
check_duplicate_factorname(factor_list, __name__)

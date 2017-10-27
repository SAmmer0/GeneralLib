#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-20 09:57:18
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
用于定义基础的监控模块类（包含所有的监控相关信息）和其他辅助的函数
'''
# 系统库
import logging
# 第三方库
from pandas import to_datetime
import pandas as pd
# 本地库
from factortest.grouptest.utils import MkvWeightCalc, EqlWeightCalc
from factortest.utils import MonRebCalcu, WeekRebCalcu
from factortest.const import TOTALMKV_WEIGHTED, MONTHLY
from portmonitor.const import LONG, PORT_DATA_PATH
from fmanager import query
from fmanager.database.const import NaS

# --------------------------------------------------------------------------------------------------


class MonitorConfig(object):
    '''
    监控信息类，包含所有的相关监控该信息
    '''

    def __init__(self, stock_filter, add_time, port_id, weight_method=TOTALMKV_WEIGHTED,
                 rebalance_type=MONTHLY, init_cap=1e10, port_type=LONG):
        '''
        Parameter
        ---------
        stock_filter: function
            筛选股票的函数，要求形式为stock_filter(datetime like: date)->list
        add_time: datetime like
            添加的时间
        port_id: str
            组合的id，不能与其他组合重复，是组合的唯一标识
        weight_method: str, default TOTALMKV_WEIGHTED
            加权的方法，目前支持factortest.const中的[EQUAL_WEIGHTED, TOTALMKV_WEIGHTED, FLOATMKV_WEIGHTED]
            对应的为等权、总市值加权、流通市值加权
        rebalance_type: str, default MONTHLY
            换仓日计算方法，目前支持factortet.const中的[MONTHLY, WEEKLY]
            注：换仓的时间是在该日期的下一个交易日
        init_cap: int, default 1e10
            初始的资本金
        port_type: str, default LONG
            组合的类型，目前支持LONG（做多）, SHORT（做空）
        '''
        self.stock_filter = stock_filter
        self.weight_method = weight_method
        self.rebalance_type = rebalance_type
        self.init_cap = init_cap
        self.add_time = to_datetime(add_time)
        self.port_id = port_id
        self.port_type = port_type


# --------------------------------------------------------------------------------------------------
# 日志设置
def set_logger():
    logger = logging.getLogger(__name__.split('.')[0])
    if not logger.handlers:     # 如果有处理函数了，表示当前的Logger已经被设置过
        logger.setLevel(logging.INFO)
        file_handle = logging.FileHandler(PORT_DATA_PATH + '\\update_log.log')
        file_handle.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
# --------------------------------------------------------------------------------------------------
# 函数


def factor_stockfilter_template(factor_name, group_id, group_num=5, stock_pool=None,
                                industry_cls=None):
    '''
    按照因子筛选股票的模板

    Parameter
    ---------
    factor_name: str
        因子名称，必须能够在fmanager.list_allfactor()中找到
    group_id: int
        将股票按照因子排序后的分组序号，要求的范围是[0, group_num)
    group_num: int, default 5
        将股票按照因子分组的组数
    stock_pool: str, default None
        股票池的名称，要求能够在fmanager.list_allfactor()中找到，默认为None表示没有股票池限制
    industry_cls: str, default None
        行业分类标准，要求能够在fmanager.list_allfactor()中找到，默认为None表示不进行行业中性化

    Return
    ------
    out: function(date)-> [code1, code2, ...]

    Notes
    -----
    筛选股票的过程中，将自动剔除ST股票和不能交易的股票
    行业中性化的方法是在每个行业分组内部将股票按照因子值排序分为几组，然后将不同行业中的各个组分别集
    合在一起
    '''
    def stock_filter(date):
        st_data = query('ST_TAG', date).iloc[0]
        trade_data = query('TRADEABLE', date).iloc[0]
        factor_data = query(factor_name, date).iloc[0]
        data = pd.DataFrame({'st_data': st_data, 'trade_data': trade_data, 'factor': factor_data})
        if stock_pool is not None:
            data = data.assign(stock_pool=query(stock_pool, date).iloc[0])
        else:
            data = data.assign(stock_pool=[1] * len(data))
        if industry_cls is not None:
            data = data.assign(industry=query(industry_cls, date).iloc[0])
            data = data.loc[data.industry != NaS]
        else:
            data = data.assign(industry=[NaS] * len(data))
        data = data.loc[(data.trade_data == 1) & (data.st_data == 0) & (data.stock_pool == 1), :].\
            dropna(subset=['factor'], axis=0)
        by_ind = data.groupby('industry')
        data = data.assign(datag=by_ind.factor.transform(lambda x: pd.qcut(x, group_num,
                                                                           labels=range(group_num))))
        by_group_id = data.groupby('datag')
        out = by_group_id.get_group(group_id).index.tolist()
        return out
    return stock_filter

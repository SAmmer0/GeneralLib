#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-30 13:54:07
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
回测用的相关工具
__version__ = 1.0
修改日期：2017-03-30
修改内容：
    初始化

__version__ = 1.1
修改日期：2017-04-06
修改内容：
    添加过滤持仓的函数
'''
__version__ = 1.1

from collections import OrderedDict
import dateshandle
import pandas as pd
from tqdm import tqdm


def get_constituent(by_date_constituent, date):
    '''
    将股票池按照时间进行分组，然后从分组中取出任何一个时间点的对应的成分
    @param:
        by_date_constituent：股票池按照日期进行分组后的groupby对象
        date: 需要获取成分的日期
    @return:
        返回给定日期的股票池列表
    '''
    valid_tds = list(by_date_constituent.groups.keys())
    valid_tds = pd.Series(valid_tds)
    newest_td = valid_tds[valid_tds < date].max()  # 假设成分相关的信息都是在日中公布的
    code_res = by_date_constituent.get_group(newest_td)['code'].tolist()
    return code_res


def get_daily_holding(signal_data, quotes_data, stock_pool, industry_cls, stock_filter,
                      rebalance_dates):
    '''
    用于根据一定的条件，从股票池中选出满足一定条件的股票，然后将其映射到这个期间的交易日中，
    最终得到每个交易日的持仓
    @param:
        signal_data: 信号数据DataFrame，必须包含time、code列
        quotes_data: 行情数据DataFramem，必须包含time、code列
        stock_pool: 时点股票池DataFrame，测试中股票池为离当前交易日最近的一个时点的股票池，
            必须包含time、code列
        industry_cls: 行业分类DataFrame，必须包含time、code列
        stock_filter: 用于选择股票的函数，形式为stock_filter(cur_signal_data, cur_ind_cls)，要求返回
            的股票为[[code1, code2, ...], [codex1, codex2, ...], ...]
        rebalance_dates: 再平衡日，即在该日期计算下个时期的股票池，然后在下个最近的交易日换仓
    @return:
        每个交易日的持仓，格式为OrderedDict，字典值与stock_filter返回值类似
    注：
        对于每个再平衡日，计算指标，筛选股票，然后在下个交易日换仓，随后的交易日的持仓都与该
        新的持仓相同，直至到下个新的在平衡日
    '''
    # 获取交易日
    start_time, end_time = rebalance_dates[0], rebalance_dates[-1]
    tds = dateshandle.wind_time_standardlization(dateshandle.get_tds(start_time, end_time))
    # 初始化
    holdings = OrderedDict()
    stockpool_bydate = stock_pool.groupby('time')
    for td, tq_idx in zip(rebalance_dates[:-1], tqdm(range(len(rebalance_dates) - 1))):
        # 获取当前的股票池
        constituent = get_constituent(stockpool_bydate, td)
        # 获取当前的行业分类
        ind_cls = industry_cls.loc[industry_cls['time'] == td]
        # 过滤下个交易日不能交易的股票
        next_td = tds[tds.index(td) + 1]
        tradeable_codes = quotes_data.loc[(quotes_data.time == next_td) & (quotes_data.STTag == 0) &
                                          quotes_data.tradable, 'code'].tolist()
        tradeable_codes = set(tradeable_codes).intersection(constituent)
        # 获取当前的信号数据
        cur_signal_data = signal_data.loc[signal_data['time'] == td]
        # 选出当前符合标准的股票
        valid_stocks = stock_filter(cur_signal_data, ind_cls)
        valid_stocks = [[c for c in group if c in tradeable_codes]
                        for group in valid_stocks]
        holdings[td] = valid_stocks

    td_holdings = OrderedDict()
    cur_holding = holdings.popitem(last=False)
    try:
        next_holding = holdings.popitem(last=False)
    except KeyError:
        next_holding = None
    for td in tds[1:]:  # 当前计算信号，下一个交易日换仓
        # 一个月的交易日中引用的是相同的持仓，如果被修改会导致该月所有持仓的修改，存在隐患
        # 但是目前考虑没有什么需要修改的地方，暂时不使用深复制
        td_holdings[td] = cur_holding[1]
        if next_holding is not None and td == next_holding[0]:
            cur_holding = next_holding
            try:
                next_holding = holdings.popitem(last=False)
            except KeyError:
                next_holding = None
    return td_holdings


def holding_filter(holding, threshold=1):
    '''
    过滤掉没有持仓的交易日
    @param:
        holding: 持仓，要求为OrderDict的类型，键为交易日，值为持仓列表[[code1, code2, ...], ...]
        threshold: 持仓股票数要求，即至少需要大于或者等于该数才被认为是有效的持仓，默认为1
    @return:
        过滤后的持仓，同样为OrderDict类型
    '''
    res = OrderedDict()
    for k, v in holding:
        if len(v) < threshold:
            continue
        res[k] = v
    return res


# 暂时添加计算日频收益率的程序，直接使用long_short_factortest中的函数，日后优化后添加
def cal_nav():
    raise NotImplementedError

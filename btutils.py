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
# --------------------------------------------------------------------------------------------------
# import
from collections import namedtuple, OrderedDict
# import datatoolkits
import dateshandle
# from enum import Enum
import pandas as pd
from tqdm import tqdm
# --------------------------------------------------------------------------------------------------
# 常量和类的定义
Position = namedtuple('Position', 'pos')


class PositionGroup(object):
    '''
    该类是持仓的集合，即分组测试中包含所有分组的一个集合（但是内部这个集合是以元组的形式存储，持仓
    的股票序列也是以元组的形式存储，避免被修改）
    例如：该集合中包含了从10%到100%分位数的10个组合
    '''

    def __init__(self, groups):
        '''
        要求groups为列表的列表，即形式为[[code1, code2, ...], ...]
        即选股函数返回的结果
        '''
        self.groups = self._groups_transformer(groups)

    def _groups_transformer(self, groups):
        groups_res = list()
        for pos in groups:
            tmp_pos = Position(pos=tuple(pos))
            groups_res.append(tmp_pos)
        return tuple(groups_res)

    def get_pos(self, idx):
        return self.groups[idx]

    def __len__(self):
        return len(self.groups)


class Portfolio(object):
    '''
    用于记录组合的持仓状况
    具体的数据包含了：组合代码、对应的数量、现金、换仓日期
    '''

    def __init__(self, pos, cash, change_date):
        '''
        @param:
            pos: df的形式，包含有code和num列
            cash: 现金，数值形式
            change_date: 换仓日，datetime形式
        '''
        self.pos = pos
        self.cash = cash
        self.change_date = change_date

    def mkt_value(self, quote, price_type='close'):
        '''
        计算当前市值的函数
        @param:
            quote: 给定交易日的行情数据，应当包含有price_type列
            price_type: 计算市值所使用的数据，默认为close，即使用收盘价计算
        @return:
            给定行情下的市值
        '''
        tmp_quote = pd.merge(self.pos, quote, on='code', how='left')
        stock_value = (tmp_quote.num * tmp_quote[price_type]).sum()
        total = stock_value + self.cash
        return total
# --------------------------------------------------------------------------------------------------
# 函数


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
        换仓日的持仓，格式为OrderedDict，字典值为PositionGroup类型
    注：
        对于每个再平衡日，计算指标，筛选股票，然后在下个交易日换仓，随后的交易日的持仓都与该
        新的持仓相同，直至到下个新的在平衡日
    '''
    # 获取交易日
    start_time, end_time = rebalance_dates[0], rebalance_dates[-1]
    tds = dateshandle.wind_time_standardlization(dateshandle.get_tds(start_time, end_time))

    # 计算对应的换仓日
    # 计算再平衡日对应的下标，最后一个交易日没有换仓日
    reb_index = [tds.index(t) for t in rebalance_dates[:-1]]
    # 计算换仓日对应的日期
    chg_dates = [tds[i + 1] for i in reb_index]
    key_dates = list(zip(rebalance_dates[:-1], chg_dates))

    # 初始化
    holdings = OrderedDict()
    stockpool_bydate = stock_pool.groupby('time')

    # 计算换仓日的股票组合
    for (reb_dt, chg_dt), tqi in zip(key_dates, tqdm(key_dates)):
        # 获取换仓日股票池
        constituent = get_constituent(stockpool_bydate, chg_dt)
        # 获取再平衡日的行业分类
        ind_cls = industry_cls.loc[industry_cls.time == reb_dt]
        # 过滤不能交易的股票
        tradeable_stocks = quotes_data.loc[(quotes_data.time == chg_dt) & (quotes_data.STTag == 0) &
                                           quotes_data.tradeable, 'code'].tolist()
        tradeable_stocks = set(tradeable_stocks).intersection(constituent)

        # 获取当前信号数据
        reb_sig_data = signal_data.loc[signal_data['time'] == reb_dt]

        # 根据信号函数计算当前的股票组
        valid_stocks = stock_filter(reb_sig_data, ind_cls)
        valid_stocks = [[c for c in group if c in tradeable_stocks]
                        for group in valid_stocks]
        holdings[chg_dt] = PositionGroup(valid_stocks)
    return holdings


def holding_filter(holding, threshold=1):
    '''
    过滤掉没有持仓的交易日
    @param:
        holding: 持仓，要求为OrderDict的类型，键为交易日，值为持仓列表[[code1, code2, ...], ...]
        threshold: 持仓股票数要求，即至少需要持仓中的每个组合的持有股票数大于或者等于该数才被认为
            是有效的持仓，默认为1
    @return:
        过滤后的持仓，同样为OrderDict类型
    注：目前来说该函数没有作用，因为如果将中间的交易日剔除会导致计算出的净值有很大的问题，应该废弃
    '''
    res = OrderedDict()
    for t, groups in holding.items():
        for g in groups:
            if len(g) < threshold:
                break
        else:
            res[t] = groups
    return res


# 暂时添加计算日频收益率的程序，直接使用long_short_factortest中的函数，日后优化后添加
def cal_nav():
    raise NotImplementedError

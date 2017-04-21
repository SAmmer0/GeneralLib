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

__version__ = 1.2
修改日期：2017-04-14
修改内容：
    1. 删除过滤持仓函数
    2. 重构计算持仓的函数
    3. 实现计算净值的函数，并添加其他辅助函数

__version__ = 1.3
修改日期：2017-04-17
修改内容：
    为PositionGroup添加__str__和__repr__方法
'''
__version__ = 1.3
# --------------------------------------------------------------------------------------------------
# import
from collections import namedtuple
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
        # self.groups中的数据存储为(pos1, pos2, ...)
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

    def __iter__(self):
        return iter(self.groups)

    def __str__(self):
        return str(self.groups)

    def __repr__(self):
        return repr(self.groups)


class Portfolio(object):
    '''
    用于记录组合的持仓状况
    具体的数据包含了：组合代码、对应的数量、现金、换仓日期
    '''

    def __init__(self, pos, cash):
        '''
        @param:
            pos: df的形式，包含有code和num列，可以为空
            cash: 现金，数值形式
        '''
        self.pos = pos
        self.cash = cash

    def mkt_value(self, quote, date, price_type='close'):
        '''
        计算当前市值的函数，若当前为空仓，则直接返回现金
        @param:
            quote: 给定交易日的行情数据，应当包含有price_type列
            date: 交易日的日期
            price_type: 计算市值所使用的数据，默认为close，即使用收盘价计算
        @return:
            给定行情下的市值
        '''
        if len(self.pos) == 0:
            return self.cash
        quote_cs = quote.loc[quote.time == date]    # 获取横截面行情数据
        if not set(self.pos.code).issubset(quote_cs.code):  # 出现退市的情况
            delist_codes = set(self.pos.code).difference(quote_cs.code)
            self.delist_value(delist_codes, quote)
        tmp_quote = pd.merge(self.pos, quote_cs, on='code', how='left')
        stock_value = (tmp_quote.num * tmp_quote[price_type]).sum()
        total = stock_value + self.cash
        return total

    def delist_value(self, delist_codes, quotes, price_type='close'):
        '''
        处理出现退市相关的情况时净值的计算
        目前的方案如下：
            当出现退市时，即行情和pos在inner merge以后的结果长度小于pos的长度，对于多余出来的股票
            按照其最后一个可交易的交易日的收盘价计算其当天卖出的价格，然后直接将其转为现金持有，并
            在pos中将其剔除
        @param:
            delist_codes: 需要进行退市处理的股票，要求为可迭代类型
            quotes: 行情数据，需要按照时间进行了升序排列
            price_type: 计算退市价格的类型，默认为close
        '''
        for code in delist_codes:
            delist_price = quotes.loc[(quotes.code == code) & quotes.tradeable, price_type].iloc[-1]
            delist_value = delist_price * self.pos.loc[self.pos.code == code, 'num'].iloc[0]
            self.cash += delist_value
        self.pos = self.pos.loc[~self.pos.code.isin(delist_codes)].reset_index(drop=True)


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
        换仓日的持仓，格式为字典类型，字典值为PositionGroup类型，因此需要注意返回的持仓并没有时间
        顺序，需要先对键进行排序
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
    holdings = dict()
    stockpool_bydate = stock_pool.groupby('time')

    # 计算换仓日的股票组合
    for (reb_dt, chg_dt), tqi in zip(key_dates, tqdm(key_dates)):
        # 获取换仓日股票池
        constituent = get_constituent(stockpool_bydate, chg_dt)
        # 获取再平衡日的行业分类
        ind_cls = industry_cls.loc[industry_cls.time == reb_dt]
        # 过滤不能交易的股票，此处会自动将建仓日不存在数据的股票过滤
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


def cal_nav(holdings, end_date, quotes, ini_capital=1e9, normalize=True, **kwargs):
    '''
    根据持仓状况计算策略的净值（以每个交易日的收盘价计算）
    @param:
        holdings: 由get_daily_holding返回的每个换仓日的持仓，或者为OrderDict类型，键为换仓日日期，
            值为对应的持仓（为PositionGroup类型）
        end_date: 最后一个换仓记录的结束时间，一般情况下，应该设置为最后一个在平衡日，可以为
            pd.to_datetime可以解析的任何类型
        quotes: 行情数据
        ini_capital: 初始资本，默认为10亿人民币，避免数额过小导致在整数约束下很多股票的数量为0
        normalize: 是否将组合的价值由金额转换为净值（转换方法为组合市场价值除以初始资本），
            默认为True，即需要转换
        kwargs: 一些其他的参数，用于传入build_pos函数
    @return:
        df类型，索引为时间，列名为group_i，其中i为分组次序
    '''
    # 对交易日与换仓日之间的关系进行映射
    start_date = min(holdings.keys())   # 开始日一定为换仓日，且换仓日一定为交易日
    tds = dateshandle.wind_time_standardlization(dateshandle.get_tds(start_date, end_date))
    tds_df = pd.DataFrame({'chg_date': list(holdings.keys())})
    tds_df['tds'] = tds_df['chg_date']
    tds_df = tds_df.set_index('tds').sort_index()
    tds_df = tds_df.reindex(tds, method='ffill')
    tds_map = dict(zip(tds_df.index, tds_df.chg_date))

    # 初始化
    portfolio_record = None     # 组合记录
    nav = list()    # 净值结果
    cols = ['time'] + ['group_%d' % i for i in range(1, len(holdings[start_date]) + 1)]    # 结果列名
    # 交易日循环
    for td, tq_idx in zip(sorted(tds_map), tqdm(tds_map)):
        # 当前为换仓日，第一个建仓日一定为换仓日
        if tds_map[td] == td:
            cur_pos = holdings[td]
            if portfolio_record is None:    # 第一次建仓
                portfolio_record = [Portfolio(pd.DataFrame(), ini_capital)]
            tmp_portrecord = list()
            for port_idx, pos in enumerate(cur_pos):
                # 此处建仓实际上假设之前的股票都在今天开盘卖出，然后再按照开盘价买入新的仓位
                # 虽然这种假设不合理，但是如果使用上个交易日的数据会带来一些复杂性，且一个交易日的
                # 收盘价和随后一天的开盘价的差值一般不会太大，故忽略这方面的影响
                tmp_port = build_pos(pos, portfolio_record[port_idx].mkt_value(quotes, td, 'open'),
                                     quotes, td)
                tmp_portrecord.append(tmp_port)
            portfolio_record = tmp_portrecord   # 更新portfolio_record
        # 计算每个组合的收盘市场价值
        tmp_mktvalue = list()
        for port in portfolio_record:
            mkt_value = port.mkt_value(quotes, td)
            tmp_mktvalue.append(mkt_value)
        tmp_mktvalue.insert(0, td)
        nav.append(dict(zip(cols, tmp_mktvalue)))
    nav = pd.DataFrame(nav)
    nav = nav.set_index('time')
    if normalize:
        for i in range(1, len(holdings[start_date]) + 1):
            nav['group_%d' % i] = nav['group_%d' % i] / ini_capital
    return nav


def build_pos(pos, cash, quotes, date, price_col='open', buildpos_type='money-weighted'):
    '''
    建仓函数
    @param:
        pos: 建仓的股票组合，为Position类型，要求在建仓日股票没有退市
        cash: 建仓可用的资金
        quotes: 股票行情数据
        date: 建仓日期（加入建仓日期参数是为了避免在程序中需要多维护一个临时参数，现在直接将所有行情
            数据在程序间传送即可，因为Python的参数传递机制是引用传递的，即使数据量比较大的行情数据也
            不会导致程序速度减慢）
        price_col: 建仓使用的价格，默认为开盘价
        buildpos_type: 建仓方式，默认为money_weighted，即资金等权建仓，每只股票分配等量的资金，目前
            不支持其他建仓方式，后续会慢慢添加
    @reutrn:
        建仓结果为Portfolio类型的对象
    内部实现为：
    按照股票数量平均分配资金，然后根据资金和价格以及买卖乘数（100）计算股票数量，多余的资金放入
    现金中
    如果没有持仓，则Portfolio中pos属性为空的df，所有价值按照现金计算
    '''
    # 目前没有实现其他权重计算方式
    if buildpos_type != 'money-weighted':
        raise NotImplementedError

    # 计算资金分配
    try:
        cash_alloc = cash / len(pos.pos)
    except ZeroDivisionError:
        return Portfolio(pd.DataFrame(), cash)

    # 计算每只股票的购买数量
    multiple = 100
    # 此处行情数据使用方法会自动剔除建仓日退市的股票，但是如果存在退市的公司，则会导致每只
    # 股票分配的资金相对比较少，问题不需要考虑，因为当天没有数据的股票会被过滤掉
    data = quotes.loc[quotes.code.isin(pos.pos) & (quotes.time == date), ['code', price_col]]
    data['num'] = cash_alloc / data[price_col]
    data['num'] = data['num'].apply(lambda x: int(x / multiple) * multiple)
    residual_cash = cash - (data['num'] * data[price_col]).sum()
    res = Portfolio(data.loc[:, ['code', 'num']], residual_cash)
    return res

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

__version__ = 1.31
修改日期：2017-04-26
修改内容：
    修改了get_daily_holdings的股票筛选逻辑，将指数成份过滤放在根据指标筛选股票前

__version__ = 1.4
修改日期：2017-05-04
修改内容：
    添加计算IC的函数

__version__ = 1.4.1
修改日期：2017-05-05
修改内容：
    实现holding2df

__version__ = 1.5
修改日期：2017-05-10
修改内容：
    添加流通市值加权的权重计算方法

__version__ = 1.6.0
修改日期：2017-05-23
修改内容：
    添加因子回测框架

__version__ = 1.6.1
修改日期：2017-05-23
修改内容：
    1. 对get_daily_holding做了小幅修改，去除不必要的行业分类的数据加载
    2. 修改了获取当前行业分类的方式

__version__ = 1.6.2
修改日期：2017-05-25
修改内容：
    添加计算换手率的函数

__version__ = 1.6.3
修改日期：2017-05-26
修改内容：
    完成换手率功能测试，修改部分bug

__version__ = 1.6.4
修改日期：2017-06-02
修改内容：
    在回测框架的analysis中添加报告简要回测数据的功能

__version__ = 1.6.5
修改日期：2017-06-06
修改内容：
    修改了BackTest类中默认的获取换仓日的方法，使用数据中有效的最小时间区间作为使用的时间区间，同时在初始化
    函数中现将开始和结束时间格式化为表示日期格式，避免后续使用出现的错误
'''
__version__ = '1.6.2'
# --------------------------------------------------------------------------------------------------
# import
from collections import namedtuple
import datatoolkits
import dateshandle
# from enum import Enum
import numpy as np
import pandas as pd
import report
import scipy.stats as spstats
from tqdm import tqdm
# --------------------------------------------------------------------------------------------------
# 常量和类的定义
# 持仓组合类型
Position = namedtuple('Position', 'pos')

# 持仓组合的集合类型


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


# 明细组合类型，包含持仓代码和持仓的数量，并能够对组合进行操作和计算
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
            # 这里面假设行情价格是按照时间升序排列的
            delist_price = quotes.loc[(quotes.code == code) & quotes.tradeable, price_type].iloc[-1]
            delist_value = delist_price * self.pos.loc[self.pos.code == code, 'num'].iloc[0]
            self.cash += delist_value
        self.pos = self.pos.loc[~self.pos.code.isin(delist_codes)].reset_index(drop=True)

# WeightCalculator的私有的权重计算函数


def _money_weighted(pos, quote):
    '''
    Parameter
    ---------
    pos: Position type
        证券持仓
    quote: DataFrame type
        单个交易日的行情数据，必须包含pos.pos中的股票
    '''
    weight = 1 / len(pos.pos)
    return dict(zip(pos.pos, [weight] * len(pos.pos)))


def _mktvalue_weighted(pos, quote):
    '''
    Parameter
    ---------
    pos: Position type
        证券持仓
    quote: DataFrame type
        单个交易日行情数据，需包含市值列，且必须包含pos.pos中的股票
    '''
    assert len(quote.time.unique()) == 1, 'Error, quote data should contain only on trading day'
    assert 'mktvalue' in quote.columns, 'Error, quote data should contain "mktvalue" column'
    tmp_quote = quote.loc[quote.code.isin(pos.pos), ['code', 'mktvalue']]
    tmp_quote = tmp_quote.set_index('code')
    tmp_quote['weight'] = tmp_quote['mktvalue'] / tmp_quote['mktvalue'].sum()
    assert datatoolkits.isclose(tmp_quote['weight'].sum(),
                                1), 'Error, sum of weights does not equal to 1'
    return tmp_quote['weight'].to_dict()


class WeightCalculator(object):
    '''
    Description
    -----------
    计算组合的权重，目前支持的权重计算方法有money-weighted（即等权), mktvalue-weighted（即市值
    加权）
    Example
    -------
    money-ewighted: Position(pos=(code1, code2, code3)) -> {code1: 1/3, code2: 1/3, code3: 1/3}
    mktvalue-weighted: Position(pos=(code1, code2)), code1_mktvalue=1 code2_mktvalue=2 ->
        {code1: 1/3, code2: 2/3}
    '''
    weight_method = {'money-weighted': _money_weighted,
                     'mktvalue-weighted': _mktvalue_weighted}

    def _checktype(self):
        if self.weighted_type not in self.weight_method:
            raise ValueError('Weight method error, valid types are {valid}, you provide {gtype}'.
                             format(valid=WeightCalculator.weight_method.keys(),
                                    gtype=WeightCalculator.weighted_type))

    def __init__(self, weighted_type):
        '''
        Parameter
        ---------
        weighted_type: str
            权重计算方法的名称
        '''
        self.weighted_type = weighted_type
        self._checktype()

    def cal_weight(self, pos, quote):
        weight_func = WeightCalculator.weight_method[self.weighted_type]
        return weight_func(pos, quote)


# 因子测试框架类
def _benchmark_filter(quote, ind_cls):
    '''
    样例函数，用于返回基准的所有股票代码
    Parameter
    ---------
    quote: pd.DataFrame
        相当于stock_filter的sig_data参数，因为返回所有股票，需要有code列
    ind_cls: pd.DataFrame
        空闲参数，保证API的一致

    Return
    ------
    out: list of list
        列表类型，中间只包含一个元素，且为列表类型，内部列表中为股票代码
    '''
    codes = quote.code.tolist()
    out = [codes]
    return out


class Backtest(object):
    '''
    Notes
    -----
    回测接口框架，主要包含以下几步：
    1. get_rawdata: 获取原始数据（必须用户实现）
    2. get_observable_data: 获取可观察的数据（可选）
    3. processing_data: 计算相关指标（必须用户实现）
    4. get_rebdates: 计算在平衡日（可选）
    5. processing_backtest: 进行回测（可选）
    6. analysis: 进行回测结果分析（可选）
    '''

    def __init__(self, quote_loader, constituent_loader, ind_loader, stock_filter, start_time,
                 end_time, freq='M', benchmark_filter=_benchmark_filter,
                 weight_method='money-weighted'):
        '''
        Parameter
        ---------
        quote_loader: datatoolkits.DataLoader
            行情数据加载器
        constituent_loader: datatoolkits.DataLoader
            指数成份数据加载器
        ind_loader: datatoolkits.DataLoader
            行业成份数据加载器
        stock_filter: function
            需要传入get_daily_holding用于获取每次筛选出股票的
        start_time: str or datetime
            回测开始时间
        end_time: str or datetime
            回测终止时间
        freq: str, default "M"
            回测频率，目前只支持月频和周频，且均在每个周期的最后一个交易日收盘后计算信号，在下个周期
            开始的时间买入，目前可选的类型有'M'表示月度换仓，'W'表示周度换仓
        benchmark_filter: func, default None
            返回基准指数的成份的函数，用于传入回测程序中获取基准指数的净值情况，如果为None则不计算，
            在计算基准净值时，参考使用的成份为constituent_loader中的数据
        '''
        self.quote_loader = quote_loader
        self.constituent_loader = constituent_loader
        self.ind_loader = ind_loader
        self.stock_filter = stock_filter
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        self.freq = freq
        self.benchmark_filter = benchmark_filter
        self.weight_method = weight_method

    def get_rawdata(self):
        '''
        用户需要自行实现获取数据的函数，该函数没有任何参数，且返回pd.DataFrame格式的数据
        Return
        ------
        out: pd.DataFrame
            整理好的原始数据
        '''
        raise NotImplementedError

    def get_observable_data(self, raw_data):
        '''
        用户可选实现的函数，目的是为了计算出每个日期能够观察到的数据，如果用户未给出实现，则直接返回
        raw_data

        Parameter
        ---------
        raw_data: pd.DataFrame
            待处理的原始数据

        Return
        ------
        out: pd.DataFrame
            处理好后的观察日数据
        '''
        return raw_data

    def processing_data(self, obs_data):
        '''
        用户必须实现的函数，用于计算在回测过程中需要使用的所有指标
        Parameter
        ---------
        obs_data: pd.DataFrame
            处理后的观察日数据

        Return
        ------
        out: pd.DataFrame
            处理后的对应每个交易日的数据（或者对应再平衡日的数据）
        '''
        raise NotImplementedError

    def get_rebdates(self):
        '''
        用于获取换仓日的数据，用户可选实现，默认为换仓日为每个月的第一个交易日；用户自行实现需要
        返回一个日期列表，日期必须为标准化后的时间，且列表需要按照升序排列
        Return
        ------
        out: list like
            按照升序排列的换仓日
        '''
        # 获取数据中的最小时间区间
        # 加载数据
        quote = self.quote_loader.load_data()
        constituent = self.constituent_loader.load_data()
        # 行情时间区间
        quote_start = quote.time.min()
        quote_end = quote.time.max()
        # 成份股时间区间
        constituent_start = constituent.time.min()
        constituent_end = constituent.time.max()
        # 计算最小时间区间
        start_time = max([quote_start, constituent_start, self.start_time])
        end_time = min([quote_end, constituent_end, self.end_time])
        # 计算再平衡日
        out = dateshandle.get_rebtd(start_time, end_time, freq=self.freq)

        return out

    def processing_backtest(self, sig_data, reb_dates):
        '''
        实行回测，用户可选实现，默认情况下，直接加载信号数据行情数据、行业成份数据和指数成份数据,
        如果用户自行实现，需要将持仓设置为self的holding属性，并将净值结果存储为nav属性；回测实例
        具有_cache属性，可用于存储后续需要使用的大量数据，在用户自行实现该函数过程中可以考虑使用
        该缓存来加速
        Parameter
        ---------
        sig_data: pd.DataFrame
            信号数据
        reb_dates: list like
        '''
        quote = self.quote_loader.load_data()
        ind_cls = self.ind_loader.load_data()
        index_constituent = self.constituent_loader.load_data()
        # 因为在DataLoader中已经加入缓存的功能，故在这个地方删去
        # # 将可能需要使用的数据加入缓存
        # self._cache['quote'] = quote
        # self._cache['ind_cls'] = ind_cls
        # self._cache['index_constituent'] = index_constituent
        # self._cache['reb_dates'] = reb_dates
        # 此处动态添加换仓日的缓存
        self.reb_dates = reb_dates
        # 进行回测
        stock_filter = self.stock_filter
        weight_method = self.weight_method
        # 将因子的计算数据缓存供后续使用
        self.factor_data = sig_data
        holding = get_daily_holding(sig_data, quote, index_constituent, ind_cls, stock_filter,
                                    reb_dates)
        self.nav = cal_nav(holding, reb_dates[-1], quote, buildpos_type=weight_method)
        self.holding = holding

    def analysis(self):
        '''
        对回测结果进行分析，用户可选实现，不返回结果，对于所有分析结果设置为回测实例的属性，默认包含
        的回测分析有：在nav中添加benchmark列，计算每列数据的年度收益（yearly_ret）和月度收益（
        monthly_ret），计算每组相对于基准的月度超额收益（monthly_excess），计算月度超额收益的t值和对
        应的p值（monthly_ttest），年度收益报表（yearlyret_tab），超额收益t检验报表（ttest_tab）
        Notes
        -----
        在本函数中默认情况下使用了回测过程中存储的缓存，如果回测函数进行了改变，该函数可能需要进行
        对应的改变
        '''
        # 从缓存中加载数据
        quote = self.quote_loader.load_data()
        ind_cls = self.ind_loader.load_data()
        index_constituent = self.constituent_loader.load_data()
        reb_dates = self.reb_dates
        # 计算基准的净值
        index_holding = get_daily_holding(quote, quote, index_constituent, ind_cls,
                                          self.benchmark_filter, reb_dates)
        index_nav = cal_nav(index_holding, reb_dates[-1], quote)
        index_nav = index_nav.rename(columns={'group_01': 'benchmark'})
        navs = pd.merge(self.nav, index_nav, left_index=True, right_index=True)
        assert len(navs) == len(self.nav) and len(navs) == len(index_nav),\
            '''Error, merge procedure produces missing data:
               len(navs) = {navs},
               len(self.nav) = {nav},
               len(index_nav) = {inavl}'''.format(navs=len(navs),
                                                  nav=len(self.nav),
                                                  inavl=len(index_nav))
        self.nav = navs
        # 计算一般评估指标，默认无风险利率为4%
        group_nav = self.nav.loc[:, self.nav.columns.str.startswith('group_')]
        brief_rpt = group_nav.apply(lambda x: report.brief_report(x, self.nav.benchmark, 0.04, 250))
        self.brief_rpt = brief_rpt.T
        # 计算月度数据和年度数据
        self.yearly_ret = self.nav.groupby(lambda x: x.year).\
            apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
        self.monthly_ret = self.nav.groupby(lambda x: x.strftime('%Y-%m')).\
            apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
        # 计算月度超额收益
        is_group_columns = self.monthly_ret.columns.str.startswith('group_')
        group_columns = sorted(self.monthly_ret.columns[is_group_columns])
        self.monthly_excess = self.monthly_ret.loc[:, group_columns].\
            apply(lambda x: x - self.monthly_ret['benchmark'])
        # 对月度超额收益进行t检验
        self.monthly_ttest = self.monthly_excess.apply(spstats.ttest_1samp, popmean=0)
        # 生成相关报表
        table_convertor = report.table_convertor
        format_set = dict(zip(group_columns, len(group_columns) * ['pct2p']))
        format_set['benchmark'] = 'pct2p'
        format_set = report.trans2formater(format_set)
        self.yearlyret_tab = table_convertor.format_df(self.yearly_ret.reset_index().
                                                       rename(columns={'index': 'time'}),
                                                       format_set,
                                                       order=['time'] + group_columns +
                                                       ['benchmark'])
        ttest = _transttest(self.monthly_ttest)
        ttest_formatset = {'tvalue': ('floatnp', 4), 'pvalue': ('floatnp', 4)}
        ttest_formatset = report.trans2formater(ttest_formatset)
        self.ttest_tab = table_convertor.format_df(ttest, ttest_formatset,
                                                   order=['group_name', 'tvalue', 'pvalue'])

    def run(self):
        '''
        回测启动函数
        '''
        raw_data = self.get_rawdata()
        obs_data = self.get_observable_data(raw_data)
        sig_data = self.processing_data(obs_data)
        reb_dates = self.get_rebdates()
        self.processing_backtest(sig_data, reb_dates)
        self.analysis()

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


def get_industrycls(ind_cls, date):
    '''
    计算当前时间的行业分类
    Parameter
    ---------
    ind_cls: pd.DataFrame
        行业分类数据，列名要求为code, time和abbr
    date: datetime or compatible type
        需要获取的行业分类的时间

    Return
    ------
    out: pd.DataFrame
        包含三列，code和abbr

    Notes
    -----
    使用这种方法获取行业分类会导致之前申万行业分类的获取速度下降很大，主要是因为申万已经有了每个
    交易日的股票行业数据，在本函数中会使用排序等，从而导致对于数据量较大的申万行业分类速度很慢，
    推荐使用中信行业分类
    '''
    data = ind_cls.loc[ind_cls.time <= date]    # 假设所有行业相关的信息都在开盘前发布
    data = data.sort_values('time')
    by_code = data.groupby('code')
    out = by_code.tail(1).reset_index(drop=True).drop('time', axis=1)
    return out


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
        # 获取再平衡日的行业分类，减少不必要的数据加载
        if industry_cls is None:
            ind_cls = None
        else:
            # ind_cls = industry_cls.loc[industry_cls.time == reb_dt]
            ind_cls = get_industrycls(industry_cls, reb_dt)
        # 过滤不能交易的股票，此处会自动将建仓日不存在数据的股票过滤
        tradeable_stocks = quotes_data.loc[(quotes_data.time == chg_dt) & (~quotes_data.STTag) &
                                           quotes_data.tradeable, 'code'].tolist()
        tradeable_stocks = set(tradeable_stocks).intersection(constituent)

        # 获取当前信号数据，加入指数成份过滤，更新pandas的版本(0.20.1)后发现此步骤速度特别慢
        reb_sig_data = signal_data.loc[(signal_data['time'] == reb_dt) &
                                       (signal_data['code'].isin(tradeable_stocks))]
        # 根据信号函数计算当前的股票组
        valid_stocks = stock_filter(reb_sig_data, ind_cls)
        # valid_stocks = [[c for c in group if c in tradeable_stocks]
        #                 for group in valid_stocks]
        holdings[chg_dt] = PositionGroup(valid_stocks)
    return holdings


def cal_nav(holdings, end_date, quotes, ini_capital=1e9, normalize=True, cal_to=False, **kwargs):
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
        cal_to: 是否计算换手率，默认为False
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
    cols = ['time'] + ['group_%02d' % i for i in range(1, len(holdings[start_date]) + 1)]    # 结果列名
    turnover = list()   # 换手率
    # 交易日循环
    for td, tq_idx in zip(sorted(tds_map), tqdm(tds_map)):
        # 当前为换仓日，第一个建仓日一定为换仓日
        if tds_map[td] == td:
            cur_pos = holdings[td]
            if portfolio_record is None:    # 第一次建仓
                portfolio_record = [Portfolio(pd.DataFrame({'code': [], 'num': []}), ini_capital)
                                    for i in range(len(cur_pos))]
            tmp_portrecord = list()
            for port_idx, pos in enumerate(cur_pos):
                # 此处建仓实际上假设之前的股票都在今天开盘卖出，然后再按照开盘价买入新的仓位
                # 虽然这种假设不合理，但是如果使用上个交易日的数据会带来一些复杂性，且一个交易日的
                # 收盘价和随后一天的开盘价的差值一般不会太大，故忽略这方面的影响
                tmp_port = build_pos(pos, portfolio_record[port_idx].mkt_value(quotes, td, 'open'),
                                     quotes, td, **kwargs)
                tmp_portrecord.append(tmp_port)
            # TODO 在此处添加换手率计算，为了保证兼容性，可以考虑加入默认参数，默认不返回换手率
            tmp_to = dict()
            ports = zip(portfolio_record, tmp_portrecord)
            for port_idx, p in enumerate(ports):
                tmp_to['turnover_%02d' % (port_idx + 1)] = cal_turnover(p[0], p[1], quotes, td)
            tmp_to['time'] = td
            turnover.append(tmp_to)
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
            nav['group_%02d' % i] = nav['group_%02d' % i] / ini_capital
    if cal_to:
        turnover = pd.DataFrame(turnover)
        return nav, turnover
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
    wc = WeightCalculator(buildpos_type)
    # # 目前没有实现其他权重计算方式
    # if buildpos_type != 'money-weighted':
    #     raise NotImplementedError

    # # 计算资金分配
    # try:
    #     cash_alloc = cash / len(pos.pos)
    # except ZeroDivisionError:
    #     return Portfolio(pd.DataFrame(), cash)

    # 计算每只股票的购买数量
    multiple = 100
    # 此处行情数据使用方法会自动剔除建仓日退市的股票，但是如果存在退市的公司，则会导致每只
    # 股票分配的资金相对比较少，问题不需要考虑，因为当天没有数据的股票会被过滤掉
    data = quotes.loc[quotes.code.isin(pos.pos) & (quotes.time == date)]
    weight = wc.cal_weight(pos, data)
    data = data.assign(num=lambda x: x.code.map(weight))
    # data['num'] = data.code.map(weight)
    data['num'] = cash * data['num'] / data[price_col]
    # data['num'] = cash_alloc / data[price_col]
    data['num'] = data['num'].apply(lambda x: int(x / multiple) * multiple)
    residual_cash = cash - (data['num'] * data[price_col]).sum()
    res = Portfolio(data.loc[:, ['code', 'num']], residual_cash)
    return res


def cal_turnover(port_ante, port_post, quote, date, price_type='open', include_cash=True):
    '''
    计算组合的换手率，换手率定义为T = 1/2 * sum(w_i^new - w_i^old)
    Parameter
    ---------
    port_ante: Portfolio
        在换手前的组合持仓
    port_post: Portfolio
        在换手后的组合持仓
    quote: pd.DataFrame
        行情数据
    date: datetime or other compatible types
        换仓时间
    price_type: str, default "open"
        换仓时价格计算类型
    include_cash: boolean, default True
        是否将现金占比包含在换仓计算中

    Return
    ------
    out: float
        换手率

    Notes
    -----
    在计算换手率的过程中，假设当前组合中有退市的股票时，先对退市股票作退市处理，将其转换为现金，然后
    再计算总体的换手率；且假设只有老的组合中存在退市的情况，新的组合中没有退市股票。
    注意：这种方法计算出的结果存在低估换手率的可能
    '''
    quote_cs = quote.loc[quote.time == date]
    if not set(port_ante.pos.code).issubset(quote_cs.code):  # 出现退市的情况
        delist_codes = set(port_ante.pos.code).difference(quote_cs.code)
        port_ante.delist_value(delist_codes, quote)
    # 计算组合持仓的各个股票的市值
    ante_df = pd.merge(port_ante.pos, quote_cs, on='code', how='left')
    post_df = pd.merge(port_post.pos, quote_cs, on='code', how='left')
    ante_df = ante_df.assign(stock_mkv=lambda x: x[price_type] * x.num).\
        loc[:, ['code', 'stock_mkv']]
    post_df = post_df.assign(stock_mkv=lambda x: x[price_type] * x.num).\
        loc[:, ['code', 'stock_mkv']]
    if include_cash:
        ante_df = ante_df.append({'code': 'CASH', 'stock_mkv': port_ante.cash}, ignore_index=True)
        post_df = post_df.append({'code': 'CASH', 'stock_mkv': port_post.cash}, ignore_index=True)
    # 计算组合持仓的权重，这个地方可以直接使用计算市值的函数计算
    ante_df = ante_df.assign(weight=lambda x: x.stock_mkv / x.stock_mkv.sum())
    post_df = post_df.assign(weight=lambda x: x.stock_mkv / x.stock_mkv.sum())
    # 计算换手率
    ante_df = ante_df.set_index('code')
    post_df = post_df.set_index('code')
    ante_df = ante_df.reindex(ante_df.index.union(post_df.index)).fillna(0)
    post_df = post_df.reindex(post_df.index.union(ante_df.index)).fillna(0)
    to = post_df.weight - ante_df.weight
    out = np.sum(np.abs(to)) * 0.5
    return out


def cal_IC(factor_data, quotes, factor_col, rebalance_dates, price_type='close',
           warning_threshold=10):
    '''
    计算IC值
    即，在rebalance day计算因子值，计算当前因子值和下一期股票收益的相关性
    @param:
        factor_data: df格式，因子值，必须包含['time', 'code', factor_col]列
        quotes: 行情数据
        factor_col: 因子值所在的列
        rebalance_dates: 因子计算的日期序列，股票的收益也会是在相隔两个rebalance day之间的收益
        price_type: 计算收益的价格类型，默认为close
        warning_threshold: 触发警告的数量阈值，因为有些股票会在月中退市，在计算时会将这些股票直接
            剔除，但是为了检查剔除的数量是否合理，设置警告的阈值，当剔除的数量超过这个阈值会触发
            警告信息
    @return:
        IC值的时间序列，pd.Series格式，索引为rebalance_dates（剔除了最后一个日期）
    假设未来一个月会退市的股票直接被提出，不参与计算IC
    '''
    factor_data = factor_data.loc[factor_data.time.isin(rebalance_dates)]
    quotes = quotes.loc[quotes.time.isin(rebalance_dates), ['time', 'code', price_type]]
    quotes = quotes.sort_values(['code', 'time']).reset_index(drop=True)
    by_code = quotes.groupby('code')
    quotes['ret'] = by_code[price_type].transform(lambda x: x.pct_change().shift(-1))
    quotes = quotes.dropna()
    # 将合并方式从right改为inner，因为经常碰到的情况是factor_data没有数据，而行情有数据
    data = pd.merge(factor_data, quotes, on=['code', 'time'], how='inner')
    if len(data) < len(factor_data) - warning_threshold:
        print('Warning: %d stocks data have been removed' % (len(factor_data) - len(data)))
        # tmp = pd.merge(factor_data, quotes, on=['code', 'time'], how='left')
        # tmp = tmp.loc[np.any(pd.isnull(tmp), axis=1)]
    by_time = data.groupby('time')
    ICs = by_time.apply(lambda x: x[factor_col].corr(x.ret))
    return ICs.sort_index()


# --------------------------------------------------------------------------------------------------
# 持仓分析模块
def holding2df(holding, fill='NAC'):
    '''
    将持仓转化为df的格式：
    包含了time和group_i列，其中，group_i有多少列视持仓分组的数量而定，按照time进行排列
    @param:
        holding: get_daily_holding返回的结果，即字典类型{time: PositionGroup}
        fill: 不同组别股票数量不一致，为了对齐需要进行填充，默认填充'NAC'(Not A Code)
    @return:
        转化后的df，对于不同的组别股票数量可能不一致，需要进行填充，以股票数量最多的组为标杆，
        其他组数量达不到则填充fill参数
    '''
    res = pd.DataFrame()
    for td in holding:
        group = holding[td]
        max_stocknum = max([len(g.pos) for g in group])
        tmp = dict()
        for idx, g in enumerate(group):
            tmp['group_%02d' % (idx + 1)] = list(g.pos) + (max_stocknum - len(g.pos)) * [fill]
        tmp = pd.DataFrame(tmp)
        tmp['time'] = td
        res = res.append(tmp)
    res = res.sort_values(['time']).reset_index(drop=True)
    return res


def _transttest(ttest_res):
    '''
    将t检验的结果转换为DataFrame的格式
    Parameter
    ---------
    ttest_res: Series
        索引为组名，值为spstats.ttest_1samp的返回结果

    Return
    ------
    out: DataFrame
        包含有三列，分别为group_name, tvalue和pvalue，每列为各组对应的值
    '''
    col_name = ['group_name', 'tvalue', 'pvalue']
    res = list()
    for col in ttest_res.iteritems():
        tmp = [col[0]] + list(col[1])
        res.append(dict(zip(col_name, tmp)))
    return pd.DataFrame(res)


if __name__ == '__main__':
    def get_stocks(sig_data, ind_cls):
        sig_data = sig_data.assign(log_mktvalue=lambda x: np.log(x['mktvalue'])).\
            assign(group_cls=lambda x: pd.qcut(x.log_mktvalue, 10, labels=range(1, 11)))
        by_cls = sig_data.groupby('group_cls')
        res = list()
        for g_idx in sorted(by_cls.groups):
            tmp_data = by_cls.get_group(g_idx)
            res.append(tmp_data.code.tolist())
        return res

    class SizeBT(Backtest):

        def get_rawdata(self):
            return self.quote_loader.load_data()

        def processing_data(self, obs_data):
            return obs_data

        def processing_backtest(self, sig_data, reb_dates):
            quote = self.quote_loader.load_data()
            ind_cls = self.ind_loader.load_data()
            index_constituent = self.constituent_loader.load_data()
            self.reb_dates = reb_dates
            # 进行回测
            stock_filter = self.stock_filter
            weight_method = self.weight_method
            holding = get_daily_holding(sig_data, quote, index_constituent, ind_cls, stock_filter,
                                        reb_dates)
            self.nav, self.turnover = cal_nav(holding, reb_dates[-1], quote,
                                              buildpos_type=weight_method, cal_to=True)
            self.holding = holding

    quote_loader = datatoolkits.DataLoader(
        'HDF', r"F:\实习工作内容\东海证券\基础数据\行情数据\quote_store.h5", key='quote_adj_20170510')
    constituent_loader = datatoolkits.DataLoader(
        'HDF', r"F:\实习工作内容\东海证券\基础数据\指数成份\index_constituents.h5",
        key='Index_000985')
    industry_loader = datatoolkits.DataLoader('None', '')
    sizebt = SizeBT(quote_loader, constituent_loader, industry_loader, get_stocks, '2015-01-01',
                    '2016-12-31')
    sizebt.run()

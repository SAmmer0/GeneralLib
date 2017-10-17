#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-08-21 17:15:53
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

# 标准库
from collections import OrderedDict
import pdb

# 第三方库
import pandas as pd
from tqdm import tqdm

# 本地库
from factortest.grouptest.utils import (Stock, Portfolio, EqlWeightCalc, MkvWeightCalc,
                                        stock_filter_template)
from factortest.utils import HDFDataProvider, NoneDataProvider, WeekRebCalcu, MonRebCalcu
from dateshandle import get_tds
from fmanager import get_factor_dict
from factortest.const import *

# ------------------------------------------------------------------------------


class BacktestConfig(object):
    '''
    回测配置设置类
    '''

    def __init__(self, start_date, end_date, quote_provider, weight_calculator,
                 tradedata_provider, reb_calculator, group_num=10, commission_rate=0.,
                 init_cap=1e10, show_progress=True):
        '''
        Parameter
        ---------
        start_date: str, datetime or other compatible type
            回测开始时间
        end_date: str, datetime or other compatible type
            回测结束时间
        quote_provider: DataProvider
            计算净值用的数据提供器
        weight_calculator: WeightCalc
            权重计算器
        tradedata_provider: DataProvider
            能否交易的数据的提供器
        reb_calculator: RebCalc
            再平衡日计算器
        group_num: int
            因子测试分的组数
        commission_rate: float
            交易成本
        init_cap: float or int
            初始资本
        show_progress: bool, default True
            是否显示回测进度，默认显示
        '''
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.quote_provider = quote_provider
        self.weight_calculator = weight_calculator
        self.tradedata_provider = tradedata_provider
        self.reb_calculator = reb_calculator
        self.group_num = group_num
        self.commission_rate = commission_rate
        self.init_cap = init_cap
        self.show_progress = show_progress


class Backtest(object):
    '''
    回测类
    '''

    def __init__(self, config, stock_filter, *args, **kwargs):
        '''
        Parameter
        ---------
        config: BacktestConfig
            回测相关配置
        stock_filter: function
            用于计算股票分组的函数，形式为function(date, *args, **kwargs)，返回值要求为
            {order: [secu_codes]}，其中order为对应股票组合的顺序，要求为range(0, config.group_num)
        args: tuple like arguments
            stock_filter需要使用的位置参数
        kwargs: dict like arguments
            stock_filter需要使用的键值参数
        '''
        self._config = config
        self._tds = get_tds(config.start_date, config.end_date)
        self.holding_result = OrderedDict()    # 用于记录各组持仓的股票
        # 用于记录附带权重的持仓，这个数据每组的持仓可能跟holding_result中不同，
        # 因为该持仓会考虑到能否交易等相关问题
        self.weighted_holding = OrderedDict()
        self.navs = OrderedDict()
        self._stock_filter = stock_filter
        self._args = args
        self._kwargs = kwargs
        self._ports = {i: Portfolio(self._config.init_cap) for i in range(self._config.group_num)}
        self._navs_pd = None
        self._offset = 10    # 避免满仓是因为小数点的问题导致资金溢出

    def build_portfolio(self, port_id, secu_list, date, last_td):
        '''
        建仓函数

        Parameter
        ---------
        port_id: str
            组合的编号
        secu_list: list of string
            需要加入组合的证券
        date: datetime or other compatible types
            加入组合的时间
        last_td: datetime or other compatible types
            持仓计算日的时间
        '''
        # 只买入今日能够交易的股票
        tradeable_stocks = self._config.tradedata_provider.get_csdata(date)
        tradeable_stocks = tradeable_stocks.loc[tradeable_stocks == 1].index.tolist()
        secu_list = list(set(secu_list).intersection(tradeable_stocks))
        # if len(secu_list) == 0:
        # pdb.set_trace()

        # 这个地方使用如果使用当天的市值计算权重是有问题的，因为当前收盘前不知道当天的市值
        # 应当使用上个交易日的市值计算相关的权重
        weights = self._config.weight_calculator(secu_list, date=last_td)  # 计算权重
        # 记录权重
        weights_recorder = self.weighted_holding.get(date, {})
        weights_recorder[port_id] = weights
        if date not in self.weighted_holding:
            self.weighted_holding[date] = weights_recorder
        port = self._ports[port_id]
        port_mkv = port.sell_all(date) - self._offset    # 卖出全部金融工具
        weights = {code: Stock(code, quote_provider=self._config.quote_provider).
                   construct_from_value(weights[code] * port_mkv, date)
                   for code in weights}
        # pdb.set_trace()
        port.buy_seculist(weights.values(), date)

    def run_bt(self):
        '''
        开启回测
        '''
        chg_pos_tag = False     # 用于标记是否到了换仓日
        chg_pos = None      # 用于记录下次换仓时持仓，类型为dict
        last_td = None      # 用于记录换仓日前的数据持仓计算日
        if self._config.show_progress:    # 需要显示进度
            tds_iter = zip(self._tds, tqdm(self._tds))
        else:
            tds_iter = enumerate(self._tds)
        for _idx, td in tds_iter:
            if chg_pos_tag:     # 表明当前需要换仓
                for port_id in self._ports:
                    self.build_portfolio(port_id, chg_pos[port_id], td, last_td)
                chg_pos_tag = False

            if self._config.reb_calculator(td):     # 当前为计算日
                chg_pos = self._stock_filter(td, *self._args, **self._kwargs)
                chg_pos_tag = True
                last_td = td
                self.holding_result[td] = chg_pos

            # 记录净值信息
            nav = {port_id: self._ports[port_id].refresh_value(td)
                   for port_id in self._ports}
            self.navs[td] = nav

    @property
    def navpd(self):
        '''
        pd.DataFrame格式的净值数据
        '''
        if self._navs_pd is not None:
            return self._navs_pd
        else:
            self._navs_pd = pd.DataFrame(self.navs).T
            self._navs_pd = self._navs_pd / self._navs_pd.iloc[0]   # 转化为净值
            self._navs_pd.columns = ['group_%02d' % c for c in self._navs_pd.columns]
            return self._navs_pd

    @property
    def start_date(self):
        '''
        返回当前回测的开始时间
        Return
        ------
        out: datetime like
            回测的开始时间
        '''
        return self._config.start_date

    @property
    def end_date(self):
        '''
        返回当前回测的结束时间
        Return
        ------
        out: datetime like
            回测的结束时间
        '''
        return self._config.end_date


class FactortestTemplate(object):
    '''
    简易的因子测试模板，仅包含回测功能
    '''

    def __init__(self, factor, start_time, end_time, weight_method=TOTALMKV_WEIGHTED,
                 reb_method=MONTHLY, group_num=5, stock_pool=None, industry_neutral=None,
                 show_progress=True):
        '''
        Parameter
        ---------
        factor: str or DataProvider
            如果提供的是因子名称，必须在fmanager.api.get_factor_dict的返回值中可以找到，
            如果是数据提供器，数据的长度必须能够与start_time, end_time契合
        start_time: datetime or other compatible types
            回测的开始时间
        end_time: datetime or other compatible types
            回测的结束时间
        weight_method: str, default equal-weighted
            权重计算方法，目前支持equal-weighted、totalmkv-weighted和floatmkv-weighted
            也可以通过const文件中的方法设置
        reb_mothod: str, default monthly
            换仓频率，目前支持weekly和monthly
        group_num: int, default 5
            分组的数量
        stock_pool: str or DataProvider, default None
            股票池限制规则，如果参数为str，要求能够在fmanager.api.get_factor_dict的返回值中可以找到
        industry_neutral: str, default None
            行业中性化的行业分类规则，要求能够在fmanager.api.get_factor_dict的返回值中可以找到
        show_progress: boolean, default True
            是否显示进度
        '''
        self._factor_dict = get_factor_dict()
        self.start_time = pd.to_datetime(start_time)
        self.end_time = pd.to_datetime(end_time)
        if isinstance(factor, str):
            self.factordata_provider = HDFDataProvider(self._factor_dict[factor]['abs_path'],
                                                       self.start_time, self.end_time)
        else:   # 直接使用提供器数据
            self.factordata_provider = factor
        self.weight_method = weight_method
        self.reb_method = reb_method
        self.group_num = group_num
        # 参数检查
        self._check_parameter()
        # 加载ST和TRADEABLE数据
        self._st_provider = HDFDataProvider(self._factor_dict['ST_TAG']['abs_path'],
                                            self.start_time, self.end_time)
        self._tradeable_provider = HDFDataProvider(self._factor_dict['TRADEABLE']['abs_path'],
                                                   self.start_time, self.end_time)
        # 加载价格数据
        self._price_provider = HDFDataProvider(self._factor_dict['ADJ_CLOSE']['abs_path'],
                                               self.start_time, self.end_time)
        # 加载股票池相关数据
        if stock_pool is not None:
            if isinstance(stock_pool, str):
                self._stockpool_provider = HDFDataProvider(self._factor_dict[stock_pool]['abs_path'],
                                                           self.start_time, self.end_time)
            else:
                self._stockpool_provider = stock_pool
        else:
            self._stockpool_provider = NoneDataProvider()
        # 加载行业分类数据
        if industry_neutral is not None:
            self._industry_provider = HDFDataProvider(self._factor_dict[industry_neutral]['abs_path'],
                                                      self.start_time, self.end_time)
        else:
            self._industry_provider = NoneDataProvider()
        self.show_progress = show_progress

    def _check_parameter(self):
        '''
        检查权重和换仓频率参数是否设置正确，并设置好相关数据
        '''
        valid_weighted_methods = [EQUAL_WEIGHTED, TOTALMKV_WEIGHTED, FLOATMKV_WEIGHTED]
        valid_reb_motheds = [WEEKLY, MONTHLY]
        assert self.weight_method in valid_weighted_methods,\
            'Weighted method setting ERROR, you provide {yp},'.format(yp=self.weight_method) +\
            'right choices are {rc}'.format(rc=valid_weighted_methods)
        assert self.reb_method in valid_reb_motheds,\
            'Rebalance date method setting ERROR, you provide {yp},'.format(yp=self.reb_method) +\
            'right choices are {rc}'.format(rc=valid_reb_motheds)
        self.reb_method_obj = self._rebstr2obj(self.reb_method)
        self.weight_method_obj = self._weightstr2obj(self.weight_method)

    def _rebstr2obj(self, reb_method):
        '''
        将换仓频率设置转换为对应的对象
        Parameter
        ---------
        reb_method: str
            换仓频率设置

        Return
        ------
        out: Rebcalcu
        '''
        if reb_method == WEEKLY:
            return WeekRebCalcu(self.start_time, self.end_time)
        if reb_method == MONTHLY:
            return MonRebCalcu(self.start_time, self.end_time)

    def _weightstr2obj(self, weighted_method):
        '''
        将计算权重的设置转换为对应的对象
        Parameter
        ---------
        weighted_method: str

        Return
        ------
        out: EqlWeightCalc
        '''
        if weighted_method == EQUAL_WEIGHTED:
            return EqlWeightCalc()
        if weighted_method == FLOATMKV_WEIGHTED:
            float_provider = HDFDataProvider(self._factor_dict['FLOAT_MKTVALUE']['abs_path'],
                                             self.start_time, self.end_time)
            return MkvWeightCalc(float_provider)
        if weighted_method == TOTALMKV_WEIGHTED:
            total_provider = HDFDataProvider(self._factor_dict['TOTAL_MKTVALUE']['abs_path'],
                                             self.start_time, self.end_time)
            return MkvWeightCalc(total_provider)

    def run_test(self):
        '''
        启动回测
        '''
        stock_filter = stock_filter_template(self._st_provider, self._tradeable_provider,
                                             self._stockpool_provider, self._industry_provider,
                                             self.group_num)
        conf = BacktestConfig(self.start_time, self.end_time, self._price_provider,
                              self.weight_method_obj, self._tradeable_provider,
                              self.reb_method_obj, self.group_num, show_progress=self.show_progress)
        bt = Backtest(conf, stock_filter, fd_provider=self.factordata_provider)
        bt.run_bt()
        return bt

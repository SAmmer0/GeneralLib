#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-20 14:21:48
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
用于管理当前添加的需要监控的组合
管理的内容包含：
    组合初始化
    组合价值更新
    组合持仓更新
    将组合的相关数据写入到文件中
    从文件中读取组合的相关数据
    将所有投资组合纳入到一个容器中集中进行管理
'''
# 系统模块
from collections import OrderDict
from datetime import datetime
# 第三方模块
import pandas as pd
import numpy as np
# 本地模块
from portmonitor.const import PORT_DATA_PATH, PORT_CONFIG_PATH, CASH
from datatoolkits import dump_pickle, load_pickle, isclose
from factortest.utils import load_rebcalculator, FactorDataProvider
from factortest.const import EQUAL_WEIGHTED, FLOATMKV_WEIGHTED, TOTALMKV_WEIGHTED
from factortest.grouptest.utils import EqlWeightCalc, MkvWeightCalc
from fmanager.update import get_endtime
from dateshandle import tds_shift, get_tds


# --------------------------------------------------------------------------------------------------
# 类
class PortfolioData(object):
    '''
    用于存储组合的数据，数据包含组合的最新持仓、历史持仓、组合当前资产总值时间序列
    '''

    def __init__(self, port_id, cur_holding, hist_holding, av_ts):
        '''
        Parameter
        ---------
        port_id: str
            组合的ID
        cur_holding: dict
            当前持仓，格式为{code: num}
        hist_holding: OrderDict
            历史持仓，格式为{time: {code: num}}，可能的时间包含计算日时间和发生分红送股的时间
        av_ts: pd.Series
            资产总值的时间序列
        '''
        self.id = port_id
        self.curholding = cur_holding
        self.histholding = hist_holding
        self.assetvalue_ts = av_ts

    @property
    def update_time(self):
        return self._assetvalue_ts.index[-1]

    @property
    def last_asset_value(self):
        return self._assetvalue_ts.iloc[-1]

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        return self.__dict__


class PortfolioMoniData(object):
    '''
    组合的监控数据，包含两个部分：
        组合的配置信息（无法进行序列化）
        组合的PortfolioData
    '''

    def __init__(self, port_config):
        '''
        Parameter
        ---------
        port_config: portmonitor.utils.MonitorConfig
            组合的配置信息
        '''
        self._port_config = port_config
        self._data_path = get_portdata_path(port_config.port_id)
        self._quote_cache = None
        self._port_data = None
        self._reb_calculator = None
        self._today = get_endtime(datetime.now())
        self._load_data()

    def _load_data(self):
        '''
        初始化监控数据，先尝试从文件中获取，如果没有找到，则初始化一个监控数据对象，仅在初始化时调用
        '''
        try:
            self._port_data = load_pickle(self._data_path)
            self._reb_calculator = load_rebcalculator(self._port_config.rebalance_type,
                                                      self._port_data.update_time, datetime.now())
        except FileNotFoundError:
            recent_rbd = self._get_recent_rbd()
            self._port_data = PortfolioData(self._port_config.port_id, None,
                                            OrderDict(),
                                            av_ts=pd.Series({recent_rbd: self._port_config.init_cap}))

    def _get_recent_rbd(self):
        '''
        获取在当前时间之前（包括当前时间）范围内的最近的换仓日，仅在初始化数据文件时调用

        Return
        ------
        out: datetime
        '''
        if self._reb_calculator is None:
            today = datetime.now()
            start_time = tds_shift(today, 40)
            self._reb_calculator = load_rebcalculator(self._port_config.rebalance_type, start_time,
                                                      today)
        return self._reb_calculator.reb_points[-1]

    def _load_weight_calculator(self):
        '''
        加载权重计算器
        '''
        weighted_method = self._port_config.weight_method
        if weighted_method == EQUAL_WEIGHTED:
            self._weight_cal = EqlWeightCalc()
        if weighted_method == FLOATMKV_WEIGHTED:
            float_provider = FactorDataProvider('FLOAT_MKTVALUE', self._port_data.update_time,
                                                self._today)
            self._weight_cal = MkvWeightCalc(float_provider)
        if weighted_method == TOTALMKV_WEIGHTED:
            total_provider = FactorDataProvider('TOTAL_MKTVALUE', self._port_data.update_time,
                                                self._today)
            self._weight_cal = MkvWeightCalc(total_provider)

    def _cal_num(self, weight, price_provider, date, total_cap):
        '''
        根据权重计算相应的持仓

        Parameter
        ---------
        weight: dict
            权重参数，格式为{code: weight}
        price_provider: DataProvider
            价格数据提供器
        date: datetime like
            计算的时间
        total_cap: float
            总资本

        Return
        ------
        out: dict
            转换成数量后的持仓，格式为{code: num}，里面包含一个特殊资产：现金

        Notes
        -----
        此处换仓假设换仓时资产的总价值是由上个持仓在计算日的收盘价计算得来，然后以计算日的收盘价
        来计算换仓后的股票仓位
        '''
        total_cap = total_cap - 10  # 避免各个证券的价值加总后大于原总资产价值（因为计算机小数加减可能导致溢出问题）
        weight_value = {code: weight[code] * total_cap for code in weight}
        price = price_provider.get_csdata(date)
        out = {code: weight_value[code] / price.loc[code] for code in weight_value}
        cash = total_cap - np.sum([out[code] * price.loc[code] for code in out])  # 现金
        out[CASH] = cash
        assert cash >= 0, ValueError('Cash cannot be negetive')
        return out

    @staticmethod
    def _cal_holding_value(holding, date, last_td, close_provider, prevclose_provider):
        '''
        计算当前持仓的价值

        Parameter
        ---------
        holding: dict
            最近计算日计算出的股票仓位，格式为{code: num}
        date: datetime like
            当前交易日的时间
        last_td: datetime like
            上个交易日的时间
        close_provider: DataProvider
            收盘价数据提供其
        prevclose_provider: DataProvider
            前收盘数据提供器，用于识别分红送股的交易日

        Return
        ------
        out: float
            给定交易日的持仓总价值
        holding_chg_flag: boolean
            分红送股时间发生的标记，如果至少有一支股票发生了该行为或者出现了退市事件，则为True
        new_holding: dict
            发生分红送股后更新的持仓，如果没有发生该事件，则其值与传入的holding参数相同，可先通过
            divident_flag进行分红送股判断，然后再更新持仓参数

        Notes
        -----
        关于分红送股事件，是采用将上个交易日的收盘价与本交易日的前收盘价做对比来识别的，如果这两个
        数据不同，说明在本交易日执行了除权，此时对对应股票的持有量按照比例进行调整
        new_holding = old_holding * last_close / prev_close
        调整隐含的假设是如果是进行了分红，则立马将分红的现金按照当前交易日的前收盘价转换为对应数量
        的股票（这个转换不太切合实际），如果是按照送股或者其他扩展股票数量的行为，则不影响
        '''
        close_data = close_provider.get_csdata(date)
        lastclose_data = close_provider.get_csdata(last_td)
        prevclose_data = prevclose_provider.get_csdata(date)
        holding_chg_flag = False   # 用于标记是否至少有一支股票发生分红送股事件或者退市
        total_value = 0
        new_holding = {}
        for code in holding:
            if code == CASH:
                total_value += holding[code]
                new_holding[code] = new_holding.get(code, 0) + holding[code]
            else:
                close = close_data.loc[code]
                lastclose = lastclose_data.loc[code]
                # 发生退市事件，假设退市当天收盘价为NaN，上个收盘价还有非NaN的数据
                # 对于退市事件的处理：发生退市时，以上个收盘价将股票全部转换为现金
                if np.isnan(close):
                    assert not np.isnan(lastclose), 'Error, last close price is NaN!'
                    secu_value = holding[code] * lastclose
                    total_value += secu_value
                    new_holding[CASH] = new_holding.get(CASH, 0) + secu_value
                    holding_chg_flag = True
                    continue
                prevclose = prevclose_data.loc[code]
                if not isclose(lastclose, prevclose):    # 表明当前发生了分红送股等事件
                    ratio = lastclose / prevclose
                    holding_chg_flag = True
                else:
                    ratio = 1
                new_num = holding[code] * ratio
                total_value += new_num * close
                new_holding[code] = new_num
        return total_value, holding_chg_flag, new_holding

    def refresh_portvalue(self):
        '''
        刷新到当前时间为止的总资产的价值，刷新的结果更新到self._port_data中

        Notes
        -----
        计算持仓的价值时，注意退市股票的处理，如果有则在该交易日按照上个交易日的价格将证券换成现金
        还有就是在持仓中，需要加入现金
        '''
        port_data = self._port_data
        start_time = port_data.update_time
        closeprice_provider = FactorDataProvider('CLOSE', start_time, self._today)
        prevclose_provider = FactorDataProvider('PREV_CLOSE', start_time, self._today)
        tds = get_tds(start_time, self._today)
        last_td = tds[0]
        for td in tds[1:]:
            if self._reb_calculator(last_td):   # 表示上个交易日是计算日，需要重新计算持仓，并在本交易日切换
                last_cap = port_data.last_asset_value
                new_holding = self._port_config.stock_filter(last_td)
                new_holding = self._cal_num(new_holding, closeprice_provider, last_td, last_cap)
                port_data.curholding = new_holding
                port_data.histholding[last_td] = new_holding
            asset_value, div_flag, new_holding = self._cal_holding_value(td, last_td,
                                                                         closeprice_provider,
                                                                         prevclose_provider)
            if div_flag:    # 发生分红送股事件，持仓数量需要更新
                port_data.curholding = new_holding
                port_data.histholding[td] = new_holding
            port_data.assetvalue_ts = port_data.assetvalue_ts.append({td: asset_value})
            # 更新上个交易日的时间
            last_td = td
        self._port_data = port_data


# --------------------------------------------------------------------------------------------------
# 函数


def get_portdata_path(port_id):
    '''
    获取组合数据的存储文件的地址

    Parameter
    ---------
    port_id: str
        组合的id

    Return
    ------
    out: str
        id对应组合的数据存储路径
    '''
    return PORT_DATA_PATH + '\\' + port_id + '.pickle'

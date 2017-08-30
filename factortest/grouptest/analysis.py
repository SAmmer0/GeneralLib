#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-08-26 19:19:23
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
# 系统模块
from abc import ABCMeta, abstractmethod
# 第三方模块
from scipy.stats import ttest_1samp
import pandas as pd
# 本地模块
from datatoolkits import price2nav
from report import brief_report


class Analysor(object, metaclass=ABCMeta):
    '''
    分析器基类，用于提供接口
    '''

    def __init__(self, bt):
        '''
        Parameter
        ---------
        bt: BackTest
            需要被分析的回测实例
        '''
        self._bt = bt

    @abstractmethod
    def analyse(self):
        '''
        执行相关的分析功能
        '''
        pass

    def output(self):
        '''
        将分析结果转换为可以展示的形式，并返回
        '''
        pass


class NavAnalysor(Analysor):
    '''
    净值分析器，主要用来计算一般的净值评估指标
    '''

    def __init__(self, bt, benchmark, riskfree_rate=0.04):
        '''
        Parameter
        ---------
        bt: BackTest
            需要被分析的回测实例
        benchmark: pd.Series
            基准的净值数据，要求起始时间与bt相同
        riskfree_rate: float, default 0.04
            无风险利率
        '''
        super().__init__(bt)
        self._benchmark = price2nav(benchmark)
        self._riskfree_rate = riskfree_rate

    def analyse(self):
        '''
        对净值数据进行分析，相关指标包括：alpha，beta，最大回撤，最大回撤开始日期，
        最大回撤结束日期，最长回撤期，夏普比率，信息比率，sortino比率
        '''
        nav = self._bt.navpd
        # 基础净值分析指标
        self.basic_msg = nav.apply(lambda x: brief_report(x, self._benchmark,
                                                          self._riskfree_rate, 250))
        # 计算月度收益和年度收益
        self.monthly_ret = nav.groupby(lambda x: x.strftime('%Y-%m')).\
            apply(lambda y: y.iloc[-1] / y.iloc[0] - 1)
        self.yearly_ret = nav.groupby(lambda x: x.year).\
            apply(lambda y: y.iloc[-1] / y.iloc[0] - 1)
        self._benchmark_monthly = self._benchmark.groupby(lambda x: x.strftime('%Y-%m')).\
            apply(lambda y: y.iloc[-1] / y.iloc[0] - 1)
        # 月度超额收益
        self.mexcess_ret = self.monthly_ret.apply(lambda x: x - self._benchmark_monthly)
        # 对超额收益进行t检验
        self.t_test = self._transttest(self.mexcess_ret.apply(ttest_1samp, popmean=0))

    def _transttest(self, ttest_res):
        '''
        将t检验结果转化为pd.DataFrame，列分别为group_name, tvalue, pvalue

        Parameter
        ---------
        ttest_res: pd.Series
            t检验后的结果，为pd.Series形式，列为组名，值为Ttest_1sampResult(statistic=..., pvalue=...)
        '''
        col_name = ['group_name', 'tvalue', 'pvalue']
        res = list()
        for col in ttest_res.iteritems():
            tmp = [col[0]] + list(col[1])
            res.append(dict(zip(col_name, tmp)))
        return pd.DataFrame(res)

    def output(self):
        '''
        将指标信息转换为可在markdown中展示的字符串
        '''
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-08-26 19:19:23
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
# 系统模块
from abc import ABCMeta, abstractmethod
# from collections import namedtuple
# 第三方模块
from scipy.stats import ttest_1samp
import pandas as pd
# 本地模块
from datatoolkits import price2nav
from report import brief_report, trans2formater, table_convertor


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

    @abstractmethod
    def output(self):
        '''
        将分析结果转换为可以展示的形式，并返回，要求不同的类型返回特定的字典作为对应的结果
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
                                                          self._riskfree_rate, 250)).T
        # 计算月度收益和年度收益
        self.monthly_ret = nav.groupby(lambda x: x.strftime('%Y-%m')).\
            apply(lambda y: y.iloc[-1] / y.iloc[0] - 1)
        self.yearly_ret = nav.groupby(lambda x: x.year).\
            apply(lambda y: y.iloc[-1] / y.iloc[0] - 1)
        self._benchmark_monthly = self._benchmark.groupby(lambda x: x.strftime('%Y-%m')).\
            apply(lambda y: y.iloc[-1] / y.iloc[0] - 1)
        self._benchmark_yearly = self._benchmark.groupby(lambda x: x.year).\
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

        Return
        ------
        out: pd.DataFrame
            转换后的结果，列名依次为[group_name, tvalue, pvalue]，行索引为分组
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
        # 年度收益数据
        yearly_data = self.yearly_ret.copy()
        yearly_data = yearly_data.assign(benchmark=self._benchmark_yearly)
        yearly_format = dict(zip(yearly_data.columns, ['pct2p'] * len(yearly_data.columns)))
        yearly_format = trans2formater(yearly_format)
        yearly_tab = table_convertor.format_df(yearly_data.reset_index().
                                               rename(columns={'index': 'time'}),
                                               yearly_format,
                                               ['time']+ list(yearly_data.columns))
        # t检验数据
        ttest_data = self.t_test
        ttest_format = {'tvalue': ('floatnp', 4), 'pvalue': ('floatnp', 4)}
        ttest_format = trans2formater(ttest_format)
        ttest_tab = table_convertor.format_df(ttest_data, ttest_format,
                                              ['group_name', 'tvalue', 'pvalue'])
        # 一般指标
        basicmsg_format = {'alpha': ('floatnp', 4), 'beta': ('floatnp', 4),
                           'info_ratio': ('floatnp', 4), 'mdd': 'pct2p', 'mddt': ('floatnp', 1),
                           'mdd_start': 'date', 'mdd_end': 'date', 'mddt_start': 'date',
                           'mddt_end': 'date', 'sharp_ratio': ('floatnp', 4),
                           'sortino_ratio': ('floatnp', 4)}
        basicmsg_format = trans2formater(basicmsg_format)
        basicmsg_tab = table_convertor.format_df(self.basic_msg.reset_index().
                                                 rename(columns={'index': 'group'}),
                                                 basicmsg_format,
                                                 ['group', 'alpha', 'beta', 'info_ratio', 'sharp_ratio',
                                                  'sortino_ratio', 'mdd', 'mdd_start', 'mdd_end',
                                                  'mddt', 'mddt_start', 'mddt_end'])

        res = dict(momthly_ret=self.monthly_ret, mexcess_ret=self.mexcess_ret,
                   yearly_data=yearly_data, basic_msg=self.basic_msg,
                   ttest=self.t_test, bm_table=basicmsg_tab, yearly_table=yearly_tab,
                   ttest_table=ttest_tab)
        return res


class IndustryAnalysor(Analysor):
    '''
    持仓行业分析器，主要用来分析每个分组的行业分布及其变化
    '''

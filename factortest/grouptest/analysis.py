#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-08-26 19:19:23
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
分析模块，用于分析回测后的相关结果
__version__ = 1.0.0
修改日期：2017-09-01
修改内容：
    添加分析器基类、净值分析器和行业分析器

__version__ = 1.0.1
修改日期：2017-09-04
修改内容：
    添加换手率分析器
'''

# 系统模块
from abc import ABCMeta, abstractmethod
import pdb
from collections import namedtuple
# 第三方模块
from scipy.stats import ttest_1samp
import pandas as pd
import numpy as np
# 本地模块
from factortest.utils import HDFDataProvider
from factortest.grouptest.utils import transholding
from factortest.correlation import get_group_factorcharacter
from datatoolkits import price2nav
from report import brief_report, trans2formater, table_convertor
from fmanager import get_factor_dict, query


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

    def __init__(self, bt, benchmark=None, riskfree_rate=0.04):
        '''
        Parameter
        ---------
        bt: BackTest
            需要被分析的回测实例
        benchmark: pd.Series, default None
            基准的净值数据，要求起始时间与bt相同，如果没有给定，则自动使用同期的上证综指收盘价
            （SSEC_CLOSE）作为参考基准
        riskfree_rate: float, default 0.04
            无风险利率
        '''
        super().__init__(bt)
        if benchmark is None:
            benchmark = query('SSEC_CLOSE', (bt.start_date, bt.end_date)).iloc[:, 0]
        self._benchmark = price2nav(benchmark)
        self._riskfree_rate = riskfree_rate
        self._result_cache = None

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
        Return
        ------
        out: namedtuple(NavAnalysisResult)
            返回相关的分析结果，包含有monthly_ret, mexcess_ret, yearly_data, basic_msg,
            ttest, bm_table, yearly_table, ttest_table，其中后缀为_table的表示为HTML标记
            字符串，用于在日志中写入相关信息，其他均为pd.DataFrame格式
        '''
        # 年度收益数据
        yearly_data = self.yearly_ret.copy()
        yearly_data = yearly_data.assign(benchmark=self._benchmark_yearly)
        yearly_format = dict(zip(yearly_data.columns, ['pct2p'] * len(yearly_data.columns)))
        yearly_format = trans2formater(yearly_format)
        yearly_tab = table_convertor.format_df(yearly_data.reset_index().
                                               rename(columns={'index': 'time'}),
                                               yearly_format,
                                               ['time'] + list(yearly_data.columns))
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
                           'mddt_end': 'date', 'sharpe_ratio': ('floatnp', 4),
                           'sortino_ratio': ('floatnp', 4)}
        basicmsg_format = trans2formater(basicmsg_format)
        basicmsg_tab = table_convertor.format_df(self.basic_msg.reset_index().
                                                 rename(columns={'index': 'group'}),
                                                 basicmsg_format,
                                                 ['group', 'alpha', 'beta', 'info_ratio',
                                                  'sharpe_ratio', 'sortino_ratio', 'mdd',
                                                  'mdd_start', 'mdd_end', 'mddt', 'mddt_start',
                                                  'mddt_end'])

        res = dict(monthly_ret=self.monthly_ret, mexcess_ret=self.mexcess_ret,
                   yearly_data=yearly_data, basic_msg=self.basic_msg,
                   ttest=self.t_test, bm_table=basicmsg_tab, yearly_table=yearly_tab,
                   ttest_table=ttest_tab)
        NavAnalysisResult = namedtuple('NavAnalysisResult', res.keys())
        res = NavAnalysisResult(**res)
        return res

    @property
    def analysis_result(self):
        '''
        执行相关分析，并缓存结果，供使用
        Return
        ------
        out: dict
            净值分析结果
        '''
        if self._result_cache is None:
            self.analyse()
            self._result_cache = self.output()
            return self._result_cache
        else:
            return self._result_cache


class IndustryAnalysor(Analysor):
    '''
    持仓行业分析器，主要用来分析每个分组的行业分布及其变化，仅分析在计算日（或者换仓日）时的持仓
    分布
    '''

    def __init__(self, bt, industry_cls='ZX_IND'):
        '''
        Parameter
        ---------
        bt: BackTest
            需要被分析的回测实例
        industry_cls: str, default ZX_IND
            分析行业持仓时使用的行业分类标准，要求必须能在fmanager.get_factor_dict中找到，默认使用
            中信行业分类
        '''
        super().__init__(bt)
        factor_dict = get_factor_dict()
        self._industry_provider = HDFDataProvider(factor_dict[industry_cls]['abs_path'],
                                                  bt.start_date, bt.end_date)
        chg_td = list(bt.holding_result.keys())[0]  # 任取一个交易日，用于获取所有的行业分类
        self._all_industry = set(self._industry_provider.get_csdata(chg_td))
        self._result_cache = None

    def analyse(self):
        '''
        针对每一个分组，主要是统计每组所选出股票的行业分布，包括股票数量分布和相关持仓权重的分布，
        具体数据为holding_result（股票数量分布），weighted_holding（持仓权重分布）
        '''
        self.holding_result = self._calc_industry_dis(self._bt.holding_result)
        self.weighted_holding = self._calc_industry_dis(self._bt.weighted_holding)

    def _calc_industry_dis(self, data):
        '''
        辅助函数，用于计算行业分布

        Parameter
        ---------
        data: dict
            原始持仓数据，格式为{time: {port_id: dict or list}}

        Return
        ------
        out: dict
            行业分布数据，结构为{port_id: pd.DataFrame(index=time, columns=industry))
        '''
        # 将原始数据进行转换
        data = transholding(data)

        res = dict()
        # 循环计算行业分布
        for port_id in data:
            tmp_data = pd.Series(data[port_id]).reset_index().\
                rename(columns={'index': 'time', 0: 'data'})
            # pdb.set_trace()
            time_idx = tmp_data.time
            tmp_data = tmp_data.apply(lambda x: self._calc_industry_weight(x['data'], x.time),
                                      axis=1)
            tmp_data.index = time_idx
            res[port_id] = tmp_data
        return res

    def _calc_industry_weight(self, holding_data, date):
        '''
        计算每个行业在持仓中的权重或者数量

        Parameter
        ---------
        holding_data: list or dict
            需要解析的数据，如果为list则表示为持仓股票，dict表示为持仓权重
        date: datetime or other compatible types
            计算日或者换仓日的日期

        Return
        ------
        out: pd.Series

        '''
        # pdb.set_trace()
        if isinstance(holding_data, list):
            holding_data = dict(zip(holding_data, [1] * len(holding_data)))
        industry_data = self._industry_provider.get_csdata(date)
        holding_data = pd.Series(holding_data)
        # pdb.set_trace()
        industry_data = industry_data.loc[holding_data.index]
        holding_data = pd.DataFrame({'weight': holding_data, 'industry': industry_data})
        out = holding_data.groupby('industry').weight.sum().\
            reindex(self._all_industry).fillna(0)
        return out

    def output(self):
        '''
        Return
        ------
        out: namedtuple
            字典结构包含的数据为plain_industry_distribution_num，plain_industry_distribution_weight
            和weighted_industry_distribution_weight，分别表示行业个数分布、行业个数占比分布和行业市值
            加权分布
        '''
        plain_inddis_weight = {}
        for port_id in self.holding_result:
            plain_inddis_weight[port_id] = self.holding_result[port_id].\
                transform(lambda x: x / x.sum(), axis=1)
        res = {'plain_industry_distribution_num': self.holding_result,
               'plain_industry_distribution_weight': plain_inddis_weight,
               'weighted_industry_distribution_weight': self.weighted_holding}
        IndustryAnalysisResult = namedtuple('IndustryAnalysisResult', res.keys())
        res = IndustryAnalysisResult(**res)
        return res

    @property
    def analysis_result(self):
        '''
        相关分析结果
        Return
        ------
        out: dict
            行业分布结果
        '''
        if self._result_cache is None:
            self.analyse()
            self._result_cache = self.output()
            return self._result_cache
        else:
            return self._result_cache


class TOAnalysor(Analysor):
    '''
    换手率分析器，用于计算每个分组的换手率情况
    '''

    def __init__(self, bt):
        '''
        Parameter
        ---------
        bt: BackTest
            需要被分析的回测实例
        '''
        super().__init__(bt)
        self._result_cache = None

    def analyse(self):
        '''
        计算换手率，换手率的定义为T = 1/2 * sum(abs(w_i ^ new - w_i ^ old))，
        对应的结果为pd.DataFrame格式，shape为(time_length, group_num)，对应的每个时间点（换仓日）
        的换手率表示本次持仓与上次持仓对比计算的换手率（第一个换仓日换手率必然为0.5或者说50%）
        注：换手率之所以要乘以1/2是因为有的股票的权重增加了必然有股票的权重减小，二者都计算则重复
        '''
        holding_data = transholding(self._bt.weighted_holding)
        to_res = dict()
        for port_id in holding_data:
            to_res[port_id] = self._calc_groupto(holding_data[port_id])
        self.to_result = pd.DataFrame(to_res)

    def output(self):
        '''
        Return
        ------
        out: pd.DataFrame
            换手率分析结果
        '''
        return self.to_result

    @property
    def analysis_result(self):
        '''
        相关分析结果
        Return
        ------
        out: dict
            换手率分析结果
        '''
        if self._result_cache is None:
            self.analyse()
            self._result_cache = self.output()
            return self._result_cache
        else:
            return self._result_cache

    def _calc_groupto(self, holdings):
        '''
        计算单个分组（组合）的换手率
        Parameter
        ---------
        holdings: dict
            持仓权重的原始数据，格式为{time: {code: weight}}

        Return
        ------
        out: pd.Series
            长度与holdings相同，索引为time，数据为换手率
        '''
        times = sorted(holdings.keys())
        times = zip([None] + times[:-1], times)
        res = dict()
        for last_t, cur_t in times:
            last_h = holdings.get(last_t, None)
            cur_h = holdings[cur_t]
            # pdb.set_trace()
            res[cur_t] = self._calc_to(last_h, cur_h)
        return pd.Series(res)

    def _calc_to(self, last_holding, cur_holding):
        '''
        计算单个换仓时间点的换手率
        Parameter
        ---------
        last_holding: dict
            上次的持仓，dict的结构为{code: weight}或者None
        cur_holding: dict
            本次的持仓，dict的结构为{code: weight}

        Return
        ------
        out: float
            换手率
        '''
        if last_holding is None:
            last_holding = pd.Series()
        else:
            last_holding = pd.Series(last_holding)
        cur_holding = pd.Series(cur_holding)
        # 融合两次持仓的股票
        # pdb.set_trace()
        codes_merge = last_holding.index.union(cur_holding.index)
        last_holding = last_holding.reindex(codes_merge).fillna(0)
        cur_holding = cur_holding.reindex(codes_merge).fillna(0)
        return np.sum(np.abs(cur_holding - last_holding)) * 0.5


class CharacterAnalysor(Analysor):
    '''
    用于分析持仓组合的其他因子特征
    '''

    def __init__(self, bt, factors, csmethod='median', tsmethod='mean'):
        '''
        Parameter
        ---------
        bt: BackTest
            需要分析的回测实例
        factors: list like
            需要计算的其他因子特征，要求能在fmanager.list_allfactor()中找到
        csmethod: str, default median
            横截面上综合计算组合内部因子特征的方法，支持均值（mean）和中位数（median）方法
        tsmethod: str, default mean
            综合组合因子时间序列数据的方法，即在时间层面上合并单个分组综合因子特征的方法，前提是已经
            在横截面上计算过分组内部的因子特征
        '''
        super().__init__(bt)
        self._factors = factors
        self._csmethod = csmethod
        if tsmethod == 'mean':
            self._tsmethod = np.mean
        elif tsmethod == 'median':
            self._tsmethod = np.median
        else:
            raise ValueError('Invalid \'tsmethod\' parameter({mtd})'.format(mtd=tsmethod))
        self._res_cache = None

    def analyse(self):
        '''
        执行分析，即针对每个因子，先在横截面上计算每个分组因子特征值的时间序列，然后再计算时间轴上的
        综合值
        '''
        holding = transholding(self._bt.holding_result)
        factor_res = {}
        for f in self._factors:
            tmp = {}
            for port_id in holding:
                tmp[port_id] = get_group_factorcharacter(holding[port_id], f, self._csmethod)
            factor_res[f] = pd.DataFrame(tmp)
        self.single_factor_result = factor_res
        out = {}
        for f in factor_res:
            tmp = factor_res[f]
            out[f] = self._tsmethod(tmp, axis=0)
        self.all_factor_result = pd.DataFrame(out)

    def output(self):
        '''
        Return
        ------
        out: CharacterAnalysisResult(single_res->{factor: pd.DataFrame}, all_res->pd.DataFrame)
            namedtuple类型，包含结果有单因子结果和所有因子的综合结果
        '''
        self.analyse()
        CharacterAnalysisResult = namedtuple('CharacterAnalysisResult', ['single_res', 'all_res'])
        self._res_cache = CharacterAnalysisResult(single_res=self.single_factor_result,
                                                  all_res=self.all_factor_result)
        return self._res_cache

    @property
    def analysis_result(self):
        '''
        Return
        ------
        out: CharacterAnalysisResult(single_res->{factor: pd.DataFrame}, all_res->pd.DataFrame)
            namedtuple类型，包含结果有单因子结果和所有因子的综合结果
        '''
        if self._res_cache is None:
            res = self.output()
        else:
            res = self._res_cache
        return res

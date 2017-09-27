#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-09-27 11:25:33
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
因子条件测试模块，主要用来分析两个因子之间的相互作用关系
__version__ = 1.0.0
修改日期：2017-09-27
修改内容：
    初始化，添加两因子条件分组回测的一些基本功能
'''
# 系统模块
from collections import namedtuple
# 本地模块
from factortest.grouptest.backtest import FactortestTemplate
from factortest.grouptest.utils import holding2stockpool
from factortest.const import MONTHLY
from factortest.grouptest.analysis import NavAnalysor, IndustryAnalysor, TOAnalysor
from factortest.utils import MemoryDataProvider
from fmanager import query

# --------------------------------------------------------------------------------------------------
# 函数定义
# --------------------------------------------------------------------------------------------------
# 类定义
FactorTestRes = namedtuple('FactorTestRes', ['backtest', 'nav_res', 'to_res', 'ind_res'])


class ConditionalTest(object):
    '''
    使用条件因子对横截面股票划分分组，然后在每个分组内部进行对测试因子进行分组测试，从而查看测试因子
    的表现是否受条件因子影响
    '''

    def __init__(self, context_factor, test_factor, start_time, end_time, reb_type=MONTHLY,
                 context_num=5, factor_groupnum=5, ind_cls='ZX_IND', show_progress=True):
        '''
        Parameter
        ---------
        contex_factor: str
            条件因子，要求能在fmaneger.get_factor_dict()返回的结果中找到
        test_factor: str
            测试因子，要求能在fmaneger.get_factor_dict()返回的结果中找到
        start_time: datetime like
            测试的开始时间
        end_time: datetime like
            测试的结束时间
        reb_type: str, default MONTHLY
            换仓频率，目前只支持月度（MONTHLY）和周度（WEEKLY）
        context_num: int, default 5
            依据条件因子对横截面进行分类的组数
        factor_groupnum: int, default 5
            每个条件的情境下，对测试因子进行分组的组数
        ind_cls: str
            行业分类标准，要求能在fmaneger.get_factor_dict()返回的结果中找到
        show_progress: boolean, default True
            是否显示进度
        '''
        self.context_factor = context_factor
        self.test_factor = test_factor
        self._start_time = start_time
        self._end_time = end_time
        self._reb_type = reb_type
        self._context_num = context_num
        self._factor_groupnum = factor_groupnum
        self.show_progress = show_progress
        self._ind_cls = ind_cls

    def _prepare_contextualfactor(self):
        '''
        对条件因子进行回测，用于获取一些各个情境下的基准表现
        '''
        # 条件因子回测
        contextualfactor_test = FactortestTemplate(self.context_factor, self._start_time,
                                                   self._end_time, group_num=self._context_num,
                                                   reb_method=self._reb_type,
                                                   show_progress=self.show_progress)
        contextualfactor_bt = contextualfactor_test.run_test()
        # 获取条件因子各组换手率、行业分布、净值的数据
        # 净值分析数据
        benchmark = query('SSEC_CLOSE', (self._start_time, self._end_time)).iloc[:, 0]
        benchmark = benchmark / benchmark.iloc[0]
        contextf_navanalysor = NavAnalysor(contextualfactor_bt, benchmark)
        contextf_navres = contextf_navanalysor.analysis_result
        # 换手率分析数据
        contextf_toanalysor = TOAnalysor(contextualfactor_bt)
        contextf_tores = contextf_toanalysor.analysis_result
        # 行业分布分析数据
        contextf_indanalysor = IndustryAnalysor(contextualfactor_bt, self._ind_cls)
        contextf_indres = contextf_indanalysor.analysis_result

        # 存储中间数据
        self.context_factor_btres = FactorTestRes(backtest=contextualfactor_bt,
                                                  nav_res=contextf_navres,
                                                  to_res=contextf_tores,
                                                  ind_res=contextf_indres)
        self._contextbt = contextualfactor_bt

    def _context_grouptest(self, context_id):
        '''
        在给定的上下文下进行测试
        Parameter
        ---------
        context_id: int
            上下文所属的id
        '''
        # 给定情境下，对因子进行回测
        context_holding = self._contextbt.holding_result
        context_stockpool = holding2stockpool(context_holding, context_id)
        context_stockpool_provider = MemoryDataProvider(context_stockpool)
        factor_test = FactortestTemplate(self.test_factor, self._start_time, self._end_time,
                                         group_num=self._factor_groupnum, reb_method=self._reb_type,
                                         show_progress=self.show_progress,
                                         stock_pool=context_stockpool_provider)
        factor_bt = factor_test.run_test()
        # 净值分析
        benchmark = self._contextbt.navpd['group_%02d' % context_id]
        nav_analysor = NavAnalysor(factor_bt, benchmark)
        nav_res = nav_analysor.analysis_result
        # 行业分布分析
        ind_analysor = IndustryAnalysor(factor_bt, self._ind_cls)
        ind_res = ind_analysor.analysis_result
        context_inddist = self.context_factor_btres.ind_res\
            .weighted_industry_distribution_weight[context_id]
        ind_diff = dict()
        # 计算各个分组与基准之间的行业差别
        for port_id in ind_res.weighted_industry_distribution_weight:
            tmp_inddist = ind_res.weighted_industry_distribution_weight[port_id]
            diff = tmp_inddist - context_inddist
            ind_diff[port_id] = diff
        # 换手率分析
        to_analysor = TOAnalysor(factor_bt)
        to_res = to_analysor.analysis_result
        return FactorTestRes(backtest=factor_bt, nav_res=nav_res, to_res=to_res, ind_res=ind_diff)

    def run(self):
        '''
        开启分析器
        '''
        if self.show_progress:
            print('Testing Contextual Factor...')
            self._prepare_contextualfactor()
        grouptest_res = list()
        for context_id in range(self._context_num):
            if self.show_progress:
                print('Testing Under Context: {context_id}'.format(context_id=context_id))
                tmp = self._context_grouptest(context_id)
                grouptest_res.append(tmp)
        self.grouptest_result = grouptest_res

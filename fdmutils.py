#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-29 15:56:04
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
用于辅助处理基本面数据的函数库
__version__ = 1.0
修改日期：2017-04-07
修改日期：
    1. 添加说明
    2. 添加cal_season函数
'''
import datatoolkits
# import dateshandle
# import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm


def handle_combine_na(data, code_col, rpt_col, update_col, showprogress=True):
    '''
    处理fdgetter.combine后的数据
    方法为对于每一个股票的每一个报告期，使用前值填充后面的NA
    @param:
        data: 需要处理的DataFrame
        code_col: 股票代码列
        rpt_col: 报告期所在列
        update_col: 更新日期所在列
        showprogress: 是否显示程序进度，默认为True
    @return:
        处理NA值后的数据，同时还会在处理前将全部为NA值和重复列的项都删除
    '''
    # 去除数据列全部为NA的项和重复项
    data = data.dropna(subset=[rpt_col, update_col], how='any')
    data_cols = data.columns.difference([rpt_col, update_col, code_col])
    data = data.dropna(subset=data_cols, how='all')
    data = data.drop_duplicates()
    # 按照股票代码、报告期、更新日期排序
    data = data.sort_values([code_col, rpt_col, update_col])
    # 按照股票代码进行分组
    by_code = data.groupby(code_col, as_index=False)
    # NA值处理函数

    def handle_na(df):
        df = df.sort_values(update_col)
        by_rpt = df.groupby(rpt_col, as_index=False)
        res = by_rpt.apply(lambda x: x.fillna(method='ffill'))
        return res
    if showprogress:
        tqdm.pandas()
        res = by_code.progress_apply(handle_na)
    else:
        res = by_code.apply(handle_na)
    res = res.sort_values([code_col, rpt_col, update_col]).reset_index(drop=True)
    return res


def get_observabel_data(df, rpt_col='rpt_date', update_col='update_time'):
    '''
    获取每个观测日可观测到的数据
    @param:
        df: 单个股票的数据，且已经按照观测日期排序
        rpt_col: 报告期列
        update_col: 更新日期列
    @return:
        每只股票每个观测日可以观测到的数据集合，即对应每个观测日都有一个可观测到的数据时间序列与其
        对应
        可用于将数据排序和按照股票代码分组后，进行apply
    '''
    res = pd.DataFrame()
    for udt in df[update_col].unique():
        data = df[df[update_col] <= udt]     # 观测日的数据需要包含当天的
        by_rpt = data.groupby(rpt_col, as_index=False)
        tmp_res = by_rpt.apply(lambda x: x.tail(1))
        tmp_res['obs_time'] = [udt] * len(tmp_res)
        res = res.append(tmp_res)
    return res


def cal_ttm(df, col_name, nperiod=4, rename=None):
    '''
    计算ttm
    @param:
        df: 单个股票单个观测期的数据
        col_name：列名，可以为列表，即多个列一起计算；也可以为列字符串
        nperiod: 回溯期，默认为4，即TTM
        rename: 重命名，默认为None，需要提供字典形式数据，格式为{old_name: new_name, ...}
    @return:
        如果数据量不够，返回空值；反之，则返回求和后的值。返回值均为Series格式
    '''
    # 将其转换为列表
    if isinstance(col_name, str):
        col_name = [col_name]
    # 得到df格式的data
    data = df.loc[:, col_name]
    res = data.tail(nperiod)
    if len(res) < nperiod:
        res = datatoolkits.gen_series(col_name)
    else:
        res = res.sum(axis=0)   # sum会忽视NA值
    if rename is not None:
        res = res.rename(rename)
    return res


def cal_yr(df, col_name, rpt_col='rpt_date', offset=1, rename=None):
    '''
    计算最近年度的数据，即当前观察日往前数offset个年报发布日的数据，例如，当前观察日能够
    看到2013-12-31、2012-12-31、2011-12-31的数据，offset=1表明使用2013-12-31的数据，offset=2
    表明2012-12-31的数据，以此类推
    @param:
        df: 单个股票的单个观察日数据，要求按照报告期排序，DataFrame格式
        col_name：列名，可以为列表，即多个列一起计算；也可以为列字符串
        rpt_col: 报告期所在列
        offset: 往前推的年度数
        rename: 重命名，默认为None，需要提供字典形式数据，格式为{old_name: new_name, ...}
    '''
    if isinstance(col_name, str):
        col_name = [col_name]
    data = df.loc[df[rpt_col].map(lambda x: x.strftime('%m-%d') == '12-31'), col_name]
    res = data.tail(offset)
    if len(res) < offset:
        res = datatoolkits.gen_series(col_name)
    else:
        res = res.head(1).iloc[0]   # 将df转换为Series
    if rename is not None:
        res = res.rename(rename)
    return res


def cal_season(df, col_name, rpt_col='rpt_date', offset, rename=None):
    '''
    计算最近季度的数据，即当前观察日往前推offset个季报发布日的数据
    例如，当前观察日能看到2013-12-31、2013-09-30、2013-06-30、2013-03-31的数据，offset=1表明使用
    2013-12-31的数据，offset=2表明使用2013-09-30的数据，以此类推
    @param:
        df: 单个股票数据的单个观测日数据，要求按照报告期进行排序，DataFrame格式
        col_name: 列名，可以为列表，即多个列一起计算，也可以为字符串，即单个列
        rpt_col: 报告期列名
        offset: 往前推的季度数
        rename: 重命名，默认为None，即不进行重命名，需要使用时，提供的格式为{old_name: new_name,...}
    '''
    if isinstance(col_name, str):
        col_name = [col_name]
    data = df.loc[:, col_name]
    res = data.tail(offset)
    if len(res) < offset:
        res = datatoolkits.gen_series(col_name)
    else:
        res = res.head(1).iloc[0]   # 将df转换为Series
    if rename is not None:
        res = res.rename(rename)
    return res

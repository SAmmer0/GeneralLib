#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-13 14:01:49
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

from collections import defaultdict
import pdb
from functools import wraps

import pandas as pd
import numpy as np

import dateshandle
import datatoolkits
from fmanager.const import UNIVERSE_FILE_PATH
'''
提供因子计算的一些基本工具
'''
# --------------------------------------------------------------------------------------------------
# 常量

ZXIND_TRANS_DICT = {'银行': 'bank', '房地产': 'realestate', '计算机': 'computer',
                    '医药': 'med', '餐饮旅游': 'food&tour', '有色金属': 'metal',
                    '商贸零售': 'retail', '交通运输': 'transportation', '机械': 'machine',
                    '综合': 'comprehensive', '电子元器件': 'electronic component',
                    '建筑': 'building', '建材': 'building materials', '家电': 'home appliances',
                    '纺织服装': 'textiles', '食品饮料': 'food&drink', '石油石化': 'petroche',
                    '汽车': 'auto', '轻工制造': 'manufacturing',
                    '电力及公用事业': 'electricity&utility', '通信': 'communication',
                    '农林牧渔': 'agriculture', '电力设备': 'power equipment',
                    '基础化工': 'chemical industry', '传媒': 'media', '煤炭': 'coal',
                    '非银行金融': 'non-bank finance', '钢铁': 'steel', '国防军工': 'war industry'}

# --------------------------------------------------------------------------------------------------
# 类定义


class Factor(object):
    '''
    用于记录一个因子的基本信息，包含因子的名称、因子的计算方法、因子添加的时间、计算依赖的因子、
    因子的相关描述说明、因子数据类型
    '''

    def __init__(self, name, calc_method, addtime, dependency=None, desc=None, data_type='f8'):
        '''
        Parameter
        ---------
        name: str
            因子的名称，要求要在所有因子中具有唯一性
        calc_method: function
            因子的计算方法，要求function的格式为func(universe, start_time, end_time)，其中universe为
            股票代码列表，start_time和end_time分别为计算的起始时间，要求为可被pd.to_datetime转化的格式，
            而且func返回的结果为pd.DataFrame，index为日期时间，columns为股票代码，数据的时间要包含
            start_time和end_time
        addtime: datetime or the like
            因子的添加时间，要求为静态时间
        dependency: list like, default None
            该因子的依赖因子，要求为列表，列表内容为字符串（因子名），如果为None，表示没有依赖项
        desc: str, default None
            因子相关描述，默认为None表示没有相关介绍
        data_type: str, default f8
            表示因子的数据格式，目前只支持f和s开头的格式描述，数字型数据默认即可，表示64位浮点数，
            字符串型数据以S开头，后面跟上最大的字符串长度（也可分配更多空间，供后续扩展）
        '''
        self.name = name
        self.calc_method = calc_method
        self.addtime = addtime
        self.dependency = dependency
        self.desc = desc
        self.data_type = data_type

    def __str__(self):
        data = {'name': self.name, 'dep': self.dependency, 'desc': self.desc,
                'at': self.addtime.strftime('%Y-%m-%d')}
        res = 'Factor(name={name}, dependency={dep}, describe={desc}, addtime={at})'.format(**data)
        return res

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.name == other.name
# --------------------------------------------------------------------------------------------------
# 函数


def check_indexorder(df):
    '''
    检查数据的时间顺序是否正确（升序）

    Parameter
    ---------
    df: pd.DataFrame
        需要检查的数据

    Return
    ------
    out: boolean
        如果顺序正确，返回True，反之返回False
    '''
    index = df.index.tolist()
    return index == sorted(index)


def check_duplicate_factorname(factor_list, mod):
    '''
    检测重复的因子名称，用于各个因子模块中

    Parameter
    ---------
    factor_list: list like
        因子模块中的因子列表
    mod: str
        因子模块的名称
    '''
    factor_dict = defaultdict(list)
    for factor in factor_list:
        factor_dict[factor.name].append(factor)
    for name in factor_dict:
        assert len(factor_dict[name]) <= 1, \
            'Error, factor name("{name}") is duplicated in module "{mod_name}"'.format(name=name,
                                                                                       mod_name=mod)


def convert_data(dfs, indices):
    '''
    将多维度的数据结合到一张表中，结合的方法为：在每个df的index中添加一个维度，填充对应的indices，然后将
    所有的数据在0轴的方向通过append结合在一起

    Parameter
    ---------
    dfs: list like
        需要通过该方法结合的数据，目前仅支持index为时间，columns为股票代码的数据（不做检测），要求
        列表中数据的长度都相同（即包含相同的时间长度，否则会造成一些日期内没有特定的数据），目前暂时
        不支持依照列进行结合，因为股票代码的数量会变动，容易出现长度的问题
    indices: list like
        每个数据对应的标记，即对应的index

    Return
    ------
    out: pd.DataFrame
        index为MultiIndex类型，最后一个维度为数据标记维度，columns为股票代码，顺序与第一个dfs的第一个
        df的columns相同
    '''
    if not hasattr(indices, '__len__'):
        indices = list(indices)
    assert len(dfs) == len(indices), \
        'Error, DataFrame list({df_len}) should have the same length as indices({idx_len})'.\
        format(df_len=len(dfs), idx_len=len(indices))
    _df_shape = [df.shape for df in dfs]
    assert all(ldf == _df_shape[0] for ldf in _df_shape), \
        'Error, input data should have the same shape!'
    out = pd.DataFrame()
    for idx, df in zip(indices, dfs):
        df = df.copy()
        df.index = pd.MultiIndex.from_product([df.index, [idx]])
        out = out.append(df)
        # pdb.set_trace()
    return out


def checkdata_completeness(data, start_time, end_time):
    '''
    检查数据的完整性，保证数据的长度与期间交易日的长度相同

    Parameter
    ---------
    data: pd.DataFrame
        需要检测完整性的数据
    start_time: str or datetime or other type that can be converted by pd.to_datetime
        起始时间
    end_time: str or datetime or other type that can be converted by pd.to_datetime
        终止时间
    '''
    tds = dateshandle.get_tds(start_time, end_time)
    return len(data) == len(tds)


def get_universe(path=UNIVERSE_FILE_PATH):
    '''
    用于获取当前数据中对应的universe
    Parameter
    ---------
    path: str, default UNIVERSE_FILE_PATH
        universe文件存储的位置

    Return
    ------
    out: list
        当前数据对应的universe（排序后）
    '''
    universe = datatoolkits.load_pickle(path)[0]
    return sorted(universe)


def get_valid_mask(start_time, end_time):
    '''
    获取给定期间内（包含起始时间）数据是否有效的掩码。

    Parameter
    ---------
    start_time: datetime like
        开始时间
    end_time: datetime like
        结束时间

    Return
    ------
    out: pd.DataFrame
        index为时间，columns为股票代码

    Notes
    -----
    数据是否有效是根据当前股票是否退市或者终止上市来判断的，凡是LIST_STATUS为退市或者终止上市（3和4）
    状态的股票均被视作为无效数据，即False
    '''
    from fmanager.factors.query import query
    ls_status = query('LIST_STATUS', (start_time, end_time))
    valid_mask = np.logical_or(ls_status == 1, ls_status == 2)
    return valid_mask


def drop_delist_data(func):
    '''
    装饰器函数，用于装饰获取数据的函数，将退市股票的相关数据设置为NA

    Parameter
    ---------
    func: function(universe, start_time, end_time)
        获取数据的函数

    Return
    ------
    out_func: function(universe, start_time, end_time)
        添加退市数据处理的函数
    '''
    @wraps(func)
    def inner(universe, start_time, end_time):
        data = func(universe, start_time, end_time)
        mask = get_valid_mask(start_time, end_time)
        data[~mask] = np.nan
        return data
    return inner

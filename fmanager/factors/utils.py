#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-13 14:01:49
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

from collections import defaultdict
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
            股票代码列表，start_time和end_time分别为计算的起始时间，要求为可被pd.to_datetime转化的格式
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
        return self.name

    def __repr__(self):
        return self.name

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

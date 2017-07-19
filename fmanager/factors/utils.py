#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-13 14:01:49
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$


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
    因子的相关描述说明
    '''

    def __init__(self, name, calc_method, addtime, dependency=None, desc=None):
        '''
        Parameter
        ---------
        name: str
            因子的名称，要求要在所有因子中具有唯一性
        '''
        self.name = name
        self.calc_method = calc_method
        self.addtime = addtime
        self.dependency = dependency
        self.desc = desc

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

# --------------------------------------------------------------------------------------------------
# 函数

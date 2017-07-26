#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-25 09:27:19
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
其他未分类的基本因子

__version__ = 1.0.0
'''

# --------------------------------------------------------------------------------------------------
# 常量和功能函数
NAME = 'other'


def get_factor_dict():
    res = dict()
    for f in factor_list:
        res[f.name] = {'rel_path': NAME + '\\' + f.name, 'factor': f}
    return res


factor_list = []

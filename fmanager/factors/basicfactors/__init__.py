#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-13 14:45:10
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
from . import financialdata
from . import general
from . import other
from . import quote
from . import test

NAME = 'basicfactors'
FACTOR_MODULES = [financialdata, general, quote, other]


def get_factor_dict():
    '''
    获取模块内部的因子字典
    '''
    factors = dict()
    for mod in FACTOR_MODULES:
        factors.update(mod.get_factor_dict())
    for f in factors:
        rel_path = NAME + '\\' + factors[f]['rel_path']
        factors[f]['rel_path'] = rel_path
    return factors

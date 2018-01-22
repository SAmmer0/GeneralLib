#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-24 14:47:56
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
因子字典模块，用于从各个因子模块中读取因子，并构造因子字典，写入文件中
__version__ = 1.0.1
修改日期：2017-09-07
修改内容：
    添加list_allfactor和get_factor_detail函数
'''

from copy import deepcopy
import pdb
from datatoolkits import dump_pickle, load_pickle
from fmanager.factors import basicfactors
from fmanager.factors import derivativefactors
from fmanager.factors import barra
from fmanager.const import FACTOR_FILE_PATH, SUFFIX, FACTOR_DICT_FILE_PATH


# --------------------------------------------------------------------------------------------------
# 常量
FACTOR_MODULES = [derivativefactors, basicfactors, barra]


# --------------------------------------------------------------------------------------------------
# 函数


def get_factor_dict():
    '''
    获取模块内部的因子字典
    因子字典结构为{factor_name: {'factor': factor, 'rel_path': relative path, 'abs_path': absolute path}}
    '''
    factors = dict()
    factor_names = set()
    for mod in FACTOR_MODULES:
        mod_dict = mod.get_factor_dict()    # 检查是否有重复因子名称
        assert len(factor_names.intersection(mod_dict.keys())) == 0, \
            'Error, duplicate factor name in module "{mod_name}"'.format(mod_name=mod.__name__)
        factors.update(mod.get_factor_dict())
        factor_names.update(mod_dict.keys())
    factors = add_abs_path(factors)
    # pdb.set_trace()
    return factors


def list_allfactor():
    '''
    列出当前因子库中的所有因子名称
    Return
    ------
    out: list
        所有因子的名称
    '''
    fd = get_factor_dict()
    return sorted(fd.keys())


def get_factor_detail(factor_name):
    '''
    获取某个因子的详细信息
    Parameter
    ---------
    factor_name: str
        需要查询的因子的名称

    Return
    ------
    out: dict
        因子相关的信息，结构为{'factor': factor, 'rel_path': relative path,
        'abs_path': absolute path}
    '''
    fd = get_factor_dict()
    assert factor_name in fd, "Error, {fn} NOT FOUND!".format(fn=factor_name)
    return fd[factor_name]


def add_abs_path(factor_dict):
    '''
    在因子字典中加入绝对路径

    Parameter
    ---------
    factor_dict: dict
        因子字典

    Return
    ------
    res: dict
        加入绝对路径的因子字典

    Notes
    -----
    该函数会在因子字典中加入新的abs_path值，用于记录因子的绝对路径
    '''
    res = deepcopy(factor_dict)
    for factor in res:
        res[factor]['abs_path'] = FACTOR_FILE_PATH + '\\' + res[factor]['rel_path'] + SUFFIX
    return res


def gen_path_dict(fd):
    '''
    将因子字典转化为{因子名称: 因子绝对路径}字典

    Parameter
    ---------
    fd: dict
        原因子字典

    Return
    ------
    out: dict
        绝对路径字典
    '''
    res = {fd[name]['factor'].name: fd[name]['abs_path'] for name in fd}
    return res


def update_factordict(path=FACTOR_DICT_FILE_PATH):
    '''
    自动更新因子字典的数据，并将其写入文件中
    '''
    all_factor = get_factor_dict()
    factor_dict = gen_path_dict(all_factor)
    dump_pickle(factor_dict, path)


def check_dict(path=FACTOR_DICT_FILE_PATH):
    '''
    检查数据字典文件是否与当前模块中的因子字典相同，如果不同，则更新数据字典文件
    '''
    try:
        file_dict = load_pickle(FACTOR_DICT_FILE_PATH)
    except FileNotFoundError:   # 如果不存在数据字典文件，则生成文件
        print('Dictionary file not found, initialization...')
        update_factordict()
        return
    module_dict = gen_path_dict(get_factor_dict())
    file_dict = load_pickle(FACTOR_DICT_FILE_PATH)
    if module_dict != file_dict:
        print('Updating dictionary file...')
        update_factordict()

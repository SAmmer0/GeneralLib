#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-14 14:20:56
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

from . import general
import datatoolkits
import sysconfiglee


def getls_test():
    upath = sysconfiglee.get_config('universe_file_path')
    uload = datatoolkits.DataLoader('HDF', upath, key='A_universe')
    universe = uload.load_data().code.tolist()
    start_time = '2015-01-01'
    end_time = '2017-01-01'
    res = general.get_liststatus(universe, start_time, end_time)
    return res


def getzxind_test():
    upath = sysconfiglee.get_config('universe_file_path')
    uloader = datatoolkits.DataLoader('HDF', upath, key='A_universe')
    universe = uloader.load_data()
    res = general.get_zxind(universe.code.tolist(), '2016-01-01', '2017-01-01')
    return res


def getst_test():
    upath = sysconfiglee.get_config('universe_file_path')
    uload = datatoolkits.DataLoader('HDF', upath, key='A_universe')
    universe = uload.load_data().code.tolist()
    start_time = '2015-01-01'
    end_time = '2017-01-01'
    res = general.get_st(universe, start_time, end_time)
    return res


def test(func):
    upath = sysconfiglee.get_config('universe_file_path')
    uload = datatoolkits.DataLoader('HDF', upath, key='A_universe')
    universe = uload.load_data().code.tolist()
    start_time = '2015-01-01'
    end_time = '2017-01-01'
    res = func(universe, start_time, end_time)
    return res

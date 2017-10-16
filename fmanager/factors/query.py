#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-18 14:47:58
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
因子查询模块

__version__ = 1.0.0
修改日期：2017-07-19
修改内容：
    添加add_abs_path

修改日期：2017-07-27
修改内容：
    给query函数添加fillna参数选项
'''
__version__ = '1.0.0'
# 第三方库
import numpy as np
# 本地库
from datatoolkits import load_pickle
from fmanager.const import FACTOR_DICT_FILE_PATH
from fmanager import database
from fmanager.factors.utils import get_universe

# --------------------------------------------------------------------------------------------------
# 常量

# --------------------------------------------------------------------------------------------------
# 函数


def query(factor_name, time, codes=None, fillna=None):
    '''
    接受外部的请求，从数据库中获取对应因子的数据

    Parameter
    ---------
    factor_name: str
        需要查询的因子名称
    time: type that can be converted by pd.to_datetime or tuple of that
        单一的参数表示查询横截面的数据，元组（start_time, end_time）表示查询时间序列数据
    codes: list, default None
        需要查询数据的股票代码，默认为None，表示查询所有股票的数据
    fillna: int or float, default（该参数将废止）
        是否给NA值进行填充，默认为None，即不需要填充，如果需要填充则将填充值传给fillna参数
    Return
    ------
    out: pd.DataFrame
        查询结果数据，index为时间，columns为股票代码，如果未查询到符合要求的数据，则返回None
    '''
    factor_dict = load_pickle(FACTOR_DICT_FILE_PATH)
    if factor_dict is None:
        raise ValueError('Dictionary file needs initialization...')
    assert factor_name in factor_dict, \
        'Error, factor name "{pname}" is'.format(pname=factor_name) +\
        ' not valid, valid names are {vnames}'.format(vnames=list(factor_dict.keys()))
    abs_path = factor_dict[factor_name]
    db = database.DBConnector(abs_path)
    data = db.query(time, codes)
    universe = get_universe()
    if codes is None:   # 为了避免数据的universe不一致导致不同数据的横截面长度不同
        data = data.reindex(columns=universe)
    if fillna is None:
        fillna = db.default_data
        if isinstance(fillna, np.bytes_):
            fillna = fillna.decode('utf8')
    data = data.fillna(fillna)
    return data

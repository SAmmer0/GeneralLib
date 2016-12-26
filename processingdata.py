#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-12-26 10:27:26
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
本程序用于对一些数据进行常用的预处理
__version__ = 1.0

修改日期：2016年12月26日
修改内容：
    添加基本函数gen_pdcolumns
'''

import pandas as pd


def gen_pdcolumns(data, operations):
    '''
    用于在data中添加新的列
    @param:
        data: 原始数据，要求为pd.DataFrame的格式
        operations: 需要添加的列，格式为{'colName': (func, {parameter dict})}或者{'colName': func}
            其中，func表示需要对数据进行处理的函数，要求函数只能调用每一行的数据，返回一个结果，且func
            的第一个参数为行数据，其他参数通过key=value的形式调用
    @return:
        res: 返回一个新的数据集，因为修改不是在原数据上进行的，因此需要采用以下方式才能将结果影响到原
            数据：data = gen_pdcolumns(data, operations)
    '''
    assert isinstance(data, pd.DataFrame), ('Parameter \"data\"" type error! Request ' +
                                            'pd.DataFrame, given %s' % str(type(data)))
    assert isinstance(operations, dict), ('Parameter \"operations\" type error! Request dict,' +
                                          'given %s' % str(type(operations)))
    res = data.copy()
    for col in operations:
        assert isinstance(col, str), 'Column name should be str, given %s' % str(type(col))
        operation = operations[col]
        if hasattr(operation, '__len__'):   # 表明为列表或其他列表类的类型
            assert len(operation) == 2, ('Operation paramter error! Request formula should like' +
                                         '(func, {parameter dict})')
            func = operation[0]
            params = operation[1]
            res[col] = res.apply(lambda x: func(x, **params), axis=1)
        else:
            res[col] = res.apply(func, axis=1)
    return res

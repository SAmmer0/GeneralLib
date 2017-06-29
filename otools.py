#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-05-22 15:09:24
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
该模块主要包含一些其他需要使用却不好分类的工具函数
__version__ = 1.0.0
修改日期：2017-05-22
修改内容：
    初始化，添加name_wrapper
'''


def name_wrapper(func):
    '''
    装饰器，用于打印当前正在执行的函数名称
    Parameter
    ---------
    func: function
        需要被打印的函数

    Return
    ------
    out: function
        修改后的函数
    '''
    def inner(*args, **kwargs):
        print('Procedure {name}'.format(name=func.__name__))
        res = func(*args, **kwargs)
        return res
    return inner

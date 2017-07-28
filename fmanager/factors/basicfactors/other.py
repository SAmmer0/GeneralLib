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
import datatoolkits
import fdgetter
import pandas as pd
import pdb

from ..utils import Factor, check_duplicate_factorname, check_indexorder
# --------------------------------------------------------------------------------------------------
# 常量和功能函数
NAME = 'other'


def get_factor_dict():
    res = dict()
    for f in factor_list:
        res[f.name] = {'rel_path': NAME + '\\' + f.name, 'factor': f}
    return res

# --------------------------------------------------------------------------------------------------
# 一致预期目标价因子


def get_ctargetprice(universe, start_time, end_time):
    '''
    获取一致预期目标价数据
    '''
    sql = '''
    SELECT TARGET_PRICE, CON_DATE, STOCK_CODE
    FROM CON_FORECAST_SCHEDULE
    WHERE
        CON_DATE >= CAST(\'{start_time}\' AS datetime) AND
        CON_DATE <= CAST(\'{end_time}\' AS datetime)
    '''
    data = fdgetter.get_db_data(sql, start_time=start_time, end_time=end_time,
                                cols=('data', 'time', 'code'), db=fdgetter.zyyx,
                                add_stockcode=False)
    # pdb.set_trace()
    data['code'] = data.code.apply(datatoolkits.add_suffix)
    data = data.pivot_table('data', index='time', columns='code')
    data = data.loc[:, sorted(universe)]
    assert check_indexorder(data), 'Error, data order is mixed!'
    return data


target_price = Factor('TARGET_PRICE', get_ctargetprice, pd.to_datetime('2017-07-28'),
                      desc='一致预期目标价')
# --------------------------------------------------------------------------------------------------

factor_list = []
check_duplicate_factorname(factor_list, __name__)

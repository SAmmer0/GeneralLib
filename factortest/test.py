#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-08-21 10:49:36
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

import datetime as dt

from dateshandle import get_tds
from .utils import HDFDataProvider

# factortest/utils.py test case
DATA_START_TIME = '2015-01-01'
DATA_END_TIME = dt.datetime.now()
SEARCH_START_TIME = '2017-01-01'
SEARCH_END_TIME = '2017-08-01'
CODE = '002230.SZ'
TDS_NUM = len(get_tds(SEARCH_START_TIME, SEARCH_END_TIME))
data_quote = HDFDataProvider(r'E:\factordata\basicfactors\quote\CLOSE.h5', DATA_START_TIME, DATA_END_TIME)
cs_test = data_quote.get_csdata(SEARCH_END_TIME)
panel_test = data_quote.get_paneldata(SEARCH_START_TIME, SEARCH_END_TIME)
ts_test = data_quote.get_tsdata(SEARCH_START_TIME, SEARCH_END_TIME, CODE)
p_data = data_quote.get_data(SEARCH_END_TIME, CODE)

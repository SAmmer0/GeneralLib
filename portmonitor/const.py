#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-10-20 14:37:02
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
# 系统模块
from os.path import dirname

# 本地模块
import sysconfiglee
import portmonitor
from factortest.const import WEEKLY, MONTHLY

# 组合的资金，持仓等明细数据所在的文件
PORT_DATA_PATH = sysconfiglee.get_config('portfolio_data_path')
# 组合的配置所在的文件，这个文件也可以不放在模块下，但是必须能够被Python解释器找到
PORT_CONFIG_PATH = dirname(portmonitor.__file__) + '\\portfolios'

# 组合的类型，包含做多、做空和多空
LONG = 'long'
SHORT = 'short'

# 组合中现金的标志
CASH = 'CASH'

# 报告频率
DAILY = 'daily'

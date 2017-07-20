#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-18 16:07:13
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

import sysconfiglee

FACTOR_FILE_PATH = sysconfiglee.get_config('factor_file_path')  # 因子存储的主目录
SUFFIX = '.h5'  # 因子文件后缀
START_TIME = '2007-01-01'   # 因子最早可追溯的时间
UNIVERSE_FILE_PATH = FACTOR_FILE_PATH + '\\' + 'universe.pickle'

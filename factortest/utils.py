#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-08-20 16:27:39
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
提供一些基础性的工具
__version__ = 1.0.0
修改日期：2017-08-20
修改内容：
    初始化，添加数据提供器类
'''
# 标准库
from abc import abstractmethod, ABCMeta
# 第三方库
import pandas as pd
# 本地库
from fmanager.database import DBConnector
# --------------------------------------------------------------------------------------------------
# 常量定义
QUOTE_BEGIN_TIME = pd.to_datetime('1990-01-01')     # 定义最早的有数据的时间
QUOTE_END_TIME = pd.to_datetime('2100-01-01')       # 定义最晚的有数据的时间
# --------------------------------------------------------------------------------------------------
# 数据提供器类


class DataProvider(object, metaclass=ABCMeta):
    '''
    数据提供器基类，主要定义相关接口
    数据提供器是数据接入的通用接口，用于处理提取相关数据的工作
    '''

    def __init__(self, data_path=None, start_time=None, end_time=None):
        '''
        Parameter
        ---------
        data_path: str, default None
            数据存储文件的路径
        start_time: types that are compatible to datetime, default None
            数据加载的起始时间
        end_time: types that are compatible to datetime, default None
            数据加载的终止时间

        Notes
        -----
        加载的数据包含start_time和end_time时点的数据（如果有）
        start_time为None，表示数据的起始时间为数据文件中数据的开始时间
        end_time为None，表示数据的终止时间为数据文件中的最新数据时间
        '''
        self._data_path = data_path
        if start_time is not None:
            self._start_time = pd.to_datetime(start_time)
        else:
            self._start_time = QUOTE_BEGIN_TIME
        if end_time is not None:
            self._end_time = pd.to_datetime(end_time)
        else:
            self._end_time = QUOTE_END_TIME
        self.loaded = False

    @abstractmethod
    def get_csdata(self, date):
        '''
        获取横截面数据

        Parameter
        ---------
        date: types that are compatible to datetime
            需要获取的数据的时间
        '''
        pass

    @abstractmethod
    def get_paneldata(self, start_date, end_date):
        '''
        获取面板数据

        Parameter
        ---------
        start_date: types that are compatible to datetime
            面板的起始日期
        end_date: types that are compatible to datetime
            面板的终止日期

        Notes
        -----
        结果会含起始和终止日期这两天的数据（如果这两天有数据）
        '''
        pass

    @abstractmethod
    def get_tsdata(self, start_date, end_date, item):
        '''
        获取某一个项目的时间序列数据

        Parameter
        ---------
        start_date: types that are compatible to datetime
            时间序列的起始日期
        end_date: types that are compatible to datetime
            时间序列的终止日期
        item: str
            需要获取数据的项目的名称

        Notes
        -----
        结果会含起始和终止日期这两天的数据（如果这两天有数据）
        '''
        pass

    @abstractmethod
    def get_data(self, date, item):
        '''
        获取某一个项目的时点的数据

        Parameter
        ---------
        date: types that are compatible to datetime
            数据的时间点
        item: str
            数据的项目名称
        '''
        pass

    @abstractmethod
    def load_data(self):
        '''
        从数据文件中加载数据
        '''
        pass


class HDFDataProvider(DataProvider):
    '''
    从fmanager中定义的HDF文件中加载数据
    '''

    def __init__(self, data_path, start_date=None, end_date=None):
        '''
        Parameter
        ---------
        data_path: str
            数据文件的路径
        start_date: types that are compatible to datetime, default None
            加载数据的起始时间
        end_date: types that are compatible to datetime, default None
            加载数据的终止时间

        Notes
        -----
        加载的数据包含start_date和end_date（如果有数据）
        '''
        super().__init__(data_path, start_date, end_date)

    def load_data(self):
        '''
        从数据文件中加载数据
        '''
        self._db = DBConnector(self._data_path)
        self.loaded = True

    def get_csdata(self, date):
        '''
        获取横截面数据

        Parameter
        ---------
        date: types that are compatible to datetime
            需要获取的数据的时间
        
        Return
        ------
        out: pd.Series
            横截面数据，index为股票代码，name为时间
        '''
        if not self.loaded:
            self.load_data()
        date = pd.to_datetime(date)
        data = self._db.query(date)
        if data is None:
            return None
        data = data.iloc[0]
        data.name = date.strftime('%Y-%m-%d')
        return data
    
    def get_paneldata(self, start_date, end_date):
        '''
        获取面板数据

        Parameter
        ---------
        start_date: types that are compatible to datetime
            面板的起始日期
        end_date: types that are compatible to datetime
            面板的终止日期
            
        Return
        ------
        out: pd.DataFrame
            面板数据，index为时间，columns为股票代码

        Notes
        -----
                结果会含起始和终止日期这两天的数据（如果这两天有数据）
        '''
        if not self.loaded:
            self.load_data()
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        data = self._db.query((start_date, end_date))
        if data is None:
            return None
        return data
    
    def get_tsdata(self, start_time, end_time, item):
        '''
        获取某一个项目的时间序列数据

        Parameter
        ---------
        start_date: types that are compatible to datetime
            时间序列的起始日期
        end_date: types that are compatible to datetime
            时间序列的终止日期
        item: str
            需要获取数据的项目的名称
        
        Return
        ------
        out: pd.Series
            时间序列数据，index为时间，name为item的名称
        Notes
        -----
        结果会含起始和终止日期这两天的数据（如果这两天有数据）
        '''
        if not self.loaded:
            self.load_data()
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        data = self._db.query((start_time, end_time), [item])
        if data is None:
            return None
        data = data.loc[:, item]    # 将结果转换为pd.Series
        data.name = item
        return data

    def get_data(self, date, item):
        '''
        获取某一个项目的时点的数据

        Parameter
        ---------
        date: types that are compatible to datetime
            数据的时间点
        item: str
            数据的项目名称
        
        Return
        ------
        out: float or str or other types
        '''
        if not self.loaded:
            self.load_data()
        date = pd.to_datetime(date)
        data = self._db.query(date, [item])
        if data is None:
            return None
        data = data.iloc[0, 0]
        return data

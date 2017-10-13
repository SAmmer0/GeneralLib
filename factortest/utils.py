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
import numpy as np
# 本地库
from fmanager.database import DBConnector
from fmanager import get_universe
from factortest.const import MONTHLY, WEEKLY
from dateshandle import get_tds
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

    @abstractmethod
    def copy(self):
        '''
        拷贝，返回没有数据缓存的新的DataProvider对象
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
        if not self.loaded:
            self._db = DBConnector(self._data_path)
            _data = self._db.query((self._start_time, self._end_time))
            # 避免universe的冲突
            universe = get_universe()
            default_data = self._db.default_data
            if isinstance(default_data, np.bytes_):
                default_data = default_data.decode('utf8')
            self._data = _data.reindex(columns=sorted(universe)).fillna(default_data)
            if self._data is None:
                self.loaded = False
            else:
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
        self.load_data()
        if self.loaded:
            date = pd.to_datetime(date)
            data = self._data.loc[date]
            data.name = date.strftime('%Y-%m-%d')
            return data
        else:
            return None

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
            面板数据，index为时间，columns为股票代码，如果没有相应的数据，返回None

        Notes
        -----
                结果会含起始和终止日期这两天的数据（如果这两天有数据）
        '''
        self.load_data()
        if self.loaded:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            mask = (self._data.index >= start_date) & (self._data.index <= end_date)
            data = self._data.loc[mask]
            return data
        else:
            return None

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

        Return
        ------
        out: pd.Series
            时间序列数据，index为时间，name为item的名称
        Notes
        -----
        结果会含起始和终止日期这两天的数据（如果这两天有数据）
        '''
        self.load_data()
        if self.loaded:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            mask = (self._data.index >= start_date) & (self._data.index <= end_date)
            data = self._data.loc[mask, item]    # 将结果转换为pd.Series
            data.name = item
            return data
        else:
            return None

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
        self.load_data()
        if self.loaded:
            date = pd.to_datetime(date)
            data = self._data.loc[date, item]
            return data
        else:
            return None

    def copy(self):
        '''
        拷贝，返回的对象数据未加载
        '''
        return HDFDataProvider(self._data_path, self._start_time, self._end_time)


class NoneDataProvider(DataProvider):
    '''
    用于对于所有的数据请求都返回None值
    使用情景主要在没有股票池或者行业限制等情况下
    '''

    def __init__(self):
        super().__init__()

    def load_data(self):
        self.loaded = True

    def get_csdata(self, date):
        return None

    def get_paneldata(self, start_date, end_date):
        return None

    def get_tsdata(self, start_date, end_date, item):
        return None

    def get_data(self, date, item):
        return None

    def copy(self):
        return NoneDataProvider()


class MemoryDataProvider(DataProvider):
    '''
    使用内存数据作为数据源
    要求数据的格式为一个pd.DataFrame的二维表，index为时间，column为股票代码
    '''

    def __init__(self, data_source):
        '''
        Parameter
        ---------
        data_source: pd.DataFrame
            内存数据的数据源

        Notes
        -----
        内部实现中使用的是数据源拷贝，但是原则上依然净值对源数据进行修改的操作
        '''
        super().__init__()
        self._data = data_source.copy()
        self.load_data()
        self._start_time = data_source.index.min()
        self._end_time = data_source.index.max()

    def load_data(self):
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
        date = pd.to_datetime(date)
        return self._data.loc[date]

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
        date = pd.to_datetime(date)
        return self._data.loc[date, item]

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
            面板数据，index为时间，columns为股票代码，如果没有相应的数据，返回None

        Notes
        -----
        结果会含起始和终止日期这两天的数据（如果这两天有数据）
        '''
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        mask = (self._data.index >= start_date) & (self._data.index <= end_date)
        return self._data.loc[mask]

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

        Return
        ------
        out: pd.Series
            时间序列数据的index为时间，name为item的名称
        Notes
        -----
        结果会含起始和终止日期这两天的数据（如果这两天有数据）
        '''
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        mask = (self._data.index >= start_date) & (self._data.index <= end_date)
        return self._data.loc[mask, item]

    @property
    def start_time(self):
        '''
        Return
        ------
        out: datetime like
            数据的开始时间
        '''
        return self._start_time

    @property
    def end_time(self):
        '''
        Return
        ------
        out: datetime like
            数据的结束时间
        '''
        return self._end_time

    def copy(self):
        '''
        Return
        ------
        out: MemoryDataProvider
            当前实例的拷贝
        '''
        return MemoryDataProvider(self._data)


class RebCalcu(object, metaclass=ABCMeta):
    '''
    用于计算换仓日的类
    '''

    def __init__(self, start_date, end_date):
        '''
        Parameter
        ---------
        start_date: datetime or other compatible types
            整体时间区间的起始时间
        end_date: datetime or other compatible types
            整体时间的终止时间
        '''
        self._start_time = pd.to_datetime(start_date)
        self._end_time = pd.to_datetime(end_date)
        self._rebdates = None

    @abstractmethod
    def _calc_rebdates(self):
        '''
        用于计算给定的时间区间内的再平衡日（指因子计算日，且类型为datetime），并将其存储在_rebdates中
        '''
        pass

    def __call__(self, date):
        '''
        判断给定的日期是否为再平衡日

        Parameter
        ---------
        date: datetime or other compatible types
        '''
        if self._rebdates is None:
            self._calc_rebdates()
        date = pd.to_datetime(date)
        return date in self._rebdates

    @property
    def reb_points(self):
        '''
        返回所有的换仓日，且换仓日按照升序排列
        Return
        ------
        out: list
            所有的换仓日
        '''
        if self._rebdates is None:
            self._calc_rebdates()
        return sorted(self._rebdates)


class MonRebCalcu(RebCalcu):
    '''
    每个月的最后一个交易日作为再平衡日
    '''

    def _calc_rebdates(self):
        '''
        用于计算给定的时间区间内的再平衡日（指因子计算日，且类型为datetime），并将其存储在_rebdates中
        '''
        tds = pd.Series(get_tds(self._start_time, self._end_time))
        tds.index = tds.dt.strftime('%Y-%m')
        self._rebdates = tds.groupby(lambda x: x).apply(lambda y: y.iloc[-1]).tolist()


class WeekRebCalcu(RebCalcu):
    '''
    每个周的最后一个交易日作为再平衡日
    '''

    def _calc_rebdates(self):
        '''
        用于计算给定的时间区间内的再平衡日（指因子计算日，且类型为datetime），并将其存储在_rebdates中
        '''
        tds = pd.Series(get_tds(self._start_time, self._end_time))
        tds.index = tds.dt.strftime('%Y-%W')
        self._rebdates = tds.groupby(lambda x: x).apply(lambda y: y.iloc[-1]).tolist()


# --------------------------------------------------------------------------------------------------
# 函数
def load_rebcalculator(reb_type, start_time, end_time):
    '''
    加载指定频率的再平衡日计算器

    Parameter
    ---------
    reb_type: str
        换仓日计算的规则，目前只支持月度(MONTHLY)和周度(WEEKLY)
    start_time: datetime like
        交易日的起始时间
    end_time: datetime like
        交易日的终止时间

    Return
    ------
    out: RebCalcu
        对应频率的再平衡日计算器
    '''
    valid_freq = [MONTHLY, WEEKLY]
    assert reb_type in valid_freq, \
        'Rebalance date method setting ERROR, you provide {yp}, '.format(yp=reb_type) +\
        'right choices are {rc}'.format(rc=valid_freq)
    if reb_type == MONTHLY:
        res = MonRebCalcu(start_time, end_time)
    else:
        res = WeekRebCalcu(start_time, end_time)
    return res

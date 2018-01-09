#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-05 15:42:53
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
__version__ = 1.0.0
修改日期：2017-07-11
修改内容：
    初步完成基础数据存储模块

'''
__version__ = "1.0.0"
# import datatoolkits
import dateshandle
import numpy as np
import pandas as pd
import pdb
import h5py

# 本地文件
from fmanager.database.const import *


class DBConnector(object):
    '''
    负责处理底层数据的存储工作类，主要功能包含：存储文件初始化、添加初始数据，数据定期更新，数据提取
    '''

    def __init__(self, path, size=MAX_COL_SIZE):
        self.path = path
        self._data_time = None   # 用于缓存数据的最新时间
        self._code_order = None  # 用于缓存数据的股票代码顺序
        self._code_dtype = 'S12'    # 用于标识标准股票代码数据格式
        self._date_dtype = 'S10'    # 用于标识标准日期数据格式
        self._size = size   # 用于标识横截面的数据的长度
        self._data_type = None   # 用于记录数据类型
        self._default_data = None    # 用于记录默认填充数据

    def init_dbfile(self, data_type='f8'):
        '''
        初始化一个HDF5文件

        Parameter
        ---------
        data_type: str
            文件的类型

        Notes
        -----
        该操作中会创建一个新的文件，如果文件已经存在会报错。创建文件后，会按照数据的模板对数据进行初始化
        的设置。
        '''
        with h5py.File(self.path, 'w-') as store:
            # 考虑未来添加数据的格式改变的而需求，将数据的格式设置的更大一些
            store.create_dataset('date', shape=(1,), maxshape=(None,), dtype=self._date_dtype)
            store.create_dataset('code', shape=(self._size,), chunks=(self._size,),
                                 dtype=self._code_dtype)
            store.create_dataset('data', shape=(1, self._size), chunks=(1, self._size),
                                 dtype=data_type, maxshape=(None, self._size))
            if data_type.startswith('f'):   # 当数据类型时，填充数据为np.nan
                self._default_data = np.nan
            else:   # 当数据为字符串时，填充NAS字符（not a string）
                self._default_data = np.bytes_(NaS)
            store.attrs['default data'] = self._default_data  # 用于标识默认填充数据
            store.attrs['status'] = 'empty'     # 用于标识是否有数据填充，包含两种状态（empty, filled）
            store.attrs['data time'] = 'nat'   # 用于标识数据的最新时间，初始化填充nat，即not a time
            store.attrs['data type'] = data_type    # 用于标识存储的数据类型，用于区分数字类数据和字符串
            store.attrs['#code'] = 0    # 记录当前数据中有效的股票数量
            store.attrs['#dates'] = 0   # 记录当前数据中有效的日期数量
            store['data'][...] = self._default_data

    def insert_data(self, code, date, data):
        '''
        向数据文件中添加数据

        Parameter
        ---------
        code: np.array(dtype='S9')
            股票列表
        dates: np.array(dtype='S10')
            时间字符串列表
        data：np.array()
            已经经过转换的数据，为二维矩阵，其中列方向（0轴）代表时间，行方向（1轴）代表股票；要求
            无论股票还是时间都应该对应住，且时间按照升序排列

        Notes
        -----
        在插入数据前会先做检查，传入的数据类型是否符合数据文件的要求
        '''
        with h5py.File(self.path, 'r+') as store:
            # 检查输入是否合法
            # assert store.attrs['status'] == 'empty', "cannot insert data to a filled dataset"
            assert store.attrs['data type'] == np.dtype(data.dtype), "data type error!" +\
                "data type in dataset is {ds_type}, you provide |{p_type}".\
                format(ds_type=data.dtype, p_type=store.attrs['data type'])
            assert data.shape[1] < self._size,\
                "data columns(len={data_len}) ".format(data_len=data.shape[1]) +\
                "should be less than {max_len}".format(max_len=self._size)
            assert data.shape == (len(date), len(code)), "input data error, " +\
                "data imply shape = {data_shape}, while code and date imply shape = {other_shape}".\
                format(data_shape=data.shape, other_shape=(len(date), len(code)))
            # 查找日期索引值
            start_date = int(store.attrs['#dates'])
            # default_data = store.attrs['default data']
            # 修改相关属性
            if store.attrs['status'] == 'empty':
                store.attrs['status'] = 'filled'
            store.attrs['data time'] = date[-1].decode('utf8')
            store.attrs['#code'] = len(code)
            new_datelen = store.attrs['#dates'] + len(date)
            store.attrs['#dates'] = new_datelen
            # 对数据进行插入
            date_dset = store['date']
            data_dset = store['data']
            code_dset = store['code']
            date_dset.resize((new_datelen, ))
            data_dset.resize((new_datelen, self._size))   # 此处resize后填充的数据为0
            date_dset[start_date:new_datelen] = date
            data_dset[start_date:new_datelen, :len(code)] = data
            data_dset[start_date:new_datelen, len(code):] = self.default_data    # 填充其余位置的数据
            code_dset[:len(code)] = code
            # 更新数据的最新时间和股票列表顺序
            self._data_time = pd.to_datetime(date[-1].decode('utf8'))
            self._code_order = [c.decode('utf8') for c in code]

    @property
    def data_time(self):
        '''
        返回最新的数据时间
        '''
        if self._data_time is None:
            with h5py.File(self.path, 'r') as store:
                data_time = store.attrs['data time']
            if data_time == 'nat':
                return None
            data_time = pd.to_datetime(data_time)
            self._data_time = data_time
            return data_time
        return self._data_time

    @property
    def code_order(self):
        '''
        返回数据的股票代码顺序
        '''
        if self._code_order is None:
            with h5py.File(self.path, 'r') as store:
                code_order = store['code'][...]
            code_order = [c.decode('utf8') for c in code_order if len(c) > 0]
            if len(code_order) == 0:
                return None
            return code_order
        return self._code_order

    @property
    def default_data(self):
        '''
        以原始的形式返回数据库的默认填充数据
        '''
        if self._default_data is None:
            with h5py.File(self.path, 'r') as store:
                self._default_data = store.attrs['default data']
        return self._default_data

    def _query_panel(self, start_time, end_time):
        '''
        查询面板数据

        Parameter
        ---------
        start_time: datetime
            查询的数据的开始时间
        end_time: datetime
            查询的数据的结束时间

        Return
        ------
        out: pd.DataFrame
            返回结果数据，索引为时间，列为股票代码
            结果中只返回数据文件中有数据的部分，若查询时间都不在数据的时间范围内，则返回None

        Notes
        -----
        查询结果同时包含start_time和end_time的数据
        '''

        out = None
        try:
            store = h5py.File(self.path, 'r')
            dset_dates = store['date']
            codes = store['code']
            code_len = store.attrs['#code']
            codes = [c.decode('utf8') for c in codes[:code_len]]
            dset_data = store['data']
            dset_dates = store['date']
            data = dset_data[:, :code_len]
            data_type = store.attrs['data type']
            if data_type[0].lower() == 's':  # 检查数据的格式，如果为字符串则进行类型转换
                new_data_type = 'U' + data_type[1:]
                data = data.astype(new_data_type)
            dates = [pd.to_datetime(d.decode('utf8')) for d in dset_dates]
            out = pd.DataFrame(data, index=dates, columns=codes)
            out = out.loc[(out.index <= end_time) & (out.index >= start_time)]
            if out.empty:
                out = None
        finally:
            store.close()
        return out

    def query(self, date, codes=None):
        '''
        根据给定的时间和股票代码的条件查询数据，支持时间点、时间区间、单个股票、多个股票或者全部股票
        的查询

        Parameter
        ---------
        date: str or datetime or tuple
            查询数据的时间，可以是时间点或者时间区间（用元组表示），类型要求是可以被pd.to_datetime转换
            的即可
        codes: list, default None
            查询的股票代码，None表示返回所有股票的结果

        Return
        ------
        out: pd.DataFrame
            返回结果的DataFrame，index为日期，columns为股票代码。返回的结果中只会返回时间轴上有数据的部分，
            股票代码轴上没有数据对应的值会设为NA，其他没有有效数据的情况返回None
        '''
        if isinstance(date, tuple):
            start_time, end_time = [pd.to_datetime(d) for d in date]
        else:
            start_time = end_time = pd.to_datetime(date)
        data = self._query_panel(start_time, end_time)
        if data is None:    # 没有符合时间要求的数据
            return None
        if codes is None:   # 返回所有股票的数据
            return data
        assert isinstance(codes, list), 'Error, parameter "codes" should be provides as a list!'
        data_codes = self.code_order
        valid_codes = [c for c in codes if c in data_codes]
        invalid_codes = list(set(codes).difference(valid_codes))
        if len(invalid_codes) > 0:
            print("Warning: invalid codes({codes}) are queryed!".format(codes=invalid_codes))
        if len(valid_codes) == 0:   # 没有提供有效的股票代码
            return None

        out = data.loc[:, codes]
        return out

    def query_all(self):
        '''
        查询所有的数据
        Return
        ------
        out: pd.DataFrame
            返回结果的DataFrame，index为日期，columns为股票代码。
        '''
        start_time = FIRST_TRADING_DAY
        end_time = self.data_time
        return self.query((start_time, end_time))

    def insert_df(self, df, data_dtype=None, filled_value=np.nan):
        '''
        将DataFrame插入数据库中

        Parameter
        ---------
        df: pd.DataFrame
            需要插入的数据，要求index为时间，columns为股票代码
        data_dtype: str, default None
            pd.DataFrame中的数据与数据库中的数据格式不匹配，需要对pd.DataFrame进行适当的转换，默认为
            None表示不需要转换，否则则需要提供转换后的格式形式
        filled_value: str or float or else, default np.nan（目前参数已废止）
            当插入数据的列与数据文件中的数据列不匹配时，需要对源数据一些空余的列做填充，默认填充
            NA

        Return
        ------
        out: pd.DataFrame
            转换后的插入数据库中的数据

        Notes
        -----
        转换过程中会检查数据在时间轴上是否连续，如果为非连续数据，则会报错
        '''
        def _check_date(df_dates):
            '''
            检测日期是否连续
            '''
            df_dates = sorted([pd.to_datetime(dt) for dt in df_dates])
            tds = dateshandle.get_tds(df_dates[0], df_dates[-1])
            return tds == df_dates

        if not (self.data_time is None or self.code_order is None):     # 非第一次插入数据
            df = df.loc[df.index > self.data_time]
            if not len(df):  # 没有需要插入的数据
                return df
            df_dates = df.index.tolist()
            df_dates.append(self.data_time)

            assert _check_date(df_dates), ValueError("Discontinuous data!")   # 检测将插入数据与数据库数据日期是否连续
            diff_codes = df.columns.difference(self.code_order)
            new_codes = self.code_order + sorted(diff_codes)
            # df = df.loc[:, new_codes].fillna(filled_value)
            # 下面这段代码应该不会起作用，因为set(diff_codes+code_order) == set(df.columns)
            df = df.loc[:, new_codes].fillna(self.default_data)  # 采用默认数据填充
        else:
            assert _check_date(df.index.tolist()), ValueError("Discontinuous data!")
            df = df.sort_index().sort_index(axis=1)
        # 将数据转化为数据库文件可以识别的格式
        codes = np.array([c for c in df.columns], dtype=self._code_dtype)
        dates = np.array([c for c in df.index], dtype=self._date_dtype)
        # pdb.set_trace()
        if data_dtype is not None:
            data = df.values.astype(data_dtype)
        else:
            data = df.values
        self.insert_data(codes, dates, data)
        return df


# 辅助函数，将数据进行迁移或者替换
def reshape_colsize(file_path, new_size, destination_path=None):
    '''
    改变原有数据的列的大小，并将其存储到（一个新的）文件中

    Parameter
    ---------
    file_path: str
        需要转换列大小的文件的路径
    new_size: int
        新的列的大小，要求必须比已经存储的数据列数要大
    destination_path: str, default
        新的存储路径（可选），如果为None，表示直接替代之前的文件
    '''
    raw_db = DBConnector(file_path)
    data = raw_db.query_all()
    assert len(data.columns) < new_size, \
        "Error, new size({ns}) must be greater than data columns size({dz})!".\
        format(ns=new_size, dz=len(data.columns))
    with h5py.File(file_path) as store:     # 获取存储数据的相关信息
        data_type = store.attrs['data type']
        default_value = store.attrs['default data']
    if destination_path is None:    # 覆盖源文件
        print('Warning: file({fpath} is removed!'.format(fpath=file_path))
        from os import remove
        remove(file_path)
        new_db = DBConnector(file_path, new_size)
    else:
        new_db = DBConnector(destination_path, new_size)
    new_db.init_dbfile(data_type)
    new_db.insert_df(data, data_type, default_value)


if __name__ == '__main__':
    from fmanager import get_factor_detail
    cpath = get_factor_detail('TOTAL_MKTVALUE')['abs_path']
    test_path = r'C:\Users\lenovo\Desktop\test\test_db.h5'
    db = DBConnector(cpath)
    data = db.query(('2007-01-01', '2016-12-01'))
    # reshape_colsize(cpath, 5000, test_path)

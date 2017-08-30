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

__version__ = 1.1
修改日期：2017年1月4日
修改内容：
    添加在数据库中股票代码后添加后缀的函数add_suffix

__version__ 1.2
修改日期：2017年1月13日
修改内容：
    添加map_data函数

__version__ 1.3
修改日期：2017年1月18日
修改内容：
    修复了map_data中的BUG，使其能够正确返回交叉时间段内的数据结果
修改日期：2017-03-07
修改内容：
    1. 重构了map_data，使用更精简的函数实现
    2. 加入存储和读取pickle数据的函数load_pickle、dump_pickle

__version__ = 1.4
修改日期：2017年3月29日
修改内容：
    1. 从processsingdata中转移而来
    2. 添加retfreq_trans函数
    3. 添加annualize_std函数

__version__ = 1.5
修改日期：2017-03-31
修改内容：
    修改了map_data，添加了一些排序过程

__version__ = 1.6
修改日期：2017-04-10
修改内容：
    在map_data中添加了重置索引，使得返回的数据没有索引

__version__ = 1.7
修改日期：2017-04-19
修改内容：
    添加quantile_sub

__version__ = 1.7.1
修改日期：2017-05-03
修改内容：
    map_data添加填充NA值的选项

__version__ = 1.8
修改日期：2017-05-10
修改内容：
    添加isclose函数，弥补Python 3.6之前math没有类似函数的缺陷

__version__ = 1.8.1
修改日期：2017-05-12
修改内容：
    修改retfreq_trans函数的相关说明

__version__ = 1.9.0
修改日期：2017-05-16
修改内容：
    添加数据标准化函数standardlize

__version__ = 1.9.1
修改日期：2017-05-17
修改内容：
    添加extract_factor_OLS和demean函数


__version__ = 1.10.0
修改日期：2017-05-22
修改内容：
    添加DataLoader

__version__ = 1.10.1
修改日期：2017-05-23
修改内容：
    在DataLoader中添加None的选项，可以加载空值数据

__version__ = 1.10.2
修改日期：2017-06-07
修改内容：
    添加自定义的rolling_apply函数

__version__ = 1.10.3
修改日期：2017-06-08
修改内容：
    为rolling_apply添加显示进度的功能

__version__ = 1.10.4
修改日期：2017-06-20
修改内容：
    添加winsorize函数

__version__ = 1.10.5
修改日期：2017-06-21
修改内容：
    添加正交化函数orthogonalize_lstsq

__version__ = 1.10.6
修改日期：2017-06-27
修改内容：
    添加计算加权平均值的wmean函数

__version__ = 1.10.7
修改日期：2017-08-28
修改内容：
    添加将（价格）数据转换为净值数据的函数price2nav
'''
__version__ = '1.10.5'

import datetime as dt
from math import sqrt
import numpy as np
import pandas as pd
import pdb
import pickle
import statsmodels.api as sm
# import six
# import time     # for test
from tqdm import tqdm

# --------------------------------------------------------------------------------------------------
# 函数


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


def add_suffix(code):
    '''
    从数据库中获取的股票代码添加后缀，以60开头的代码添加.SH，其他添加.SZ
    @param:
        code: 需要转换的代码
    @return:
        转换后的代码
    注：转换前会检测是否需要转换，但是转换前不会检测代码是否合法
    '''
    if code.endswith('.SH') or code.endswith('.SZ'):
        return code
    if code.startswith('60'):   #
        suffix = '.SH'
    else:
        suffix = '.SZ'
    return code + suffix


def drop_suffix(code, suffixLen=3, suffix=('.SH', '.SZ')):
    '''
    将Wind等终端获取的数据的后缀转换为数据库中无后缀的代码
    @param:
        code: 需要转换的代码
        suffixLen: 后缀代码的长度，包含'.'，默认为3
        suffix: 后缀的类型，默认为('.SH', '.SZ')
    @return:
        转换后的代码
    '''
    for s in suffix:
        if code.endswith(s):
            break
    else:
        return code
    return code[:-suffixLen]


def map_data(rawData, days, timeCols='time', fromNowOn=False, fillna=None):
    '''
    将给定一串时点的数据映射到给定的连续时间上，映射规则如下：
    若fromNowOn为True时，则在rawData中给定时点以及该时点后的时间的值等于该时点的值，因此第一个日期无论
    其是否为数据更新的时间点，都会被抛弃
    若fromNowOn为False时，则在rawData中给定时点后的时间的值等于该时点的值
    最终得到的时间序列数据为pd.DataFrame的格式，数据列为在当前时间下，rawData中所对应的最新的值，对应
    方法由映射规则给出
    例如：若rawData中包含相邻两项为(2010-01-01, 1), (2010-02-01, 2)，且fromNowOn=True，则结果中，从
    2010-01-01起到2010-02-01（不包含当天）的对应的值均为1，若fromNowOn=False，则结果中，从2010-01-01
    （不包含当天）起到2010-02-01对应的值为1
    注：该函数也可用于其他非时间序列的地方，只需要有序列号即可，那么rowData的数据应为[(idx, value), ...]
    @param:
        rawData: 为pd.DataFrame格式的数据，要求包含timeCols列
        days: 所需要映射的日期序列，要求为列表或者其他可迭代类型（注：要求时间格式均为datetime.datetime）
        timeCols: 时间列的列名，默认为time
        fromNowOn: 默认为False，即在给定日期之后的序列的值才等于该值
        fillna: 填充缺省值，默认为None，即不做任何填充，填充方式为{col: func}，func接受rawData为参数，
            返回一个值，作为填充
    @return:
        pd.DataFrame格式的处理后的数据，数据长度与参数days相同，<del>且时间列为索引</del>
    '''
    if not isinstance(days, list):
        days = list(days)
    days = sorted(days)
    time_col = sorted(rawData[timeCols].tolist())
    time_col = [t for t in time_col if t < days[0]] + days
    data = rawData.set_index(timeCols)
    data = data.sort_index()
    try:
        data = data.reindex(time_col, method='ffill')
    except ValueError as e:
        print(rawData.code.iloc[0])
        raise e
    if not fromNowOn:
        data = data.shift(1)    # 若报告发布日期在两个交易日之间，则会导致数据会滞后一个交易日
    data = data.reindex(days)
    data = data.reset_index()
    if fillna:
        for col in fillna:
            func = fillna[col]
            fill_value = func(rawData)
            data.loc[:, col] = data[col].fillna(fill_value)
    return data


def date_processing(date, dateFormat='%Y-%m-%d'):
    '''
    用于检查日期的类型，如果为str则转换为datetime的格式
    @param:
        date: 输入的需要处理的日期
        dateFormat: 日期转换的格式，默认为None，表示格式为YYYY-MM-DD
    @return:
        按照转换方法转换后的格式
    '''
    if not isinstance(date, dt.datetime):
        date = dt.datetime.strptime(date, dateFormat)
    return date


def load_pickle(path):
    '''
    读取pickle文件
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(data, path):
    '''
    将数据写入到pickle文件中
    '''
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def retfreq_trans(init_ret, new_freq):
    '''
    将收益的频率进行转换，例如将日收益率转化为年化或者将年化收益率转化为日收益率
    该函数只支持标量转换

    Parameter
    ---------
    init_ret: float
        初始的需要转换的收益率
    new_freq: int
        最终收益率的频率，例如将月度的收益率年化则为12

    Return
    ------
        转换频率后的收益率

    Notes
    -----
    计算方法如下：
    new_ret = (1 + init_ret)**new_freq - 1
    '''
    return (1 + init_ret)**new_freq - 1


def avg_compunded_ret(rets):
    '''
    计算复合平均收益率
    计算方法如下：
    avg_ret = cumprod(1 + rets)**(1/len(rets)) - 1
    @param:
        rets: 收益率序列，要求为pd.Series类型
    @return:
        复合平均收益率
    '''
    return (1 + rets).cumprod().iloc[-1]**(1 / len(rets)) - 1


def annualize_ret(rets, ret_freq):
    '''
    通过收益率序列计算年化收益率
    计算方法如下：
    annualize_ret = cumprod(1+rets)**(ret_freq/len(rets)) - 1
    @param:
        rets: 收益率序列，为pd.Series类型
        ret_freq: 转换频率，例如，月度数据年化为12，日数据年化为250
    @return:
        年化后的收益率
    '''
    return (1 + avg_compunded_ret(rets))**ret_freq - 1


def annualize_std(rets, ret_freq):
    '''
    计算年化的波动率
    计算方法如下：
    new_std = sqrt((init_std**2 + (1+init_ret)**2)**ret_freq - (1+init_ret)**(2*ret_freq))
    init_std和init_ret使用样本标准差和样本均值计算
    @param:
        rets: 需要计算年化波动率的收益率序列，为pd.Series类型
        ret_freq: 一年的区间数（例如，12表示月度数据年化，250表示日数据年化）
    @return:
        按照上述方法计算的年化波动率
    '''
    init_std = rets.std()
    init_ret = rets.mean()
    return sqrt((init_std**2 + (1 + init_ret)**2)**ret_freq - (1 + init_ret)**(2 * ret_freq))


def gen_series(cols, fill=np.nan):
    '''
    生成给定数据填充的Series
    @param:
        cols: 对应index，要求为可迭代类型
        fill: 默认为np.nan，也可定义，为填充数据
    @return:
        index给定，数据为fill的一列Series
    '''
    return pd.Series(dict(zip(cols, [fill] * len(cols))))


def gen_df(cols, fill=np.nan):
    '''
    生成给定数据填充的DataFrame
    @param:
        cols: 对应index，要求为可迭代类型
        fill: 默认为np.nan，也可定义，为填充数据
    @return:
        index给定，数据为fill的只有一行数据的DataFrame
    '''
    return pd.DataFrame(dict(zip(cols, [[fill]] * len(cols))))


def quantile_sub(data, qtls, sub=np.nan):
    '''
    将超过分位数的数据进行替换
    @param:
        data: pd.Series
        qtls: 分位数标准，格式为(min_qtl, max_qtl)
        sub: 替代的数据，默认为np.nan
    @return:
        df，分位数定义的异常值均被sub提供的数据替代
    '''
    df = data.copy()
    qtl1, qtl2 = df.quantile(qtls)
    df.loc[(df < qtl1) | (df > qtl2)] = sub
    return df


def winsorize(s, qtls):
    '''
    将超过分位数的数据拉回到给定的分位数中

    Parameter
    ---------
    s: pd.Series
        需要被拉回的序列
    qtls: list/tuple or other types with index (0, 1)
        上限和下限的分为点，格式为(lower_quantile, upper_quantile)

    Return
    ------
    out: pd.Series
        经过分位数拉回的序列
    '''
    out = s.copy()
    lower_quantile, upper_quantile = out.quantile(qtls)
    out.loc[out > upper_quantile] = upper_quantile
    out.loc[out < lower_quantile] = lower_quantile
    return out


def isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    '''
    判断两个数值是否接近，计算方法如下
    abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    Parameter
    ---------
    a: float
        需要比较的数之一
    b: float
        需要比较的数之二
    rel_tol: float
        相对的容错度
    abs_tol: float
        绝对容错度

    Return
    ------
    out: bool
        当两个数字之间的距离满足上述公式，返回True，反之则返回False
    '''
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def standardlize(datas):
    '''
    对序列进行标准化，即对所有的数据减去均值除以方差

    Parameter
    ---------
    datas: Series or list like
        需要进行标准化的数据

    Return
    ------
    out: Series
        标准化后的数据
    '''
    if not isinstance(datas, pd.Series):
        datas = pd.Series(datas)
    out = (datas - datas.mean()) / datas.std()
    return out


def wmean(data, skipna=True, weight=None):
    '''
    计算数据的（加权）均值

    Parameter
    ---------
    data: pd.Series or types that can be converted to pd.Series
        需要计算均值的数据
    skipna: boolean, default True
        是否忽略NA值，默认忽略
    weight: pd.Series, default None
        加权的权重，默认为None，表示等权。要求weight与data具有相同的长度和索引，对weight的数值不要求
        和为一，计算过程中会自动归一化，但是权重的和不能为0

    Return
    ------
    out: float
        计算的（加权）均值结果

    Notes
    -----
    这种方法计算的速度要比原生的要慢一倍，尽量少使用
    '''
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    else:
        data = data.copy()
    if weight is None:
        return data.mean(skipna=skipna)
    weight = weight.copy()
    if skipna:
        weight[pd.isnull(data)] = 0
        data[pd.isnull(data)] = 0
        weight[pd.isnull(weight)] = 0
    assert len(weight) == len(data), "Error, length of data must be equal to length of weight"
    weight_sum = weight.sum()
    try:
        weight = weight / weight_sum
    except ZeroDivisionError:
        print('Error, valid weights sum to ZERO!')
        return np.nan
    out = data.dot(weight)
    return out


def extract_factor_OLS(data, factor_col, x_cols, standardization=True):
    '''
    使用OLS回归的方法，剔除新的因子中与以前因子相关的部分，即通过使用因子值对
    现有因子做回归，取残差
    Parameter
    ---------
    data: DataFrame
        输入的数据，必须包含factor_col和x_cols参数中的列
    factor_col: str
        需要进行处理的因子值的列名
    x_cols: str or list
        需要从因子中剔除影响的列的列名
    standardization: bool, default True
        是否对因子进行标准化处理

    Return
    ------
    out: Series
        处理后的因子数据
    '''
    if isinstance(x_cols, str):
        x_cols = [x_cols]
    x_data = data.loc[:, x_cols]
    y_data = data.loc[:, factor_col]
    if standardization:
        for col in x_cols:
            x_data.loc[:, col] = standardlize(x_data[col])
        y_data = standardlize(y_data)
    x_data = sm.add_constant(x_data)
    model = sm.OLS(y_data, x_data)
    res = model.fit()
    out = pd.Series(res.resid, index=data.index)
    return out


def demean(data, weight=None, skipna=True):
    '''
    对数据减去均值，默认减去等权的均值，也可自行提供权重，不要求权重的和为1，程序会自动对权重
    进行正则化
    Parameter
    ---------
    data: Series
        需要去除均值的序列
    weight: Series, default None
        权重，不要求和为1，默认为None表示等权
    skipna: bool, default True
        是否跳过data中的NA值
    Return
    ------
    out: Series
        减去加权均值后的序列
    '''
    if weight is None:
        out = data - data.mean(skipna=True)
    else:
        weight = weight / np.sum(weight)
        data_mean = (data * weight).sum(skipna=skipna)
        out = data - data_mean
    return out


def rolling_apply(df, func, period, min_period=None, show_progress=False, **kwargs):
    '''
    在移动窗口中进行计算的函数
    Parameter
    ---------
    df: DataFrame
        需要进行滚动窗口计算的DataFrame
    func: function(df, **kwargs) -> value
        要求函数必须以DataFrame为参数传入，且返回单一一个数值结果
    period: int
        窗口长度
    min_period: int
        最小窗口长度，如果未给定，则与period给定的参数相同
    show_progress: bool, default False
        显示计算的进度
    kwargs: additional parameters
        用于提供给func的其他参数

    Return
    ------
    out: Series
        移动窗口计算后的结果，数值不足填充NA，索引与原来给定的df的索引相同
    '''
    if min_period is None:
        min_period = period
    res = pd.Series(np.nan, index=df.index)
    df = df.copy()
    idx_range = range(1, len(df) + 1)
    # 显示进度
    if show_progress:
        idx_range = zip(idx_range, tqdm(idx_range))
    for idx in idx_range:
        if show_progress:
            i, _ = idx
            # time.sleep(1)
        else:
            i = idx
        tmp_df = df.iloc[max(0, i - period): i]
        if len(tmp_df) >= min_period:
            idx = tmp_df.index[-1]
            res[idx] = func(tmp_df, **kwargs)
    return res


def orthogonalize_lstsq(a, b, weight=None):
    '''
    使用最小二乘的方法对数据进行正交化处理

    Parameter
    ---------
    a: pd.Series
        需要被正交化处理的序列
    b: pd.Series or pd.DataFrame
        正交化处理时使用的参考序列
    weight: pd.Series or None, default None
        正交化过程中的权重，如果提供的权重为(w1, w2, ...)，则会事先对数据都乘以(sqrt(w1), sqrt(w2),...)
        默认为None表示等权

    Return
    ------
    out: pd.Series
        正交化处理以后的序列，index与a相同

    Notes
    -----
    正交化过程中，如果需要被正交化的数据中有NA值，则将其当做均值处理，最后再将NA填充至原位置
    '''
    if isinstance(b, pd.Series):
        b = pd.DataFrame({'0': b})
    # 填充NA值
    filtered_a = a.fillna(a.mean())
    filtered_b = b.fillna(b.mean())
    # 数据按照权重进行转换
    if weight is None:
        weight = pd.Series(np.ones(len(a)))
    else:
        weight = np.sqrt(weight)
    # 对数据进行转换
    trans_a = weight * filtered_a
    trans_b = filtered_b.mul(weight, axis=0)
    # 使用最小二乘计算系数
    param = np.linalg.lstsq(trans_b, trans_a)[0]
    # pdb.set_trace()
    out = a - b.dot(param)
    return out


def price2nav(price_data):
    '''
    将价格数据转换为净值数据

    Parameter
    ---------
    price_data: pd.DataFrame or pd.Series
        需要转换的原始数据，要求数据按照时间纵向排列

    Return
    ------
    out: pd.DataFrame or pd.Series, dependent to input
    '''
    out = price_data / price_data.iloc[0]
    return out

# --------------------------------------------------------------------------------------------------
# 类
# 通用加载数据类


def _hdf_loader(path, key, **kwargs):
    '''
    读取HDF文件数据
    Parameter
    ---------
    path: str
        文件路径
    key: str
        数据关键字名称
    kwargs: dict
        其他HDFStore.select参数

    Return
    ------
    out: pd.DataFrame
        结果数据
    '''
    with pd.HDFStore(path) as store:
        out = store.select(key, **kwargs)
    return out


def _none_loader(path, **kwargs):
    return None


class DataLoader(object):
    '''
    一个加载数据的通用接口，目前支持HDF和pickle的格式
    '''
    _loader = {'pickle': load_pickle, 'HDF': _hdf_loader, 'None': _none_loader}

    def __init__(self, data_type, path, **kwargs):
        '''
        Parameter
        ---------
        data_type: str
            目前仅支持HDF和pickle
        path: str
            文件路径
        kwargs: dict
            其他参数，主要针对HDF格式的文件，需要说明文件所使用的数据名称，对于HDF文件，该参数的形式
            必须为key="name"，也可以包含HDFStore.select的其他参数
        '''
        self.data_type = data_type
        self.path = path
        self.kwargs = kwargs
        self._cache = None

    def _check_validtype(self, data_type):
        '''
        检测参数中给定的数据类型是否合法
        '''
        assert data_type in DataLoader._loader, \
            'Error, valid data types are {valid}, you provide {given}'. \
            format(valid=list(DataLoader._loader.keys()), given=data_type)

    def load_data(self):
        '''
        通用加载数据接口
        '''
        if self._cache is None:
            loader = DataLoader._loader[self.data_type]
            kwargs = self.kwargs
            out = loader(self.path, **kwargs)
            self._cache = out
        else:
            out = self._cache
        return out

    def clear_cache(self):
        '''
        清理缓存数据
        '''
        self._cache = None


if __name__ == '__main__':
    np.random.seed(100)
    df = pd.DataFrame(np.random.rand(10, 3), columns=['group_%d' % i for i in range(1, 4)])

    def foo(df):
        res = df['group_1'].mean() - 0.5 * df['group_2'].mean() - 0.5 * df['group_3'].mean()
        return res
    result = rolling_apply(df, foo, 4, min_period=2, show_progress=True)

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:26:14 2016

@author: hao
模块描述：
    本模块用于利用数据画出简单的K线图
    输入要求为pandas.DataFrame的格式，必须有open, close, high, low数据列
    可以有时间数据（time）列
"""
__version__ = 2.0

from itertools import groupby
import pdb

import matplotlib.pyplot as plt
from matplotlib import finance
# import matplotlib.dates as mpldates
from matplotlib import ticker


def plot_candle(data, cols=None, time_index=True, time_col=None, major_group_fmt='%Y',
                major_loc_fmt='%Y-%m', enable_minor_loc=True, minor_group_fmt='%Y-%m',
                minor_loc_fmt='%m-%d', rotation=0, stick_width=0.8, alpha=1):
    '''
    使用data提供的数据画k线图

    Parameter
    ---------
    data: pd.DataFrame or dict of Series
        输入的原始数据
    cols: iterable, default None
        显式按照顺序说明'open', 'high', 'low', 'close'所在的列的列名，如果该参数为None，则默认各个列
        名依次为'open', 'high', 'low', 'close'
    time_index: boolean, default True
        时间或者日期数据是否在DataFrame的Index中，默认为True。如果为False，则time_col不能为None，
        要求时间的数据为datetime或者其子类
    time_col: string, default None
        时间或者日期数据所在的列，只有在time_index为False时调用
    major_group_fmt: string, default '%Y'
        依据major_group_fmt对日期进行分组，并以每个分组的最小日期作为位置，要求能被datetime.strftime
        使用
    major_loc_fmt: string, default '%Y-%m'
        major tick格式化的方式，格式要求与major_group_fmt相同
    enable_minor_loc: boolean, default True
        是否启用minor tick
    minor_group_fmt: string, default '%Y-%m'
        若启用了minor tick，则提供与major_group_fmt相同的定位功能
    minor_loc_fmt: string, default '%Y-%m-%d'
        若启用了minor tick, 则提供与major_loc_fmt相同的格式化功能
    rotation: float, default 0
        横轴标签的旋转角度
    stick_width: float, default 0.6
        每根k线的宽度
    alpha: float, default 1
        透明度
    '''
    # 预处理
    if cols is None:
        cols = ('open', 'high', 'low', 'close')
    if time_index:
        date_index = data.index.tolist()
    else:
        date_index = data[time_col].tolist()

    # locator定位函数，返回经过给定格式分类后生成的tick的位置和label
    def locate_ticker(group_fmt, loc_fmt):
        by_fmt = groupby(date_index, lambda x: x.strftime(group_fmt))
        idx = [date_index.index(min(g)) for _, g in by_fmt]
        dates = [date_index[i].strftime(loc_fmt) for i in idx]
        return idx, dates

    # 初始化图片
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # 计算并设置major tick
    # 计算位置和相关格式
    major_index, major_label = locate_ticker(major_group_fmt, major_loc_fmt)
    major_loc = ticker.FixedLocator(major_index)
    major_fmter = ticker.FixedFormatter(major_label)
    # 设置到图片
    ax.xaxis.set_major_locator(major_loc)
    ax.xaxis.set_major_formatter(major_fmter)
    # 计算minor tick
    if enable_minor_loc:
        minor_index, minor_label = locate_ticker(minor_group_fmt, minor_loc_fmt)
        minor_loc = ticker.FixedLocator(minor_index)
        minor_fmter = ticker.FixedFormatter(minor_label)
        ax.xaxis.set_minor_locator(minor_loc)
        ax.xaxis.set_minor_formatter(minor_fmter)
        # pdb.set_trace()
    # 加载数据，画图
    ochl_data = data.loc[:, cols].values.T
    finance.candlestick2_ohlc(ax, *ochl_data, width=stick_width, colorup='red', colordown='green',
                              alpha=alpha)
    plt.setp(ax.get_xticklabels(), rotation=rotation)
    if enable_minor_loc:
        # pdb.set_trace()
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=rotation)
        for xlabel in ax.xaxis.get_majorticklabels():   # 隐藏major tick
            xlabel.set_visible(False)
        for xlabel in ax.xaxis.get_minorticklabels():   # 显示minor tick
            xlabel.set_visible(True)

    xmin, _ = plt.xlim()
    plt.xlim(xmin=xmin - 1)
    plt.show()


if __name__ == '__main__':
    import pandas as pd
    import fmanager
    start_time = '2016-01-01'
    end_time = '2017-02-01'
    open_data = fmanager.query('OPEN', (start_time, end_time)).iloc[:, 0]
    close_data = fmanager.query('CLOSE', (start_time, end_time)).iloc[:, 0]
    high_data = fmanager.query('HIGH', (start_time, end_time)).iloc[:, 0]
    low_data = fmanager.query('LOW', (start_time, end_time)).iloc[:, 0]
    data = pd.DataFrame({'open': open_data, 'close': close_data, 'high': high_data,
                         'low': low_data})
    plot_candle(data.reset_index(), time_col='index', time_index=False, rotation=45)

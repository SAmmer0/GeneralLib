# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:26:14 2016

@author: hao
模块描述：
    本模块用于利用数据画出简单的K线图
    输入要求为pandas.DataFrame的格式，必须有open, close, high, low数据列
    可以有时间数据（time）列
"""
__version__ = 1.0
import matplotlib.pyplot as plt
from matplotlib import finance
#import matplotlib.dates as mpldates
from matplotlib import ticker

def time_formatterSetting(ax, times, majorStep=20, majorOffset=0, timeFormat='%Y-%m-%d', 
                          minor=False, minorStep=5, minorOffset=0):
    '''
    设置横轴时间轴的格式，与FixedFormatter相配合
    @param:
        ax: 当前图对应的axes
        times: 日期列表，要求可迭代，且列表中元素为对应的日期时间类型，原则上应该
               为datetime.datetime类型，但是如果有strftime方法返回字符串也可
        majorStep: 主刻度之间的下标间隔，e.g., 5可以表示为[0, 5, 10,...]
        majorOffset: 根据该公式选择刻度，即(index-majorOffset)%majorStep==0，
                     默认为1
        timeFormat: 时间解析的形式，按照datetime要求的形式解析，默认为解析为
                    yyyy-mm-dd的日期形式
        minor: 是否设置次刻度，默认为False，即不设置
        minorStep: 同majorStep，默认为5
        minorOffset: 同majorOffset，默认为1
    @return:
        res: 按照给定形式解析后的时间字符串列表
        同时设置刻度表示的日期形式
    '''
    strDates = list(map(lambda x: x.date().strftime(timeFormat), times))
    majorDates = [strDates[i] for i in range(len(strDates)) 
                  if (i-majorOffset)%majorStep==0]
    formatter = ticker.FixedFormatter(majorDates)
    ax.xaxis.set_major_formatter(formatter)
    if minor:
        minorDates = [strDates[i] for i in range(len(strDates))
                      if (i-minorOffset)%minorStep==0]
        minorFormatter = ticker.FixedFormatter(minorDates)
        ax.xaxis.set_minor_formatter(minorFormatter)
    return strDates

def plot(data, columnNames=None, timeColumnName=None,
         majorLocatorStep=20, majorOffset=0, minorLocatorStep=5, minorOffset=0,
         displayMinor=False, timeFormat='%Y-%m-%d', rotation=45, 
         stickWidth=0.6, alpha=1):
    '''
    根据所给的数据，画出蜡烛图
    @param:
        data: 原始数据，要求为pandas.DataFrame的形式，默认的有open, close, high
              low这四列，反之需要通过columnNames参数提供，依次的顺序为openName,
              closeName, highName, lowName,若要画出时间轴，则需要数据中有date列，
              否则则需要通过使用的时候通过timeFormat来提供
        columnNames: 默认data中有open, close, high, low这些列，反之则需要自行提供
        timeColumnName: 当需要画以时间为横轴的图时，需要提供时间列的列名
        majorLocatorStep: 主刻度间隔，默认为间隔20个数据
        minorLocatorStep: 副刻度间隔，默认为间隔5个数据
        displayMinor: 是否在副刻度上也标明时间或者顺序下标，默认不标明
        majorFormat: 解析时间列主刻度的方法，需按照datetime中的形式解析，默认为%Y-%m-%d
        minorFormat: 解析时间列副刻度的方法，需按照datetime中的形式解析，默认为%Y-%m-%d
        rotation: 横轴值旋转度数，默认旋转45度
        stickWidth: 蜡烛图的宽度，默认为0.6
        alpha: 蜡烛图颜色的深度，默认为1
    '''
    plt.style.use('seaborn-ticks')
    fig, ax = plt.subplots()
    if not columnNames is None:
        openName, closeNames, highName, lowName = columnNames
    else:
        openName, closeNames, highName, lowName = ('open', 'close', 'high', 'low')
    opens, closes, highs, lows = data[openName].tolist(), data[closeNames].tolist(), data[highName].tolist(), data[lowName].tolist()
    fig.subplots_adjust(bottom=0.2)
    majorLocator = ticker.IndexLocator(majorLocatorStep, majorOffset)
    ax.xaxis.set_major_locator(majorLocator)
    minorLocator = ticker.IndexLocator(minorLocatorStep, minorOffset)
    ax.xaxis.set_minor_locator(minorLocator)

    if not timeColumnName is None:
        times = data[timeColumnName].tolist()
        time_formatterSetting(ax, times, majorLocatorStep, majorOffset, timeFormat,
                              displayMinor, minorLocatorStep, minorOffset)
    finance.candlestick2_ohlc(ax, opens, highs, lows, closes, width=stickWidth,
                              colorup='r', colordown='g', alpha=alpha)
    plt.setp(ax.get_xticklabels(), rotation=rotation)
    if displayMinor:
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=rotation)
    xmin, xmax = plt.xlim()
    plt.xlim(xmin=xmin-1)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    import pickle
    with open(r'F:\GeneralLib\test_sources\candleplot_testdata.pickle', 'rb') as f:
        testData = pickle.load(f)
    plot(testData, timeColumnName='time', majorLocatorStep=10, minorLocatorStep=1, displayMinor=True, rotation=90)
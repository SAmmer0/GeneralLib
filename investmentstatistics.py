# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 17:03:08 2016

@author: hao
修改日志：
修改日期：2016年10月20日
修改内容：
    1. 添加函数extendNetValue，用于将净值扩展到期间的交易日，即对应策略期间每一个
       交易日，都有一个净值（目前考虑删除 2016年11月4日）
    2. 添加计算sharp比率的函数

修改日期：2016年12月1日
修改内容：
    在ret_stats函数中加入代码，使函数能够返回数据个数

修改日期：2016年12月13日
修改内容：
    在ret_stats函数中加入计算盈亏比的代码
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


def max_drawn_down(netValues, columnName=None):
    '''
    计算净值序列的最大回撤：maxDrawndDown = max(1-D_i/D_j) i > j
    @param:
        netValues 净值序列，可以是pd.DataFrame,pd.Series和np.array,list类型，如果时pd.DataFrame的类型，
            则默认净值所在的列列名为netValue，如果为其他列名，则需要自行通过columnName参数提供
    @return:
        (maxDrawnDown, startTime(index), endTime(index))
    '''
    if isinstance(netValues, pd.DataFrame):
        if columnName is None:
            netValues = netValues['netValues']
        else:
            netValues = netValues[columnName]
    if isinstance(netValues, pd.Series):
        netValues = netValues.values
    assert len(netValues) > 0, ValueError('length of netValues should be greater than 0')
    localMax = netValues[0]
    startIdx = 0
    endIdx = 0
    period = [startIdx, endIdx]
    minRatio = float('inf')
    for idx, d in enumerate(netValues):
        if d > localMax:
            localMax = d
            startIdx = idx
        try:
            currentRatio = d/localMax
        except ZeroDivisionError:
            continue
        if currentRatio < minRatio:
            minRatio = currentRatio
            endIdx = idx
            period = [startIdx, endIdx]
    return (1 - minRatio, period[0], period[1])


def ret_stats(retValues, columnName=None, displayHist=False):
    '''
    计算策略交易收益的统计数据，包含胜率、均值、中位数、最大值、最小值、峰度、偏度
    @param:
        retValues 收益率序列数据，要求为序列或者pd.DataFrame或者pd.Series类型
        columnName 若提供的数据类型为pd.DataFrame，默认为None表明retValues中
                   有retValues这列数据，否则则需要通过columnName来传入
        displayHist 若为True，则依照收益率序列画出直方图
    @return:
        [winProb, retMean, retMed, retMax, retMin, retKurtosis, retSkew]
    '''
    if not (isinstance(retValues, pd.DataFrame) or isinstance(retValues, pd.Series)):
        retValues = pd.Series(retValues)
    if isinstance(retValues, pd.DataFrame):
        if 'retValues' in retValues.columns:
            retValues = pd.Series(retValues['retValues'].values)
        else:
            if columnName is None:
                raise KeyError('optional parameter \'columnName\' should be provided by user')
            else:
                retValues = pd.Series(retValues[columnName].values)
    winProb = np.sum(retValues > 0)/len(retValues)
    count = len(retValues)
    if displayHist:
        plt.hist(retValues, bins=int(len(retValues/30)))
    plRatio = (retValues[retValues > 0].sum()/abs(retValues[retValues <= 0].sum())
               if np.sum(retValues < 0) > 0 else float('inf'))
    return {'winProb': winProb, 'PLRatio': plRatio, 'mean': retValues.mean(),
            'median': retValues.median(), 'max': retValues.max(),
            'min': retValues.min(), 'kurtosis': retValues.kurtosis(),
            'skew': retValues.skew(), 'count': count}


def info_ratio(retValues, columnName=None, benchMark=.0):
    '''
    计算策略的信息比率：
        info_ratio = mean(ret - benchMark)/std(ret-benchMark)
    @param:
        retValues 收益率序列数据，要求为序列或者pd.DataFrame或者pd.Series类型
        columnName 若提供的数据类型为pd.DataFrame，默认为None表明retValues中
                   有retValues这列数据，否则则需要通过columnName来传入
        benckMark 基准收益率，默认为0
    @return:
        infoRatio 即根据上述公式计算出的信息比率
    '''
    if not (isinstance(retValues, pd.DataFrame) or isinstance(retValues, pd.Series)):
        retValues = pd.Series(retValues)
    if not isinstance(benchMark, (float, int)):
        assert hasattr(benchMark, '__len__'), ValueError(
            'given benchMark should be series object, eg: list, pd.DataFrame, etc...')
        assert len(benchMark) == len(retValues), ValueError(
            'given benchMark should have the same length as retValues')
    else:
        benchMark = [benchMark]*len(retValues)
    if isinstance(retValues, pd.DataFrame):
        if 'retValues' in retValues.columns:
            retValues = retValues['retValues']
        else:
            if columnName is None:
                raise KeyError('optional parameter \'columnName\' should be provided by user')
            else:
                retValues = retValues[columnName]
    excessRet = retValues - np.array(benchMark)
    return np.mean(excessRet)/np.std(excessRet)


def sharp_ratio(retValues, columnName=None, riskFreeRate=.0):
    '''
    计算策略的夏普比率：
        sharp_ratio = (mean(ret) - riskFreeRate)/std(ret)
    注：要求ret与riskFreeRate的频率是相同的，比如都为年化的；由于夏普比率是信息比率的一种特殊情况，因此该函数通过调用信息比率函数计算
    @param:
        retValues 收益率序列数据，要求为序列或者pd.DataFrame或者pd.Series类型
        columnName 若提供的数据类型为pd.DataFrame，默认为None表明retValues中有retValues这列数据，否则需要通过columnName来传入
        riskFreeRate 无风险利率，默认为.0
    @return:
        sharpRatio 即根据上述公式计算的夏普比率
    '''
    return info_ratio(retValues, columnName, riskFreeRate)

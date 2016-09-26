# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 17:03:08 2016

@author: hao
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def max_drawn_down(netValues, columnName=None):
    '''
    计算净值序列的最大回撤：maxDrawndDown = max(1-D_i/D_j) i > j
    @param: netValues 净值序列，可以是pd.DataFrame,pd.Series和np.array,list类型
    @return: (maxDrawnDown, startTime(index), endTime(index))
    '''
    if isinstance(netValues, pd.DataFrame):
        if 'netValue' in netValues.columns:
            netValues = netValues['netValue'].values
        else:
            if columnName is None:
                raise KeyError('optional parameter \'columnName\' must be provided')
    if isinstance(netValues, pd.Series):
        netValues = netValues.values
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
    winProb = np.sum(retValues>0)/len(retValues)
    if displayHist:
        plt.hist(retValues, bins=int(len(retValues/30)))
    return {'winProb':winProb, 'mean': retValues.mean(), 
            'median': retValues.median(), 'max': retValues.max(),
            'min': retValues.min(), 'kurtosis': retValues.kurtosis(), 
            'skew': retValues.skew()}

def info_ratio(retValues, columnName=None, benchMark=0):
    '''
    计算策略的信息比率：
        info_ratio = (mean(ret) - banckMark)/std(ret)
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
    if isinstance(retValues, pd.DataFrame):
        if 'retValues' in retValues.columns:
            retValues = pd.Series(retValues['retValues'].values)
        else:
            if columnName is None:
                raise KeyError('optional parameter \'columnName\' should be provided by user')
            else:
                retValues = pd.Series(retValues[columnName].values)
    return (np.mean(retValues)-benchMark)/np.std(retValues)
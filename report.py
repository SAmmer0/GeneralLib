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

__version__ = 1.1
修改日期：2017年3月29日
修改内容：
    1. 重构了最大回撤函数
    2. 重构了信息比率和夏普比率计算函数
    3. 添加了计算简单beta和简单alpha的函数
    4. 添加了DataFrame转化为html表格的类

__version__ = 1.2
修改日期：2017-04-06
修改内容：
    1. 在HTMLTable中添加clear函数
    2. 将部分函数的参数由收益率序列改为净值序列
    3. 添加将净值序列转换为收益率序列的函数
"""
__version__ = 1.2
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datatoolkits


def cal_ret(net_value, period=1):
    '''
    将净值数据转换为收益率数据
    @param:
        net_value: 净值数据序列，要求为pd.Series类型
    @return:
        转换后的收益率序列，为pd.Series类型
    '''
    return net_value.pct_change(period).dropna()


def max_drawn_down(netValues, columnName=None):
    '''
    计算净值序列的最大回撤：maxDrawndDown = max(1-D_i/D_j) i > j
    @param:
        netValues 净值序列，可以是pd.DataFrame,pd.Series和np.array,list类型，如果时pd.DataFrame的类型，
            则默认净值所在的列列名为netValue，如果为其他列名，则需要自行通过columnName参数提供
    @return:
        (maxDrawnDown, startTime(or index), endTime(or index))
    '''
    if isinstance(netValues, list):
        nav = pd.Series(netValues)
    elif isinstance(netValues, pd.DataFrame):
        if columnName is None:
            nav = netValues['netValue']
        else:
            nav = netValues[columnName]
    else:
        nav = netValues
    cumMax = nav.cummax()
    dd = 1 - nav / cumMax
    mdd = dd.max()
    mddEndTime = dd.idxmax()
    mddStartTime = nav[nav == cumMax[mddEndTime]].index[0]
    return mdd, mddStartTime, mddEndTime


def ret_stats(net_value, columnName=None, displayHist=False):
    '''
    计算策略交易收益的统计数据，包含胜率、均值、中位数、最大值、最小值、峰度、偏度
    @param:
        net_value 净值序列数据，要求为序列或者pd.DataFrame或者pd.Series类型
        columnName 若提供的数据类型为pd.DataFrame，默认为None表明retValues中
                   有net_value这列数据，否则则需要通过columnName来传入
        displayHist 若为True，则依照收益率序列画出直方图
    @return:
        [winProb, retMean, retMed, retMax, retMin, retKurtosis, retSkew]
    '''
    if not (isinstance(net_value, (pd.DataFrame, pd.Series))):
        net_value = pd.Series(net_value)
    if isinstance(net_value, pd.DataFrame):
        if 'net_value' in net_value.columns:
            net_value = pd.Series(net_value['net_value'].values)
        else:
            if columnName is None:
                raise KeyError('optional parameter \'columnName\' should be provided by user')
            else:
                net_value = pd.Series(net_value[columnName].values)
    retValues = cal_ret(net_value)
    winProb = np.sum(retValues > 0) / len(retValues)
    count = len(retValues)
    if displayHist:
        plt.hist(retValues, bins=int(len(retValues / 30)))
    plRatio = (retValues[retValues > 0].sum() / abs(retValues[retValues <= 0].sum())
               if np.sum(retValues < 0) > 0 else float('inf'))
    return pd.Series({'winProb': winProb, 'PLRatio': plRatio, 'mean': retValues.mean(),
                      'median': retValues.median(), 'max': retValues.max(),
                      'min': retValues.min(), 'kurtosis': retValues.kurtosis(),
                      'skew': retValues.skew(), 'count': count})


def info_ratio(retValues, retFreq, benchMark=.0, columnName=None):
    '''
    计算策略的信息比率：
        info_ratio = annualized_ret(ret - benchMark)/annualized_std(ret-benchMark)
    @param:
        retValues 收益率序列数据，要求为序列或者pd.DataFrame或者pd.Series类型
        retFreq: 收益率数据的转化为年化的频率，例如，月度数据对应12，年度数据对应250
        benckMark: 基准收益率，默认为0，可以为DataFrame、Series、list和数值的形式，要求如果为DataFrame
            或者Series形式时，其index应该与收益率序列相同
        columnName 若提供的数据类型为pd.DataFrame，默认为None表明retValues中
                   有retValues这列数据，否则则需要通过columnName来传入
    @return:
        infoRatio 即根据上述公式计算出的信息比率
    '''
    if not isinstance(retValues, (pd.DataFrame, pd.Series)):
        retValues = pd.Series(retValues)
    if isinstance(retValues, pd.DataFrame):
        if 'retValues' in retValues.columns:
            retValues = retValues['retValues']
        else:
            if columnName is None:
                raise KeyError('optional parameter \'columnName\' should be provided by user')
            else:
                retValues = retValues[columnName]
    if not isinstance(benchMark, (float, int)):
        assert hasattr(benchMark, '__len__'), ValueError(
            'given benchMark should be series object, eg: list, pd.DataFrame, etc...')
        assert len(benchMark) == len(retValues), ValueError(
            'given benchMark should have the same length as retValues')
        if isinstance(benchMark, list):
            benchMark = pd.Series(benchMark, index=retValues.index)
    else:   # 现将基准转化为年化
        benchMark = pd.Series([datatoolkits.retfreq_trans(benchMark, 1 / retFreq)] * len(retValues),
                              index=retValues.index)
    excessRet = retValues - benchMark
    annualized_ret = datatoolkits.annualize_ret(excessRet, retFreq)
    annualized_std = datatoolkits.annualize_std(excessRet, retFreq)
    return annualized_ret / annualized_std


def sharp_ratio(retValues, retFreq, riskFreeRate=.0, columnName=None):
    '''
    计算策略的夏普比率：
        sharp_ratio = annualiezd_ret(ret - riskFreeRate)/annualized_std(ret)
    注：要求ret与riskFreeRate的频率是相同的，比如都为年化的；由于夏普比率是信息比率的一种特殊情况，
        因此该函数通过调用信息比率函数计算
    @param:
        retValues 收益率序列数据，要求为序列或者pd.DataFrame或者pd.Series类型
        retFreq: 收益率数据的转化为年化的频率，例如，月度数据对应12，年度数据对应250
        riskFreeRate: 无风险利率，默认为.0
        columnName: 若提供的数据类型为pd.DataFrame，默认为None表明retValues中有retValues这列数据，
            否则需要通过columnName来传入
    @return:
        sharpRatio 即根据上述公式计算的夏普比率
    '''
    return info_ratio(retValues, retFreq, riskFreeRate, columnName)


def cal_rawbeta(net_value, benchMark):
    '''
    计算收益的简单beta
    计算方法如下：
        beta = cov(rets, benchMark)/var(benchMark)
    @param:
        net_value: 需要评估的净值序列，要求为pd.Series格式
        benchMark: 基准净值序列，要求为pd.Series格式
    @return:
        beta值
    '''
    rets = cal_ret(net_value)
    benchMark = cal_ret(benchMark)
    return rets.cov(benchMark) / benchMark.var()


def cal_rawalpha(net_value, benchMark, freq, riskFreeRate=0.):
    '''
    计算收益的简单alpha
    计算方法如下：
        alpha = annualized_ret - riskFreeRate - beta*(annualized_benchMark - riskFreeRate)
    @param:
        net_value: 净值序列，要求为pd.Series格式
        benchMark: 净值序列，要求为pd.Series格式
        freq: 转换频率，例如，日收益率序列对应250，月收益率序列对应12
        riskFreeRate: 无风险利率，年化
    @return:
        alpha值
    '''
    rets = cal_ret(net_value)
    benchmark_ret = cal_ret(benchMark)
    annualized_ret = datatoolkits.annualize_ret(rets, freq)
    annualized_benchMark = datatoolkits.annualize_ret(benchmark_ret, freq)
    beta = cal_rawbeta(net_value, benchMark)
    return annualized_ret - riskFreeRate - beta * (annualized_benchMark - riskFreeRate)


def brief_report(net_value, benchmark, riskfree_rate, freq):
    '''
    输入策略的净值等数据，对策略的基本状况做一个简报
    内容包含:
        粗略alpha，粗略beta，最大回撤，最大回撤起始期，夏普比率，信息比率
    @param:
        net_value: 策略净值数据，要求为pd.Series格式
        benchmark: 对比基准净值数据，要求为pd.Series格式
        riskfree_rate: 无风险利率
        freq: 数据的频率，例如日净值数据对应250，月净值数据对应12
    '''
    assert len(net_value) == len(
        benchmark), '\'net_value\' should have the same length as \'benchmark\''
    nav_ret = cal_ret(net_value)
    benchmark_ret = cal_ret(benchmark)
    alpha = cal_rawalpha(net_value, benchmark, freq, riskfree_rate)
    beta = cal_rawbeta(net_value, benchmark)
    mdd, mdd_start, mdd_end = max_drawn_down(net_value)
    sharp = sharp_ratio(nav_ret, freq, riskFreeRate=riskfree_rate)
    info = info_ratio(nav_ret, freq, benchMark=benchmark_ret)
    return pd.Series({'alpha': alpha, 'beta': beta, 'mdd': mdd, 'mdd_start': mdd_start,
                      'mdd_end': mdd_end, 'sharp_ratio': sharp, 'info_ratio': info})

# 方便写Markdown报告的表格工具


class HTMLTable(object):

    '''
    使用方法：
        使用该模块所提供的table_convertor对象，调用foramt_df函数，其中函数中的formater参数
        可以依靠ColFormater类的对象formater来生成
        例如：
        test_data = datatoolkits.load_pickle(r"F:\GeneralLib\CONST_DATAS\htmltable.pickle")
        res = table_convertor.format_df(test_data.reset_index(),
                                        formater={'nav': formater.get_modformater('pctnp', 3),
                                                  'CSI700': formater.get_modformater('pctnp', 2)},
                                        order=['nav', 'index', 'CSI700'])
    '''

    def __init__(self):
        self.table = '<table>\n{content}</table>'
        self.row = '<tr>{content}</tr>'
        self.col = '<td>{content}</td>'
        self.row_lists = list()

    def clear(self):
        self.table = '<table>\n{content}</table>'
        self.row = '<tr>{content}</tr>'
        self.col = '<td>{content}</td>'
        self.row_lists = list()

    def _format_col(self, col_content):
        return self.col.format(content=col_content)

    def _format_row(self, row_contents):
        content = ''
        for c in row_contents:
            content += self._format_col(c)
        res = self.row.format(content=content)
        res += '\n'
        return res

    def add_header(self, header):
        '''
        添加首行标题
        @param:
            contents: 需要添加的数据，要求为可迭代形式
        '''
        row = self._format_row(header)
        self.row_lists.insert(0, row)

    def add_row(self, content):
        '''
        向行添加内容
        '''
        row = self._format_row(content)
        self.row_lists.append(row)

    def _trans_df(self, df, formater=None, order=None):
        '''
        将DataFrame的内容转换为字符串的形式
        @param:
            df: 需要转换的DataFrame
            formater: 默认为None，即对于任何数据，使用str函数转化，可以自行提供，要求为
                字典形式{col: format_func}，目前只支持根据列自定义转化方式
            order: 数据列的顺序，默认为None，即使用DataFrame的默认数据列顺序，如果提供，则需要
                将所有需要生成报表的列都写出
        @return:
            转化后的DataFrame，内容全部为字符串形式，index部分会被忽视
        '''
        new_df = df.copy()
        if formater is None:
            formater = dict()
        for col in new_df.columns:  # 设置其他没有提供formater的列的格式化的方法
            formater.setdefault(col, str)
        if order is not None:   # 改变顺序
            new_df = new_df.loc[:, order]
        for col in new_df:
            new_df.loc[:, col] = new_df[col].apply(formater[col])
        return new_df

    def format_df(self, df, formater=None, order=None):
        '''
        将DataFrame转化为html
        @param:
            df: 需要转化的DataFrame
            formater: 提供的formater方法，默认为None，可以自定义，要求必须为字典形式{col: format_func}，
                目前只支持根据列自定义转化方式
            order: 数据列的顺序，默认为None，即使用DataFrame的默认数据列顺序，如果提供，则需要
                将所有需要生成报表的列都写出
        '''
        df_str = self._trans_df(df, formater, order)
        self.add_header(df_str.columns)
        for idx in range(len(df_str)):
            self.add_row(df_str.iloc[idx].tolist())
        res_str = ''.join(self.row_lists)
        res_str = self.table.format(content=res_str)
        return res_str


class ColFormater(object):

    def __init__(self):
        # 目前提供的便捷formater
        self.formaters = {'date': lambda x: x.strftime('%Y-%m-%d'),
                          'float2p': lambda x: '%.2f' % x,
                          'pct2p': lambda x: '%.2f%%' % (100 * x)}
        # 提供的一些基础formater的原型，可以用于自定义
        self.prototypes = {'date': lambda y: lambda x: x.strftime(y),
                           'floatnp': lambda y: lambda x: '%.{}f'.format(y) % x,
                           'pctnp': lambda y: lambda x: '%.{}f%%'.format(y) % (100 * x)}

    def get_basicformater(self, formater_type):
        assert formater_type in self.formaters, 'Error, valid formaters are {}'.\
            format(list(self.formaters.keys()))
        return self.formaters[formater_type]

    def get_modformater(self, formater_type, param):
        '''
        用于使用基础formater原型生成自定义的formater
        '''
        assert formater_type in self.prototypes, 'Error, valid formaters are {}'.\
            format(list(self.formaters.keys()))
        return self.prototypes[formater_type](param)


formater = ColFormater()
table_convertor = HTMLTable()

if __name__ == '__main__':
    test_data = datatoolkits.load_pickle(r"F:\GeneralLib\CONST_DATAS\htmltable.pickle")
    res = table_convertor.format_df(test_data.reset_index(),
                                    formater={'nav': formater.get_modformater('pctnp', 3),
                                              'CSI700': formater.get_modformater('pctnp', 2)},
                                    order=['nav', 'index', 'CSI700'])

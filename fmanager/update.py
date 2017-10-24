#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-07-19 13:08:23
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$

'''
用于更新因子数据

__version__ = 1.0.0
修改日期：2017-07-19
修改内容：
    初始化，添加基本功能
'''
__version__ = '1.0.0'

from collections import deque
from fmanager.const import UNIVERSE_FILE_PATH, START_TIME, FACTOR_FILE_PATH
from fmanager import database
import datatoolkits
import dateshandle
import datetime as dt
from fmanager.factors.dictionary import get_factor_dict, update_factordict
import fdgetter
import logging
from os.path import exists
from os import makedirs, system
import pdb
import time

# 日志设置
logger = logging.getLogger(__name__.split('.')[0])
file_handle = logging.FileHandler(FACTOR_FILE_PATH + '\\update_log.log')
file_handle.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S')
file_handle.setFormatter(formatter)
logger.addHandler(file_handle)


def update_universe(path=UNIVERSE_FILE_PATH):
    '''
    获取最新的universe，并将最新的universe与之前文件中的universe对比，如果发生了更新，打印相关信息
    随后，将最新的universe存储在指定文件中，存储文件为一个tuple(universe, update_time)

    Parameter
    ---------
    path: str, default UNIVERSE_FILE_PATH
        存储universe数据的文件

    Return
    ------
    universe: list
        当前最新的universe

    Notes
    -----
    不能自行调用该函数用于获取universe，可能造成获取的universe与因子数据的universe不一致，
    获取当前的universe，使用fmanger.factors.utils.get_universe函数
    '''
    new_universe = fdgetter.get_db_data(fdgetter.BASIC_SQLs['A_UNIVERSE'], cols=('code', ),
                                        add_stockcode=False)
    new_universe['code'] = new_universe.code.apply(datatoolkits.add_suffix)
    new_universe = new_universe.code.tolist()
    try:
        universe_save = datatoolkits.load_pickle(path)
        universe, _ = universe_save
        nu_set = set(new_universe)
        ou_set = set(universe)
        if nu_set != ou_set:
            add_diff = list(nu_set.difference(ou_set))
            minus_diff = list(ou_set.difference(nu_set))
            msg = 'Warning: universe UPDATED, {drop} are DROPED, {add} are ADDED'.\
                format(drop=minus_diff, add=add_diff)
            logger.info(msg)
            print(msg)
    except FileNotFoundError:
        pass
    data = (new_universe, dt.datetime.now())
    datatoolkits.dump_pickle(data, path)
    return new_universe


def get_endtime(t, threshold=18):
    '''
    根据给定的时间计算对应的结束时间，该功能用于确定当前更新数据时最新的数据时间

    Parameter
    ---------
    t: dt.datetime or the like
        参考转换的时间
    threshold: int(0, 23], default 18
        每天更新的时间节点，即如果t的时间在18点之前（<18），则返回的时间为昨天（上一天），反之返回的时间为今天

    Return
    ------
    out: dt.datetime or the like
    '''
    if t.hour < 18:
        out = t - dt.timedelta(1)
    else:
        out = t
    return out


def is_updated(path):
    '''
    检查数据是否最新，检查方法为将数据的日期与当前日期最近的交易日对比

    Parameter
    ---------
    path: str
        因子数据存储文件的路径

    Return
    ------
    out: boolean
        True表示为最新，False表示非最新

    Notes
    -----
    “最新”的定义：如果当前时间为17点之前，则以上个交易日对应的数据为最新数据（因为从收市到数据库收录
    相关数据，到同步到本地的数据库中间需要些时间，有的数据更新时间甚至在7点左右），17点以后以当前交易
    日为最新；非最新的情况除了包含数据的时间戳非最新外，还包含数据文件为空的情况
    '''
    try:
        connector = database.DBConnector(path)
        if connector.data_time is None:
            return False
        data_time = connector.data_time
        now = get_endtime(dt.datetime.now())
        rct_td = dateshandle.get_recent_td(now)
        return rct_td.date() == data_time.date()
    except OSError:  # 表示当前没有对应的文件
        return False


def check_dependency(factor_name, factor_dict):
    '''
    检查当前因子依赖的因子是否更新完成

    Parameter
    ---------
    factor_name: str
        需要检查的因子的名称
    factor_dict:
        因子字典

    Return
    ------
    out: boolean
        若因子的依赖因子全部都更新好，则返回True，反之返回False
    '''
    dependency = factor_dict[factor_name]['factor'].dependency
    if dependency is None:
        return True
    res = list()
    for factor in dependency:
        path = factor_dict[factor]['abs_path']
        res.append(is_updated(path))
    return all(res)


def update_factor(factor_name, factor_dict, universe):
    '''
    更新数据，如果数据文件不存在，则创建一个数据文件，并写入数据，数据的时间从START_TIME开始，到当前
    时间为止
    Parameter
    ---------
    factor_name: str
        需要更新的因子名称
    factor_dict: dict
        因子字典
    universe: list
        股票universe
    Return
    ------
    out: boolean
        如果更新成功，返回True，反之返回False
    '''
    assert factor_name in factor_dict, 'Error, invalid factor name({name})!'.format(
        name=factor_name)
    if not check_dependency(factor_name, factor_dict):  # 检查因子依赖是否满足
        return False
    factor_msg = factor_dict[factor_name]
    abs_path = factor_msg['abs_path']
    if exists(abs_path) and is_updated(abs_path):  # 当前已经是最新，不用取数据更新
        return True
    connector = database.DBConnector(factor_msg['abs_path'])
    start_time = None   # 更新的起始时间
    now = dt.datetime.now()
    end_time = get_endtime(now)
    # if now.hour > 17:    # 18点之前将昨天的数据视为最新
    #     end_time = now    # 更新的截止时间
    # else:
    #     end_time = now - dt.timedelta(1)
    if not exists(factor_msg['abs_path']):  # 检查文件是否存在
        connector.init_dbfile(factor_msg['factor'].data_type)
        start_time = START_TIME     # 数据文件初始化，从最早的时间开始
    else:
        if connector.data_time is not None:
            start_time = connector.data_time
        else:
            start_time = START_TIME
    factor_func = factor_msg['factor'].calc_method
    try:
        factor_data = factor_func(universe, start_time, end_time)
    except AssertionError as e:  # 如果出现意外的错误，将该错误写入到日志中，并返回True，进行下一个更新
        logger.exception(e)
        return True
    # pdb.set_trace()
    connector.insert_df(factor_data, data_dtype=factor_msg['factor'].data_type)
    return True


def update_all_factors(factor_dict, max_iter=300, order=None, show_progress=False):
    '''
    更新所有因子的数据

    Parameter
    ---------
    factor_dict: dict
        因子字典
    max_iter: int, default 300
        最大循环次数，超过这个次数更新过程被强制中断
    order: list, default None
        因子更新顺序，目前不实现对应功能，供未来扩展用（未来需要根据因子的依赖关系，解析更新顺序）
    show_progress: boolean, default False
        显示进度，默认不显示

    Return
    ------
    out: boolean
        如果成功更新所有因子，返回True，反之返回False
    '''
    if order is not None:
        raise NotImplementedError
    iter_num = 0
    factor_queue = deque(sorted(factor_dict.keys()), maxlen=len(factor_dict))
    universe = update_universe()
    while len(factor_queue):
        if iter_num > max_iter:
            break
        iter_num += 1   # 更新循环次数
        factor_name = factor_queue.pop()
        msg = 'Iter Num: {iter_num}, Factor Name: {name},'.format(iter_num=iter_num,
                                                                  name=factor_name)
        logger.info(msg)
        if show_progress:
            print(msg)
        update_res = update_factor(factor_name, factor_dict, universe)
        if not update_res:  # 未成功更新
            factor_queue.appendleft(factor_name)
            # 日志中添加添加队列的操作提示
            queue_msg = "Append \"{fct}\" to the left of the queue".format(fct=factor_name)
            logger.info(queue_msg)
        update_res_str = 'success' if update_res else 'fail'
        res_msg = 'Result: {res}'.format(res=update_res_str)
        logger.info(res_msg)
        if show_progress:
            print(res_msg)
    else:
        return True
    return False


def auto_update_all(max_iter=200, show_progress=False):
    '''
    自动化更新所有因子，并更新因子字典

    Parameter
    ---------
    max_iter: int, default 200
        最大循环次数，超过这个次数更新过程被强制中断
    show_progress: boolean, default False
        显示更新进度，默认为不显示
    '''
    all_factors = get_factor_dict()
    gen_folders(all_factors)
    update_factordict()  # 每次更新前先更新因子字典
    success = update_all_factors(all_factors, max_iter=max_iter, show_progress=show_progress)
    if not success:
        print('Updating process FAILED')
        logger.info('Updating process FAILED')


def gen_folders(fd):
    '''
    用于生成因子数据文件所在的文件夹

    Parameter
    ---------
    fd: dict
        因子字典数据
    '''
    path_list = [fd[name]['abs_path'] for name in fd]
    folder_list = ['\\'.join(p.split('\\')[:-1]) for p in path_list]
    folder_list = list(set(folder_list))
    for folder in folder_list:
        if exists(folder):
            continue
        makedirs(folder)


if __name__ == '__main__':
    logger.info('-' * 10 + 'START UPDATING' + '-' * 10)
    now = dt.datetime.now()
    if now.hour < 19:
        target_time = now.replace(hour=19, minute=0)
        seconds = (target_time - now).seconds
        logger.info("Wait for {second}s".format(second=seconds))
        time.sleep(seconds)
    try:
        auto_update_all(show_progress=True)
    except Exception as e:
        logger.exception(e)
        raise e
    finally:
        logger.info('-' * 10 + 'COMPLETE UPDATING' + '-' * 10)
        system('shutdown -s -t 100')

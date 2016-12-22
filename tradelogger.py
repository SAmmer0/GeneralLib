#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-10-31 10:12:57
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : ${link}
# @Version : 1.0

'''
tradelogger 用于记录交易中的操作或者数据
使用方法：
    首先需要申明一个Logger对象，可在对象中提供路径和文件名的信息（注：路径是指文件夹路径，不包含文件名）。
    若未在申明时提供，则需要在调用write_log函数前通过add_path函数提供路径，其中可以通过change_logfile
    函数改变存储日志的文件名，默认情况下采用log.txt为文件名。本程序是先将日志添加到缓存中，然后在程序调用完后
    写到文件中，因此，一方面每次需要写日志时，需要将Logger对象传入函数中；另一方面，还需要在所有的程序调用
    结束后，调用write_log函数，将缓存写入文件中。

example:
    logger = Logger('C:\\Users\\Default\\Desktop')
    some_func(params, logger)
        doing something
        logger.add_log('log message')
    ...
    logger.write_log()
'''
import os


class Logger(object):

    def __init__(self, dirPath=None, logFile='log.txt', pattern='a',
                 delimiter='\n', generatePath=True):
        self.dirPath = dirPath
        self.content = list()
        self.pattern = pattern
        self.delimiter = delimiter
        self.generatePath = generatePath
        self.logFile = logFile

    def add_path(self, path):
        self.dirPath = path

    def change_logfile(self, logFile):
        self.logFile = logFile

    def add_log(self, msg):
        self.content.append(msg)

    def write_log(self):
        try:
            if not os.path.exists(self.dirPath):
                if self.generatePath:
                    os.makedirs(self.dirPath)
                else:
                    raise ValueError('Path does not exist!')
            with open(self.dirPath+'\\'+self.logFile, self.pattern) as f:
                for msg in self.content:
                    f.write(msg + self.delimiter)
        except TypeError:
            raise ValueError('Path has not been provided!')

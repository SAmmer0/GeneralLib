#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-06 16:15:42
# @Author  : Li Hao (howardlee_h@outlook.com)
# @Link    : https://github.com/SAmmer0
# @Version : $Id$
'''
本模块主要用于行情数据的存储、提取和相关处理
当前将数据存储在mongoDB下，存储的结构如下：
mongoDB->
    quotes->    # 行情数据
        wind_code: data

需要实现的功能：
    将现有的pandas的数据存入数据库中，在存储过程中需要检查核对存入的数据与现有数据的列是否相同
    在存储过程中，还要检查当前存入的数据是否重复（行情数据中可以通过时间戳来标记）。添加wind接口，
    从Wind每日的行情中获取数据并存储。获取给定证券的行情数据，并转化为pd.DataFrame的格式

开发日志：
2017-03-06
    考虑对于db和collection分别建两个类，db负责处理数据库范围的事务，collection专门处理表的事务，
    collection的主键可有db来定义，collection只管按照主键和条件筛选值以及数据校验等工作，db负责
    其他方面的事务
'''

import pandas as pd
import pymongo


ID_COL = '_id'


class Error(Exception):

    '''
    异常类的主类
    '''
    pass


class CollectionNotFoundError(Error):

    '''
    当数据库的表中不包含collection时，返回该错误
    '''
    pass


class DatabaseNotFoundError(Error):
    pass


class DBHandle(object):

    '''
    对MongoDB进行包装，实现上述提到的功能
    对存储数据的要求较严格，需要校验每个记录是否具有相同的field，以及校验是否具有重复记录，鉴于
    该数据库主要用来记录行情等时间序列的数据，且每个证券单独存放一张表，故是使用时间列作为主键
    '''

    def __init__(self, db, server='localhost', port=27017):
        self._valid_db = None
        self._client = pymongo.MongoClient(server, port)
        self._validate_db(db)
        self._db = self._client[db]
        self._collection = None

    @property
    def db(self):
        return self._db

    @property
    def valid_db(self):
        if self._valid_db is None:
            sys_filter = set(['admin', 'local'])
            all_db = self._client.database_names()
            valid_db = set(all_db).difference(sys_filter)
            self._valid_db = list(valid_db)
            return self._valid_db
        else:
            return self.valid_db

    @property
    def valid_collections(self):
        if self._collection is None:
            self._collection = self._db.collection_names()
            return self._collection
        else:
            return self._collection

    def _validate_db(self, db):
        if db not in self.valid_db:
            error_msg = '{db} is not valid, the databases '.format(db=db) +\
                'available are {vdbs}'.format(vdbs=self.valid_db)
            raise DatabaseNotFoundError(error_msg)

    def validate_collection(self, collection):
        if collection not in self.valid_collections:
            error_msg = '{clect} not found, collections available '.format(clect=collection) +\
                'are {valid_clect}'.format(valid_clect=self.valid_collections)
            raise CollectionNotFoundError(error_msg)

    def list_field(self, collection):
        clect = CollectionHandle(self, collection)
        return clect.field

    def find_one(self, collection, date, col_name):
        '''
        用于从数据库中查找某个证券在某个时间上的数据

        @param:
            collection: 表名
            date: 可以是dt.datetime的格式，也可以是pd.to_datetime可以解析的字符串形式
            col_name: 时间列的列名（可以依靠list_cols函数获取有哪些）
        @return:
            如果有记录，则返回pd.Series，如果没有记录，则返回None
        '''
        if isinstance(date, str):
            date = pd.to_datetime(date)
        clect = CollectionHandle(self, collection, col_name)
        return clect.find_one(date)

    def find_many(self, collection, dates, col_name):
        '''
        用于从数据库中查找某个证券在一系列时间上的数据
        @param:
            collection: 表名，字符串形式
            dates: 可迭代形式，内容可以是dt.datetime的格式，也可以是pd.to_datetime可以解析的字符串形式
            col_name: 时间列的列名（可以依靠list_cols函数获取有哪些）
        @return:
            如果有记录，则返回pd.DataFrame，如果没有，则返回None
        '''
        if isinstance(dates[0], str):
            dates = [pd.to_datetime(d) for d in dates]
        clect = CollectionHandle(self, collection, col_name)
        return clect.find_many(dates)

    def insert(self, collection, data):
        '''
        用于向数据库中插入数据
        @param:
            collection: 表名，字符串形式
            data: 需要插入的数据，可以为字典类型，也可以为
        '''


class InvalidFieldError(Error):
    pass


class NonuniformFieldError(Error):
    pass


class InvalidInsertFieldError(Error):
    pass


class CollectionHandle(object):

    '''
    内部类，用来代表每个collection，一个collection为一张表，要求field相同，只能通过提供的
    主键获取对应的数据
    要求提供DBHandle对象，collection的名称（字符串），primary_key的名称（字符串）
    '''

    def __init__(self, db, collection, primary_key=None):
        self._db = db
        self._field = None
        self._db.validate_collection(collection)
        self._collection = self._db.db[collection]
        if primary_key is not None:
            self.isin_field(primary_key)
        self._primary_key = primary_key

    def set_primarykey(self, pk):
        self.isin_field(pk)
        self._primary_key = pk

    @property
    def field(self):
        if self._field is None:
            if self._collection.count() == 0:
                self._field = ()
                return self._field
            else:
                record_field = list(self._collection.find_one().keys())
                record_field = tuple(set(record_field).difference([ID_COL]))
                self._field = record_field
                return record_field
        else:
            return self._field

    def isin_field(self, key):
        if self._field is None:
            pass
        if key not in self.field:
            raise InvalidFieldError('{field} is not found, the fields '.format(field=key) +
                                    'available are {keys}'.format(keys=self.field))

    def check_insert_data(self, data):
        '''
        检查要插入的参数的类型是否合法，即插入数据的field与现有数据的field是否一致
        如果当前没有数据，则检查
        '''
        if isinstance(data, (tuple, list)):
            fields = list()
            for d in data:
                fields += list(d.keys())
            if self._field is None:     # 表明当前数据库中没有数据
                if set(fields) != set(data[0].keys()):
                    raise NonuniformFieldError('Fields in input data should be the same')
            elif set(fields) != set(self._field):
                raise InvalidInsertFieldError
        else:
            if self._field is None:
                continue
            elif set(data.values()) != set(self._field):
                raise InvalidInsertFieldError

    def find_one(self, value):
        '''
        依靠主键来查询值，如果没有查询到，返回None；如果有记录，返回pd.Series，且删除id
        '''
        data = self._collection.find_one({self._primary_key: value})
        if len(data) == 0:
            return None
        data = pd.Series(data).drop(ID_COL)
        return data

    def find_many(self, values):
        '''
        依靠主键来查询值，如果没有查询到，返回None；如果有记录，返回pd.DataFrame，且删除id
        '''
        data = list(self._collection.find({self._primary_key: {'$in': values}}))
        if len(data) == 0:
            return None
        data = pd.DataFrame(data)
        data = data.drop(ID_COL, axis=1)
        data = data.sort_values(self._primary_key).reset_index(drop=True)
        return data

    def insert_one(self, data):
        '''
        向collection中插入数据，要求插入数据的field与数据库中的数据field相同
        @param:
            data: 需要插入的数据，为字典形式
        '''
        self.check_insert_data(data)
        self._collection.insert_one(data)

    def insert_many(self, data):
        '''
        向collection中插入数据，要求插入数据的field与数据库中的数据field相同
        @param:
            data: 需要插入的数据，为可迭代的形式，元素为字典形式
        '''
        self.check_insert_data(data)
        self._collection.insert_many(data)

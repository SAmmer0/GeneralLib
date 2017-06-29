# -*-encoding:utf-8 -*-
import pyodbc


class SQLserver:

    def __init__(self, DATABASE, SERVER, UID, PWD, DRIVER='{SQL Server}'):
        self.database = DATABASE
        self.server = SERVER
        self.uid = UID
        self.pwd = PWD
        self.driver = DRIVER

    def connect(self):
        self.cur = self.__GetConnect()

    def __GetConnect(self):
        self.conn = pyodbc.connect(DATABASE=self.database, SERVER=self.server, UID=self.uid,
                                   PWD=self.pwd, DRIVER=self.driver)
        cur = self.conn.cursor()
        if not cur:
            raise ConnectionError
        else:
            return cur

    def fetchall(self, sql, *args, **kwargs):
        try:
            self.cur.execute(sql, *args, **kwargs)
        except pyodbc.Error as e:
            print(e)
            raise e
        rows = self.cur.fetchall()
        return rows

    def fetchone(self, sql, *args, **kwargs):
        try:
            self.cur.execute(sql, *args, **kwargs)
        except pyodbc.Error as e:
            print(e)
            raise e
        row = self.cur.fetchone()
        return row

    def close(self):
        self.conn.close()


class Error(Exception):
    pass


class ConnectionError(Error):

    def __init__(self):
        self.errorMsg = u'连接数据库失败'

    def __str__(self):
        return self.errorMsg


if __name__ == '__main__':
    pass

# -*-encoding:utf-8 -*-
import pyodbc


class SQLserver:

    def __init__(self, DATABASE, SERVER, UID, PWD, DRIVER='{SQL Server}'):
        self.database = DATABASE
        self.server = SERVER
        self.uid = UID
        self.pwd = PWD
        self.driver = DRIVER
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
    # SQL_dfcfzx = SQLserver(DATABASE='dfcf', SERVER='128.6.5.18', UID='dfcfzx', PWD='dfcfzx')
    # SQL_zyyx = SQLserver(DATABASE='zyyx', SERVER='128.6.5.18', UID='zyyx', PWD='zyyx')
    SQL_jydb = SQLserver(DATABASE='jydb', SERVER='128.6.5.18', UID='jydb', PWD='jydb')
    # test_dfcfzx = SQL_dfcfzx.ExecQuery ("""select top 100 INFOCODE , RELATECODE , RELATENAME ,
    # RELATEVARIETYCODE , INFOTYPE , NEWSCOLUMN ,  MKTPOSTFIX
    # from INFO_AN_NEWSRELATION
    # where NEWSCOLUMN is not null """)
    # test_zyyx = SQL_zyyx.ExecQuery("select * from I_report_type ")
    test_jydb = SQL_jydb.fetchall(
        "select * from QT_TradingDayNew where IfTradingDay=1 and TradingDate between ? and ? ", '2015-01-01', '2015-02-01')
    # print(test_dfcfzx[0:50])
    # print(test_zyyx[0])
    print(test_jydb[0])
# 构造DataFrame：pd.DataFrame(np.array(test_dfcfzx),columns=['INFOCODE' ,
# 'RELATECODE' , 'RELATENAME' ,'RELATEVARIETYCODE' , 'INFOTYPE' ,
# 'NEWSCOLUMN' ,  'MKTPOSTFIX'])

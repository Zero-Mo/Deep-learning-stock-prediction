
import os
import baostock as bs
import tushare as ts
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# 设置数据下载时间段
data_start_date = '2019-12-31'
data_end_date = '2021-06-26'
train_data_start_date = '2015-01-01'
train_data_end_date = '2019-12-31'
base_data_path = './data/'
data_path = './data/stocks/'
train_data_path = './data/train_data/'

# 使用tushare的taken码（vip版）
apikey = ''
if os.path.exists('tushare_apikey.txt'):
    with open('tushare_apikey.txt', 'r') as f:
        apikey = f.read()
pro = ts.pro_api(apikey)

download_stocks = True
download_indexes = True
fields = "date,code,open,high,low,close,preclose,volume," \
         "amount,turn,peTTM,psTTM,pcfNcfTTM,pbMRQ"
ts_basic_fields = "trade_date,total_share,float_share,free_share,total_mv,circ_mv"

lg = bs.login()
if lg.error_code != '0':
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)


def status(raw):
    if raw.error_code != '0':
        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)


def data_load(raw):
    data = []
    while (raw.error_code == '0') & raw.next():
        data.append(raw.get_row_data())
    df = pd.DataFrame(data, columns=raw.fields)
    return df


def ts_c(bs_stock_code: str) -> str:
    return bs_stock_code[3:]+'.'+bs_stock_code[0:2].upper()


def ts_d(date):
    return date.replace('-', '')


def bs_d(ts_date_series):
    date_series = []
    for ts_date in ts_date_series:
        date = str(ts_date)
        date_series.append(date[:4] + '-' + date[4:6] + '-' + date[6:])
    return date_series


if download_stocks:
    rs = bs.query_hs300_stocks()
    result = data_load(rs)
    result.to_csv(base_data_path+'hs300_stocks.csv')
    for code in tqdm(result['code']):
        for start_date, end_date, path in ((data_start_date, data_end_date, data_path),
                                           (train_data_start_date, train_data_end_date, train_data_path)):
            rs = bs.query_history_k_data_plus(code, fields,
                                              start_date=start_date, end_date=end_date,
                                              frequency="d", adjustflag="3")
            status(rs)

            df1 = data_load(rs)
            df1.set_index('date', inplace=True)

            df2 = pro.daily_basic(ts_code=ts_c(code), start_date=ts_d(start_date),
                                  end_date=ts_d(end_date), fields=ts_basic_fields)
            df2['trade_date'] = bs_d(df2['trade_date'])
            df2.set_index('trade_date', inplace=True)

            df1 = df1.join(df2)
            df1.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
            df1.fillna(0, inplace=True)

            df1.to_csv(path+code+'.csv')
            del df1, df2
        time.sleep(0.2)


if download_indexes:
    rs = bs.query_history_k_data_plus('sh.000300',
                                      "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                      start_date=data_start_date, end_date=data_end_date, frequency="d")
    status(rs)
    data_load(rs).to_csv(data_path+'sh.000300.csv')

bs.logout()

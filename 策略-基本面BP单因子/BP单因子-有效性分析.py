import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import  datetime,timedelta


np.set_printoptions(threshold=np.inf)  #设置阈值为无限
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#基本面单因子测试——以BP因子为例
#https://www.joinquant.com/view/community/detail/4de6d47826f94528e89dc957001ebb13

# 获取数据
import  jqdatasdk
from jqdatasdk import *
auth('15818677767','20200505Abc')

#获取指定周期的日期列表 'W、M、Q'
def get_period_date(period,start_date, end_date):
    #设定转换周期period_type  转换为周是'W',月'M',季度线'Q',五分钟'5min',12天'12D'
    stock_data = get_price('000300.XSHG', start_date, end_date, 'daily', fields=['close'])
#     print ('stock_data is :', stock_data)
#                close
# 2017 - 01 - 03  8.65
# 2017 - 01 - 04  8.65
# 2017 - 01 - 05  8.66
# 2017 - 01 - 06  8.62
    #记录每个周期中最后一个交易日
    stock_data['date']=stock_data.index
    #进行转换，周线的每个变量都等于那一周中最后一个交易日的变量值
    period_stock_data=stock_data.resample(period,how='last')
#     print ('period_stock_data is :', period_stock_data)
#                 close        date
# 2017 - 01 - 31   8.81   2017 - 01 - 26
# 2017 - 02 - 2    8.95   2017 - 02 - 28
# 2017 - 03 - 31   8.66   2017 - 03 - 31
    date=period_stock_data.index
    pydate_array = date.to_pydatetime()
    # print ('pydate_array is :', pydate_array)
    # pydate_array is: [datetime.datetime(2017, 1, 31, 0, 0) datetime.datetime(2017, 2, 28, 0, 0)
    #                   datetime.datetime(2017, 3, 31, 0, 0) datetime.datetime(2017, 4, 30, 0, 0)
    #                   datetime.datetime(2017, 5, 31, 0, 0) datetime.datetime(2017, 6, 30, 0, 0)
    date_only_array = np.vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array)
    date_only_series = pd.Series(date_only_array)
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    start_date=start_date-timedelta(days=1)
    start_date = start_date.strftime("%Y-%m-%d")
    date_list=date_only_series.values.tolist()
    date_list.insert(0,start_date)
    return date_list

#去除上市距beginDate不足1年且退市在endDate之后的股票（为退市）
def delect_stop(stocks,beginDate,endDate,n=365):
    stockList=[]
    beginDate = datetime.strptime(beginDate, "%Y-%m-%d")
    endDate = datetime.strptime(endDate, "%Y-%m-%d")
    for stock in stocks:
        start_date=get_security_info(stock).start_date
        end_date=get_security_info(stock).end_date
        if start_date<(beginDate-timedelta(days=n)).date() and end_date>endDate.date():
            stockList.append(stock)
    return stockList

#获取股票池
def get_stock(stockPool,start_date,end_date):
    if stockPool=='沪深300':
        stockList=get_index_stocks('000300.XSHG',start_date)
    elif stockPool=='中证500':
        stockList=get_index_stocks('000905.XSHG',start_date)
    elif stockPool=='中证1000':
        stockList=get_index_stocks('000852.XSHG',start_date)
    elif stockPool=='中证800':
        stockList=get_index_stocks('000906.XSHG',start_date)
    elif stockPool=='CYBZ':
        stockList=get_index_stocks('399006.XSHE',start_date)
    elif stockPool=='ZXBZ':
        stockList=get_index_stocks('399005.XSHE',start_date)
    elif stockPool=='A':
        stockList=get_index_stocks('000002.XSHG',start_date)+get_index_stocks('399107.XSHE',start_date)
    #剔除ST股
    st_data=get_extras('is_st',stockList, count = 1, end_date = start_date)
    #print ('st_data is:', st_data)
    # st_data is:  000001.XSHE  000002.XSHE...603885.XSHG 603993.XSHG
    # 2016 - 12 - 30 False     False...   False    False
    stockList = [stock for stock in stockList if not st_data[stock][0]]
    #剔除停牌、新股及退市股票
    stockList=delect_stop(stockList,start_date,end_date)
    return stockList


def get_all_data(start_date, end_date, stockPool, period):
    warnings.filterwarnings("ignore")

    # 获取日期数据
    date_period = get_period_date(period, start_date, end_date)
    # print ('date_period is:', date_period)
    # date_period is: ['2016-12-31', '2017-01-31', '2017-02-28', '2017-03-31', '2017-04-30', '2017-05-31', '2017-06-30',
    #                  '2017-07-31', '2017-08-31', '2017-09-30', '2017-10-31', '2017-11-30', '2017-12-31', '2018-01-31',
    #                  '2018-02-28', '2018-03-31', '2018-04-30', '2018-05-31', '2018-06-30', '2018-07-31', '2018-08-31',
    #                  '2018-09-30', '2018-10-31', '2018-11-30', '2018-12-31', '2019-01-31']

    # 获取申万一级行业数据
    indu_code = get_industries(name='sw_l1')
#     print ('indu_code is :', indu_code)
#          name      start_date
# 801740 国防军工I 2014 - 02 - 21
# 801020 采掘I     2004 - 02 - 10
# 801110 家用电器I 2004 - 02 - 10

    indu_code = list(indu_code.index)

    data = pd.DataFrame()

    for date in date_period[:-1]:
        # 获取股票列表
        stockList = get_stock(stockPool, date, end_date)  # 获取date日的成份股列表
        # print ('stockList is:', stockList)
        # ['000001.XSHE', '000002.XSHE', '000006.XSHE', '000008.XSHE', '000009.XSHE', '000012.XSHE', '000021.XSHE',
        #  '000025.XSHE', '000027.XSHE', '000028.XSHE', '000031.XSHE', '000039.XSHE', '000049.XSHE', '000050.XSHE',

         # 获取横截面收益率
        df_close = get_price(stockList, date, date_period[date_period.index(date) + 1], 'daily', ['close'])
        df_pchg = df_close['close'].iloc[-1, :] / df_close['close'].iloc[0, :] - 1

        # 获取权重数据，流通市值的平方根为权重
        q = query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(stockList))
        R_T = get_fundamentals(q, date)
        # print ('R_T is :', R_T)
        #    code     circulating_market_cap
        # 0 000001.XSHE   1331.4374
        # 1 000002.XSHE   1995.0161
        R_T.set_index('code', inplace=True, drop=False)
        R_T['Weight'] = np.sqrt(R_T['circulating_market_cap'])  # 流通市值的平方根作为权重
        # 删除无用的code列和circulating_market_cap列
        del R_T['code']
        del R_T['circulating_market_cap']

        # 中证800指数收益率
        index_close = get_price('000300.XSHG', date, date_period[date_period.index(date) + 1], 'daily', ['close'])
        index_pchg = index_close['close'].iloc[-1] / index_close['close'].iloc[0] - 1
        R_T['pchg'] = df_pchg - index_pchg  # 每支股票在date日对沪深300的超额收益率（Y）
        # 目前，R_T包含索引列code，权重列Weight，对沪深300的超额收益率pchg

        # 获取行业暴露度、哑变量矩阵
        Linear_Regression = pd.DataFrame()
        for i in indu_code:
            i_Constituent_Stocks = get_industry_stocks(i, date)
            i_Constituent_Stocks = list(set(i_Constituent_Stocks).intersection(set(stockList)))
            try:
                temp = pd.Series([1] * len(i_Constituent_Stocks), index=i_Constituent_Stocks)
                # print ('temp is :', temp)
                # 600316.XSHG  1
                # 601989.XSHG  1
                # 600435.XSHG  1
                # 600685.XSHG  1
                # 000738.XSHE  1
                # 600150.XSHG  1
                # 600184.XSHG  1
                #dataframe中的column:
                temp.name = i
            except:
                print(i)
            Linear_Regression = pd.concat([Linear_Regression, temp], axis=1)
            # print('Linear_Regression is :', Linear_Regression)
            #             801740  801020
            # 000006.XSHE   NaN     1.0
            # 000552.XSHE   NaN     1.0

        Linear_Regression.fillna(0.0, inplace=True)
        Linear_Regression = pd.concat([Linear_Regression, R_T], axis=1)
        Linear_Regression = Linear_Regression.dropna()
        Linear_Regression['date'] = date
        Linear_Regression['code'] = Linear_Regression.index
        data = data.append(Linear_Regression)
        #print (date + ' getted!!')
    return data

start_date = '2014-01-30'
end_date = '2021-02-26'
stockPool='沪深300'
period='M'
Group=10
factor = 'pb_ratio'#这个地方用了get_fundamentals里面有的因子数据，如果是别的数据，可以不写这行

#获取市值权重、行业哑变量数据
# data = get_all_data(start_date,end_date,stockPool,period)
# data.to_csv('data/get_all_data_new.csv')
data = pd.read_csv('data/get_all_data_new.csv')

from scipy import stats
from sklearn import preprocessing
from scipy.stats import mstats
#
# 获取新的一个因子数据并进行缩尾和标准化，因子一定要求是get_fundamentals里的
def get_factor_data(start_date, end_date, stockPool, period, factor):
    date_period = get_period_date(period, start_date, end_date)

    # 获取stockvaluaton格式的因子名
    # sheet = get_sheetname(factor)
    # str_factor = sheet + '.' + factor
    # str_factor = eval(str_factor)
    str_factor = eval('valuation.%s' % factor)

    factor_data = pd.DataFrame()

    for date in date_period[:-1]:
        # 获取股票列表
        stockList = get_stock(stockPool, date, end_date)  # 获取date日的成份股列表

        # 获取股票数据
        q = query(valuation.code, str_factor).filter(valuation.code.in_(stockList))
        temp = get_fundamentals(q, date)
        #print ('temp is:', temp)

        # 因子数据正态化
        temp[factor] = stats.boxcox(temp[factor])[0]

        # 生成日期列
        temp['date'] = date

        # 缩尾处理 置信区间95%
        temp[factor] = mstats.winsorize(temp[factor], limits=0.025)

        # 数据标准化
        temp[factor] = preprocessing.scale(temp[factor])

        factor_data = factor_data.append(temp)
        #print (date + ' getted!!')

    return factor_data

# #这部分为获取因子数据，如果因子数据为外部数据则可以忽略此步，导入你自己的因子数据即可
# factor_data = get_factor_data(start_date, end_date, stockPool, period,factor)
# factor_data.to_csv('data/get_factor_data_new.csv')
factor_data = pd.read_csv('data/get_factor_data_new.csv')

#将因子数据与权重、行业数据合并。如果获取的因子数据用的是自己的因子数据，则保证code和date列可以确定一行观测即可
result = pd.merge(data, factor_data, how = 'left', on = ['code','date'])
result = result.dropna()
#print (result.head(10))

import scipy.stats as st
import statsmodels.api as sm

# 有效性检验(t/IC)
def t_test(result, period, start_date, end_date, factor):
    # 获取申万一级行业数据
    indu_code = get_industries(name='sw_l1')
    indu_code = list(indu_code.index)

    # 生成空的dict，存储t检验、IC检验结果
    WLS_params = {}
    WLS_t_test = {}
    IC = {}

    date_period = get_period_date(period, start_date, end_date)

    for date in date_period[:-2]:
        temp = result[result['date'] == date]
        X = temp.loc[:, indu_code + [factor]]
        #print ('X is:', X)
        Y = temp['pchg']
        # WLS回归
        wls = sm.WLS(Y, X, weights=temp['Weight'])
        output = wls.fit()
        # 因子收益率
        WLS_params[date] = output.params[-1]
        WLS_t_test[date] = output.tvalues[-1]
        # IC检验
        IC[date] = st.pearsonr(Y, temp[factor])[0]
        # print (date + ' getted!!!')
    return WLS_params, WLS_t_test, IC

#t检验，IC检验
WLS_params,WLS_t_test,IC = t_test(result, period, start_date, end_date, factor)
WLS_params = pd.Series(WLS_params)
WLS_t_test = pd.Series(WLS_t_test)
IC = pd.Series(IC)

#t检验结果
n = [x for x in WLS_t_test.values if np.abs(x)>1.96]
print ('t值序列绝对值平均值——判断因子的显著性是否稳定',np.sum(np.abs(WLS_t_test.values))/len(WLS_t_test))
print ('t值序列绝对值大于1.96的占比——判断因子的显著性是否稳定',len(n)/float(len(WLS_t_test)))
#WLS_t_test.plot('bar',figsize=(20,8))

#IC检验结果
print ('IC 值序列的均值大小',IC.mean())
print ('IC 值序列的标准差',IC.std())
print ('IR 比率（IC值序列均值与标准差的比值）',IC.mean()/IC.std())
n_1 = [x for x in IC.values if x > 0]
print ('IC 值序列大于零的占比',len(n_1)/float(len(IC)))
n_2 = [x for x in IC.values if np.abs(x) > 0.02]
print ('IC 值序列绝对值大于0.02的占比',len(n_2)/float(len(IC)))
IC.plot('bar',figsize=(20,8))

plt.show()


# t值序列绝对值平均值——判断因子的显著性是否稳定 2.168120928408959
# t值序列绝对值大于1.96的占比——判断因子的显著性是否稳定 0.4588235294117647
# IC 值序列的均值大小 -0.008661842738773696
# IC 值序列的标准差 0.23841514633972996
# IR 比率（IC值序列均值与标准差的比值） -0.03633092474096001
# IC 值序列大于零的占比 0.47058823529411764
# IC 值序列绝对值大于0.02的占比 0.9647058823529412


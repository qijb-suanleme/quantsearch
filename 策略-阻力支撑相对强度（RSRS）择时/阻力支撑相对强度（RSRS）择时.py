#!/usr/bin/python
#coding:utf-8
import pandas as pd
import os
import datetime
import numpy as np
import statsmodels.formula.api as sml
import matplotlib.pyplot as plt
import tushare as ts
import scipy.stats as scs
import matplotlib.mlab as mlab
from pandas.testing import assert_frame_equal


np.set_printoptions(threshold=np.inf)  #设置阈值为无限
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


token=''
ts.set_token(token)
pro=ts.pro_api()

#这个函数的数据处理接近实战:
def getdata(stock, dateStart, dateEnd, N, M):
    #get_k_data() 接口 : 多留意：
    # HS300 = ts.get_k_data('000300', index=True, start='{}'.format(dateStart), end='{}'.format(dateEnd))
    HS300 = ts.pro_bar(ts_code=stock, asset='I', start_date=dateStart, end_date=dateEnd)
    HS300 = HS300[['trade_date', 'high', 'low', 'open', 'close']]
    HS300['date'] =  HS300['trade_date']
    HS300['trade_date'] = pd.to_datetime(HS300['trade_date'])
    HS300.sort_values(by='trade_date', inplace=True)
    HS300.index = range(0,len(HS300))
    # 1.斜率
    HS300['beta'] = 0
    HS300['R2'] = 0
    #dataframe的切片索引理解起来头大:
    for i in range(1, len(HS300) - 1):
        df_ne = HS300.loc[i-N+1:i,:]
        #print ('获取数据:',i-N+1,i)
        #print ('df_ne is:', df_ne)
        #如何使用模型:
        model = sml.ols(formula='high~low', data=df_ne)
        result = model.fit()
        #斜率：
        HS300.loc[i + 1, 'beta'] = float(result.params[1])
        HS300.loc[i + 1, 'R2'] = float(result.rsquared)
        #print  (HS300)

    #日收益率
    #HS300['ret'] = HS300.close.pct_change(1)
    HS300['hs300'] = HS300.close.pct_change(1)  #为了和后面的代码匹配
    #2.标准分
    #1: 包括600个完整数据的序列处理：
    HS300['beta_norm'] = (HS300['beta']-HS300.beta.rolling(M).mean().shift(1))/HS300.beta.rolling(M).std().shift(1)

    #2：前600个数据的处理：
    for i in range(M):
        # print (i)
        # print(type(HS300.loc[i, 'beta']))
        HS300.loc[i, 'beta_norm'] = (HS300.loc[i,'beta']-HS300.loc[:i-1,'beta'].mean()) / HS300.loc[:i-1,'beta'].std()

    HS300.loc[2, 'beta_norm'] = 0
    #3.修正标准分
    HS300['RSRS_R2'] = HS300.beta_norm * HS300.R2
    #print (HS300)
    HS300 = HS300.fillna(0)
    #4.右偏标准分
    HS300['beta_right'] = HS300.RSRS_R2 * HS300.beta
    return (HS300)


#构建策略:
def RSRS1(df, S1=1.0, S2=0.8, cost=0.0):

    '''输入参数：
    df为数据表: 包含open,close,low,high,vol，指数收益率数据hs300
    S1:买入参数, 大于1买入
    S2:卖出参数，小于0.8卖出
    cost为手续费+滑点价差，可以根据需要进行设置，默认为0.0
    '''
    #设计买卖信号，为尽量贴近实际，加入涨跌停不能买卖的限制
    #当beta值大于1且第二天开盘没有涨停发出买入信号设置为1
    df = df.copy()
    df.loc[(df.beta>S1) & (df['open'] < df['close'].shift(1) * 1.097), 'signal'] = 1
    #当btea值小于0.8且第二天开盘没有跌停发出卖入信号设置为0
    df.loc[(df.beta<S2) & (df['open'] > df['close'].shift(1) * 0.903), 'signal'] = 0
    df['position']=df['signal'].shift(1)
    df['position'].fillna(method='ffill',inplace=True)
    df['position'].fillna(0,inplace=True)
    #根据交易信号和仓位计算策略的每日收益率
    df.loc[df.index[0], 'capital_ret'] = 0
    #今天开盘新买入的position在今天的涨幅(扣除手续费)
    df.loc[df['position'] > df['position'].shift(1), 'capital_ret'] = \
                         (df['close'] / df['open']-1) * (1- cost)
    #卖出同理
    df.loc[df['position'] < df['position'].shift(1), 'capital_ret'] = \
                   (df['open'] / df['close'].shift(1)-1) * (1-cost)
    # 当仓位不变时,当天的capital是当天的change * position
    df.loc[df['position'] == df['position'].shift(1), 'capital_ret'] = \
                        df['hs300'] * df['position']
    #计算策略、指数的累计收益率
    df['capital_line']=(df.capital_ret+1.0).cumprod()
    #df['rets_line']=(df.rets+1.0).cumprod()
    df['hs300_line']=(df.hs300+1.0).cumprod()
    return df

def RSRS2(df, S1=0.7, cost=0.0):

    '''输入参数：
    df为数据表: 包含open,close,low,high,vol，指数收益率数据hs300
    S1:买入参数, 大于0.7买入
    S2:卖出参数，小于-0.7卖出
    cost为手续费+滑点价差，可以根据需要进行设置，默认为0.0
    '''
    df = df.copy()
    #设计买卖信号，为尽量贴近实际，加入涨跌停不能买卖的限制
    #beta_norm.7且第二天开盘没有涨停发出买入信号设置为1
    df.loc[(df.beta_norm>S1) & (df['open'] < df['close'].shift(1) * 1.097), 'signal'] = 1
    #beta_norm-0.7且第二天开盘没有跌停发出卖入信号设置为0
    df.loc[(df.beta_norm<-S1) & (df['open'] > df['close'].shift(1) * 0.903), 'signal'] = 0
    df['position']=df['signal'].shift(1)
    df['position'].fillna(method='ffill',inplace=True)
    df['position'].fillna(0,inplace=True)
    #根据交易信号和仓位计算策略的每日收益率
    df.loc[df.index[0], 'capital_ret'] = 0
    #今天开盘新买入的position在今天的涨幅(扣除手续费)
    df.loc[df['position'] > df['position'].shift(1), 'capital_ret'] = \
                         (df['close'] / df['open']-1) * (1- cost)
    #卖出同理
    df.loc[df['position'] < df['position'].shift(1), 'capital_ret'] = \
                   (df['open'] / df['close'].shift(1)-1) * (1-cost)
    # 当仓位不变时,当天的capital是当天的change * position
    df.loc[df['position'] == df['position'].shift(1), 'capital_ret'] = \
                        df['hs300'] * df['position']
    #计算策略、指数的累计收益率
    df['capital_line']=(df.capital_ret+1.0).cumprod()
    #df['rets_line']=(df.rets+1.0).cumprod()
    df['hs300_line']=(df.hs300+1.0).cumprod()
    return df


def RSRS3(df, S1=0.7, cost=0.0):
    '''输入参数：
    df为数据表: 包含open,close,low,high,vol，指数收益率数据hs300
    S1:买入参数, 大于0.7买入
    S2:卖出参数，小于-0.7卖出
    cost为手续费+滑点价差，可以根据需要进行设置，默认为0.0
    '''
    #设计买卖信号，为尽量贴近实际，加入涨跌停不能买卖的限制
    #当RSRS_R2值大于0.7且第二天开盘没有涨停发出买入信号设置为1
    df = df.copy()
    df.loc[(df.RSRS_R2>S1) & (df['open'] < df['close'].shift(1) * 1.097), 'signal'] = 1
    #当RSRS_R2值小于-0.7且第二天开盘没有跌停发出卖入信号设置为0
    df.loc[(df.RSRS_R2<-S1) & (df['open'] > df['close'].shift(1) * 0.903), 'signal'] = 0
    df['position']=df['signal'].shift(1)
    df['position'].fillna(method='ffill',inplace=True)
    df['position'].fillna(0,inplace=True)
    #根据交易信号和仓位计算策略的每日收益率
    df.loc[df.index[0], 'capital_ret'] = 0
    #今天开盘新买入的position在今天的涨幅(扣除手续费)
    df.loc[df['position'] > df['position'].shift(1), 'capital_ret'] = \
                         (df['close'] / df['open']-1) * (1- cost)
    #卖出同理
    df.loc[df['position'] < df['position'].shift(1), 'capital_ret'] = \
                   (df['open'] / df['close'].shift(1)-1) * (1-cost)
    # 当仓位不变时,当天的capital是当天的change * position
    df.loc[df['position'] == df['position'].shift(1), 'capital_ret'] = \
                        df['hs300'] * df['position']

    #计算策略、指数的累计收益率
    df['capital_line']=(df.capital_ret+1.0).cumprod()
    #df['rets_line']=(df.rets+1.0).cumprod()
    df['hs300_line']=(df.hs300+1.0).cumprod()
    return df


# 关掉pandas的warnings
pd.options.mode.chained_assignment = None
def trade_indicators(df):
    '''
    记录买入或者加仓时的日期和初始资# 计算资金曲线
    记录买入或者加仓时的日期和初始资产
    输出账户交易各项指标产
    :param df:
    :return:
    '''


def performance(df):
    '''
    计算每一年(月,周)hs300指数,资金曲线的收益
    计算策略的年（月，周）胜率
    计算总收益率、年化收益率和风险指标
    :param df:
    :return:
    '''


def return_plot(result1, result2, result3):
    xtick = np.arange(0,result1.shape[0],int(result1.shape[0]/7))
    xticklabel = pd.Series(result1.date[xtick])
    plt.figure(figsize=(15,3))
    fig = plt.axes()
    plt.plot(np.arange(result1.shape[0]),result1.capital_line,label = 'RSRS1',linewidth = 2)
    plt.plot(np.arange(result1.shape[0]),result2.capital_line,label = 'RSRS2',linewidth = 2)
    plt.plot(np.arange(result1.shape[0]),result3.capital_line,label = 'RSRS3',linewidth = 2)
    plt.plot(np.arange(result1.shape[0]),result1.hs300_line,color = 'yellow',label = 'HS300',linewidth = 2)
    fig.set_xticks(xtick)
    fig.set_xticklabels(xticklabel,rotation = 45)
    plt.legend()
    plt.show()

def main():
    stock = '000300.SH'
    # dateStart = '20140130'
    # dateEnd = '20210226'
    dateStart = '20050301'
    dateEnd = '20210226'
    # N：回归的时间长度，同研报
    # M：算标准分的实际长度，同研报
    N = 18
    M = 600
    HS300 = getdata(stock, dateStart, dateEnd, N, M)
    HS300 = HS300.loc[2:]

    # 重新调整索引：
    # set_index: drop:默认为true，表示是否将作为新索引的列删除; true则删除原列,如果为false，则保留原来的列
    # reset_index是set_index的逆操作，将索引重新转换为列; drop:true则新建索引,是否保留原索引，默认false保留原索引
    HS300 = HS300.reset_index(drop=True)

    #用copy()函数，否则HS300对象会被共用:
    result1 = RSRS1(HS300)
    result2 = RSRS2(HS300)
    result3 = RSRS3(HS300)

    # assert_frame_equal(result1, result2)
    # assert_frame_equal(result1, result3)
    # assert_frame_equal(result2, result3)

    #策略指标result1:
    print('RSRS1策略:')
    print(f'回测标的：{stock}')
    print(f'回测期间：{dateStart}—{dateEnd}')
    trade_indicators(result1)
    performance(result1)
    # 策略指标result2:
    print('RSRS2策略:')
    print(f'回测标的：{stock}')
    print(f'回测期间：{dateStart}—{dateEnd}')
    trade_indicators(result2)
    performance(result2)
    # 策略指标result3:
    print('RSRS3策略:')
    print(f'回测标的：{stock}')
    print(f'回测期间：{dateStart}—{dateEnd}')
    trade_indicators(result3)
    performance(result3)

    return_plot(result1,result2,result3)

main()






#!/usr/bin/python
#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)  #设置阈值为无限
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#正常显示画图时出现的中文和负号
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
import talib as ta
import tushare as ts

# token='2db4ac8eb2c40f4bf07d90fa3484130fbe5c168dd297f8ebfef55c96'  #2020
token='f6b511d8d4529f19319e1861edadda749e64a5b8573102deec80cfd8'  #2021
ts.set_token(token)
pro=ts.pro_api()

index={'上证综指': '000001.SH','深证成指': '399001.SZ',
        '沪深300': '000300.SH','创业板指': '399006.SZ',
        '上证50': '000016.SH','中证500': '000905.SH',
        '中小板指': '399005.SZ','上证180': '000010.SH'}

#获取当前交易的股票代码和名称
def get_code():
    #exchange: 交易所 SSE上交所 SZSE深交所
    #L上市 D退市 P暂停上市
    df = pro.stock_basic(exchange='', list_status='L')
    codes=df.ts_code.values    #ndarray类型
    names=df.name.values
    stock=dict(zip(names,codes))
    #合并指数和个股成一个字典
    #字典值传递需要双指针，类似于**kwargs
    stocks=dict(stock,**index)
    return stocks

#获取行情数据
def get_data(stock,start='20140130',end='20210226'):
    #如果代码在字典index里，则取的是指数数据
    code=get_code()[stock]
    if code in index.values():
        #asset:资产类别：E股票 I沪深指数
        df=ts.pro_bar(ts_code=code,asset='I',start_date=start, end_date=end)
    #否则取的是个股数据
    else:
        #None未复权 qfq前复权 hfq后复权
        df=ts.pro_bar(ts_code=code, adj='qfq',start_date=start, end_date=end)
    #将交易日期设置为索引值
    df.index=pd.to_datetime(df.trade_date)
    df=df.sort_index()
    return df

stock='中国平安'

df=get_data(stock)

def kline_plot(data):
    data['ma20']=data.close.rolling(20).mean()
    data['ma5']=data.close.rolling(5).mean()
    date = data.index.strftime('%Y%m%d').tolist()
    k_value = data.apply(lambda row: [row.open, row.close, row.low, row.high], axis=1).tolist()
    #引入pyecharts画图
    from pyecharts import Kline,Line, Bar, Scatter,Overlap
    kline = Kline('股价行情走势')
    kline.add("日K线图", date, k_value,
              is_datazoom_show=True,is_splitline_show=False)
    #加入20日均线
    line = Line()
    v0=data['ma5'].round(2).tolist()
    v=data['ma20'].round(2).tolist()
    line.add('5日均线', date,v0 ,is_symbol_show=False,line_width=2)
    line.add('20日均线', date,v ,is_symbol_show=False,line_width=2)
    # 成交量
    bar = Bar()
    bar.add("成交量", date, data['vol'],tooltip_tragger="axis", is_legend_show=False,
            is_yaxis_show=False, yaxis_max=5*max(data["vol"]))
    overlap = Overlap()
    overlap.add(kline)
    overlap.add(line,)
    overlap.add(bar,yaxis_index=1, is_add_yaxis=True)
    overlap.render('html/股价行情走势.html')

#kline_plot(df)

returns=df.close.pct_change().dropna()

#日收益率标准化图:
# returns.plot(figsize=(14,6),label='日收益率')
# plt.title('中国平安日收益图',fontsize=15)
# my_ticks = pd.date_range('2014-01-30','2021-02-26',freq='q')
# plt.xticks(my_ticks,fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('',fontsize=12)
# # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
# plt.axhline(returns.mean(), color='r',label='日收益均值')
# plt.axhline(returns.mean()+1.5*returns.std(), color='g',label='正负1.5倍标准差')
# plt.axhline(returns.mean()-1.5*returns.std(), color='g')
# plt.legend()
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# plt.show()


#20日收益率标准化图
# ret_20=returns.rolling(20).mean()
# std_20=returns.rolling(20).std()
# score=((returns-ret_20)/std_20)
# score.plot(figsize=(14,6),label='20日收益率标准化')
# plt.title('中国平安20日收益标准化图',fontsize=15)
# my_ticks = pd.date_range('2014-01-30','2021-02-26',freq='q')
# plt.xticks(my_ticks,fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('',fontsize=12)
# # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
# plt.axhline(score.mean(), color='r',label='日收益均值')
# plt.axhline(score.mean()+1.5*score.std(), color='g',label='正负1.5倍标准差')
# plt.axhline(score.mean()-1.5*score.std(), color='g')
# plt.legend()
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# plt.show()

#获取数据
def data_feed(stock,start='20140130',end='20210226'):
    #获取个股数据
    df=get_data(stock,start,end)[['open','close','low','high','vol']]
    #指数数据,作为参照指标
    df['hs300']=get_data('沪深300',start,end).close.pct_change()
    #计算收益率
    df['rets']=df.close.pct_change().dropna()
    return df.dropna()


#构建策略:
def MR_Strategy(df,lookback=20,buy_threshold=-1.5,sell_threshold=1.5,cost=0.0):
    '''输入参数：
    df为数据表: 包含open,close,low,high,vol，标的收益率rets，指数收益率数据hs300
    lookback为均值回归策略参数，设置统计区间长度，默认20天
    buy_threshold:买入参数，均值向下偏离标准差的倍数，默认-1.5
    sell_threshold:卖出参数，均值向上偏离标准差的倍数，默认1.5
    cost为手续费+滑点价差，可以根据需要进行设置，默认为0.0
    '''
    #计算均值回归策略的score值
    ret_lb=df.rets.rolling(lookback).mean()
    std_lb=df.rets.rolling(lookback).std()
    df['score']=(df.rets-ret_lb)/std_lb
    df.dropna(inplace=True)
    #设计买卖信号，为尽量贴近实际，加入涨跌停不能买卖的限制
    #当score值小于-1.5且第二天开盘没有涨停发出买入信号设置为1
    df.loc[(df.score<buy_threshold) &(df['open'] < df['close'].shift(1) * 1.097), 'signal'] = 1
    #当score值大于1.5且第二天开盘没有跌停发出卖入信号设置为0
    df.loc[(df.score>sell_threshold) &(df['open'] > df['close'].shift(1) * 0.903), 'signal'] = 0
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
                        df['rets'] * df['position']
    #计算标的、策略、指数的累计收益率
    df['capital_line']=(df.capital_ret+1.0).cumprod()
    df['rets_line']=(df.rets+1.0).cumprod()
    df['hs300_line']=(df.hs300+1.0).cumprod()
    return df

# d0=data_feed('中国平安')
# d1=MR_Strategy(d0)
# print(d1.tail())

# 关掉pandas的warnings
pd.options.mode.chained_assignment = None
def trade_indicators(df):
    '''

    :param df:
    :return:
    '''
    # 计算资金曲线
    df['capital'] = (df['capital_ret'] + 1).cumprod()
    df = df.reset_index()
    # 记录买入或者加仓时的日期和初始资# 计算资金曲线
    #     # 记录买入或者加仓时的日期和初始资产
    #     # 输出账户交易各项指标产
    df.loc[df['position'] > df['position'].shift(1), 'start_date'] = df['trade_date']
    df.loc[df['position'] > df['position'].shift(1), 'start_capital'] = df['capital'].shift(1)
    df.loc[df['position'] > df['position'].shift(1), 'start_stock'] = df['close'].shift(1)
    # 记录卖出时的日期和当天的资产
    df.loc[df['position'] < df['position'].shift(1), 'end_date'] = df['trade_date']
    df.loc[df['position'] < df['position'].shift(1), 'end_capital'] = df['capital']
    df.loc[df['position'] < df['position'].shift(1), 'end_stock'] = df['close']
    # 将买卖当天的信息合并成一个dataframe
    df_temp = df[df['start_date'].notnull() | df['end_date'].notnull()]

    df_temp['end_date'] = df_temp['end_date'].shift(-1)
    df_temp['end_capital'] = df_temp['end_capital'].shift(-1)
    df_temp['end_stock'] = df_temp['end_stock'].shift(-1)

    # 构建账户交易情况dataframe：'hold_time'持有天数，
    # 'trade_return'该次交易盈亏,'stock_return'同期股票涨跌幅
    trade = df_temp.loc[df_temp['end_date'].notnull(), ['start_date', 'start_capital', 'start_stock',
                                                        'end_date', 'end_capital', 'end_stock']]
    trade['hold_time'] = (trade['end_date'] - trade['start_date']).dt.days
    trade['trade_return'] = trade['end_capital'] / trade['start_capital'] - 1
    trade['stock_return'] = trade['end_stock'] / trade['start_stock'] - 1

    trade_num = len(trade)  # 计算交易次数
    max_holdtime = trade['hold_time'].max()  # 计算最长持有天数
    average_change = trade['trade_return'].mean()  # 计算每次平均涨幅
    max_gain = trade['trade_return'].max()  # 计算单笔最大盈利
    max_loss = trade['trade_return'].min()  # 计算单笔最大亏损
    total_years = (trade['end_date'].iloc[-1] - trade['start_date'].iloc[0]).days / 365
    trade_per_year = trade_num / total_years  # 计算年均买卖次数

    # 计算连续盈利亏损的次数
    trade.loc[trade['trade_return'] > 0, 'gain'] = 1
    trade.loc[trade['trade_return'] < 0, 'gain'] = 0
    trade['gain'].fillna(method='ffill', inplace=True)
    # 根据gain这一列计算连续盈利亏损的次数
    rtn_list = list(trade['gain'])
    successive_gain_list = []
    num = 1
    for i in range(len(rtn_list)):
        if i == 0:
            successive_gain_list.append(num)
        else:
            if (rtn_list[i] == rtn_list[i - 1] == 1) or (rtn_list[i] == rtn_list[i - 1] == 0):
                num += 1
            else:
                num = 1
            successive_gain_list.append(num)
    # 将计算结果赋给新的一列'successive_gain'
    trade['successive_gain'] = successive_gain_list
    # 分别在盈利和亏损的两个dataframe里按照'successive_gain'的值排序并取最大值
    max_successive_gain = trade[trade['gain'] == 1].sort_values(by='successive_gain', \
                                                                ascending=False)['successive_gain'].iloc[0]
    max_successive_loss = trade[trade['gain'] == 0].sort_values(by='successive_gain', \
                                                                ascending=False)['successive_gain'].iloc[0]

    #  输出账户交易各项指标
    print('\n==============每笔交易收益率及同期股票涨跌幅===============')
    print(trade[['start_date', 'end_date', 'trade_return', 'stock_return']])
    print('\n====================账户交易的各项指标=====================')
    print('交易次数为：%d   最长持有天数为：%d' % (trade_num, max_holdtime))
    print('每次平均涨幅为：%f' % average_change)
    print('单次最大盈利为：%f  单次最大亏损为：%f' % (max_gain, max_loss))
    print('年均买卖次数为：%f' % trade_per_year)
    print('最大连续盈利次数为：%d  最大连续亏损次数为：%d' % (max_successive_gain, max_successive_loss))
    return trade


def performance(df):
    '''
    计算每一年(月,周)股票,资金曲线的收益
    计算策略的年（月，周）胜率
    计算总收益率、年化收益率和风险指标
    :param df:
    :return:
    '''

def plot_strategy_signal(df, trade, stock):
    '''
    #对K线图和买卖信号进行可视化
    :param df:
    :param trade:
    :param stock:
    :return:
    '''


def plot_performance(df,stock):
    '''
     # 对策略和标的股票累计收益率进行可视化
     # df策略返回的据框，包含策略的收益率
     # stock为回测的股票简称
    :param df:
    :param stock:
    :return:
    '''



def main(stock,start,end):
    d0=data_feed(stock,start,end)
    d1=MR_Strategy(d0)
    print(f'回测标的：{stock}')
    print(f'回测期间：{start}—{end}')
    trade=trade_indicators(d1)
    performance(d1)
    plot_performance(d1,stock)
    return d1,trade

stock='中国平安'
d1,trade=main(stock,'20140130','20210226')
plot_strategy_signal(d1,trade,stock).render('html/买卖信号.html')
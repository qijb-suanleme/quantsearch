#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,unicode_literals)
import backtrader as bt
import pandas as pd
import tushare as ts
import matplotlib.pyplot as plt


from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

def get_data(code,start='2014-01-30',end='2021-2-26'):
    df=ts.get_k_data(code,autype='qfq',start=start,end=end)
    df.index=pd.to_datetime(df.date)
    df['openinterest']=0
    df=df[['open','high','low','close','volume','openinterest']]
    return df


#由于交易过程中需要对仓位进行动态调整，每次交易一单元股票（不是固定的一股或100股，根据ATR而定），因此交易头寸需要重新设定。
class TradeSizer(bt.Sizer):
    params = (('stake', 1),)
    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return self.p.stake
        position = self.broker.getposition(data)
        if not position.size:
            return 0
        else:
            return position.size
        #return self.p.stake


# 交易策略:
# 回顾一下海龟交易法则的策略思路：
# 入场条件：当收盘价突破20日价格高点时，买入一单元股票；
# 加仓条件：当价格大于上一次买入价格的0.5个ATR（平均波幅），买入一单元股票，加仓次数不超过3次；
# 止损条件：当价格小于上一次买入价格的2个ATR时清仓；
# 离场条件：当价格跌破10日价格低点时清仓。
# 这里的20日价格高点和10日价格低点构成唐奇安通道，所以海龟交易法则也可以理解成通道突破的趋势跟踪。
class TurtleStrategy(bt.Strategy):
#默认参数
    params = (('long_period',20),
              ('short_period',10),
              ('printlog', False), )

    def __init__(self):
        self.order = None
        self.buyprice = 0
        self.buycomm = 0
        self.buy_size = 0
        self.buy_count = 0
        # 海龟交易法则中的唐奇安通道和平均波幅ATR
        self.H_line = bt.indicators.Highest(self.data.high(-1), period=self.p.long_period)
        self.L_line = bt.indicators.Lowest(self.data.low(-1), period=self.p.short_period)
        self.TR = bt.indicators.Max((self.data.high(0)- self.data.low(0)),\
                                    abs(self.data.close(-1)-self.data.high(0)), \
                                    abs(self.data.close(-1)-self.data.low(0)))
        self.ATR = bt.indicators.SimpleMovingAverage(self.TR, period=14)
        # 价格与上下轨线的交叉
        self.buy_signal = bt.ind.CrossOver(self.data.close(0), self.H_line)
        self.sell_signal = bt.ind.CrossOver(self.data.close(0), self.L_line)

    def next(self):
        if self.order:
            return

        #核心策略:
        #入场：价格突破上轨线且空仓时
        if self.buy_signal > 0 and self.buy_count == 0:
            self.buy_size = self.broker.getvalue() * 0.01 / self.ATR
            self.buy_size  = int(self.buy_size / 100) * 100
            #这个变量是专门用来控制买卖数量的:
            self.sizer.p.stake = self.buy_size
            self.buy_count = 1
            self.order = self.buy()

        #加仓：价格上涨了买入价的0.5的ATR且加仓次数少于3次（含）
        elif self.data.close >self.buyprice+0.5*self.ATR[0] and self.buy_count > 0 and self.buy_count <=4:
            #print ('self.buyprice+0.5*self.ATR[0]:',self.buyprice+0.5*self.ATR[0])
            self.buy_size  = self.broker.getvalue() * 0.01 / self.ATR
            self.buy_size  = int(self.buy_size/100) * 100
            self.sizer.p.stake = self.buy_size
            self.order = self.buy()
            self.buy_count += 1

        #离场：价格跌破下轨线且持仓时
        elif self.sell_signal < 0 and self.buy_count > 0:
            self.order = self.sell()
            self.buy_count = 0

        #止损：价格跌破买入价的2个ATR且持仓时
        elif self.data.close < (self.buyprice - 2*self.ATR[0]) and self.buy_count > 0:
            self.order = self.sell()
            self.buy_count = 0

    #交易记录日志（默认不打印结果）
    def log(self, txt, dt=None,doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()},{txt}')

    #记录交易执行情况（默认不输出结果）
    def notify_order(self, order):
        # 如果order为submitted/accepted,返回空
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 如果order为buy/sell executed,报告价格结果
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入:\n价格:{order.executed.price},\
                成本:{order.executed.value},\
                手续费:{order.executed.comm}')

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'卖出:\n价格：{order.executed.price},\
                成本: {order.executed.value},\
                手续费{order.executed.comm}')

            self.bar_executed = len(self)

        # 如果指令取消/交易失败, 报告结果
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('交易失败')
        self.order = None

    #记录交易收益情况（可省略，默认不输出结果）
    def notify_trade(self,trade):
        if not trade.isclosed:
            return
        self.log(f'策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}')

    def stop(self):
        self.log(f'(组合线：{self.p.long_period},{self.p.short_period})； \
        期末总资金: {self.broker.getvalue():.2f}', doprint=True)


#找到最优参数:
# def main(code,long_list,short_list,start,end='',startcash=1000000,com=0.001):
#     #创建主控制器
#     cerebro = bt.Cerebro()
#     #导入策略参数寻优
#     cerebro.optstrategy(TurtleStrategy,long_period=long_list,short_period=short_list)
#     #获取数据
#     df=ts.get_k_data(code,autype='qfq',start=start,end=end)
#     df.index=pd.to_datetime(df.date)
#     df=df[['open','high','low','close','volume']]
#     #将数据加载至回测系统
#     data = bt.feeds.PandasData(dataname=df)
#     cerebro.adddata(data)
#     #broker设置资金、手续费
#     cerebro.broker.setcash(startcash)
#     cerebro.broker.setcommission(commission=com)
#     #设置买入设置，策略，数量
#     cerebro.addsizer(TradeSizer)
#     print('期初总资金: %.2f' % cerebro.broker.getvalue())
#     cerebro.run(maxcpus=1)
#
# long_list=range(20,70,5)
# short_list=range(5,20,5)
# main('sh',long_list,short_list,'2014-01-30','2021-02-26')


#输出参数评测报告:
def performance(code,long,short,start,end='',startcash=1000000,com=0.001):
    cerebro = bt.Cerebro()
    #导入策略参数寻优
    cerebro.addstrategy(TurtleStrategy,long_period=long,short_period=short)
    #获取数据
    df=ts.get_k_data(code,autype='qfq',start=start,end=end)
    df.index=pd.to_datetime(df.date)
    df=df[['open','high','low','close','volume']]
    #将数据加载至回测系统
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    #broker设置资金、手续费
    cerebro.broker.setcash(startcash)
    cerebro.broker.setcommission(commission=com)
    #设置买入设置，策略，数量
    cerebro.addsizer(TradeSizer)
    df00,df0,df1,df2,df3,df4=bt.out_result(cerebro)
    return df00,df0,df1,df2,df3,df4

long=25
short=15

df00,df0,df1,df2,df3,df4=performance('sh',long,short,'2014-01-30','2021-02-26')
df00.to_csv('常用指标.csv')
df0.to_csv('账户收益率.csv')
df1.to_csv('总的杠杆.csv')
df2.to_csv('滚动的对数收益率.csv')
df3.to_csv('每年累积收益率.csv')
df4.to_csv('总的持仓价值.csv')

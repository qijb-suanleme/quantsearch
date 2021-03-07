#!/usr/bin/python
#coding:utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import datetime  # 用于datetime对象操作
import os.path   # 用于管理路径
import sys       # 用于在argvTo[0]中找到脚本名称
import backtrader as bt   # 引入backtrader框架
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
import pandas as pd
import numpy as np

np.set_printoptions(threshold=np.inf)  #设置阈值为无限
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 创建策略
class SmaCross(bt.Strategy):
    # 可配置策略参数
    params = dict(
        pfast=5,  # 短期均线周期
        pslow=60,  # 长期均线周期
        pstake=1000,  # 单笔交易股票数目
        trailamount=0,
        trailpercent=0.03  #控制回撤幅度
    )

    def __init__(self):
        #双字典结构:
        self.inds = dict()
        self.log_file = open('position_log.txt', 'w')  # 用于输出仓位信息
        #字典也可以存储一个引用的地址:
        for i, d in enumerate(self.datas):
            #信号的设置和存储方式：
            # print ('in _init_................... ', self.datas[i]._name)
            #多股指标计算的时候，关于最小周期出错的调试:
            self.inds[d] = dict()
            self.inds[d]['sma1'] = bt.ind.SMA(d.close, period=self.p.pfast)  # 短期均线
            self.inds[d]['sma2'] = bt.ind.SMA(d.close, period=self.p.pslow)  # 长期均线
            self.inds[d]['cross'] = bt.ind.CrossOver(self.inds[d]['sma1'], self.inds[d]['sma2'], plot=False)

    def next(self):
        for i, d in enumerate(self.datas):
            pos = self.broker.getposition(d).size
            if not pos:  # 不在场内，则可以买入
                if self.inds[d]['cross'] > 0:  #如果金叉
                    #buy用Market订单类型
                    # self.buy(data=d, size=self.p.pstake, exectype=bt.Order.Market)
                    # print('BUY CREATE, exectype Market, close %.2f'.format(self.data.close[0]),file=self.log_file)
                    self.buy(data=d, size=self.p.pstake)  # 买

            elif self.inds[d]['cross'] < 0:    #在场内，且死叉
            #     #用sell用StopTrail订单来控制止损
            #     st_order=self.sell(data=d, size=self.p.pstake, exectype=bt.Order.StopTrail,
            #                      trailamount=self.p.trailamount,
            #                      trailpercent=self.p.trailpercent)
            #     if self.p.trailamount:
            #         check = self.data.close - self.p.trailamount
            #     else:
            #         check = self.data.close * (1.0 - self.p.trailpercent)
            #
            #     print('SELL CREATE, exectype StopTrail, close %.2f, stop price %.2f, check price %.2f'.format(
            #         self.data.close[0], st_order.created.price, check),
            #         file=self.log_file)

                 self.close(data=d)  # 卖

        # 打印仓位信息
        print('******************************', file=self.log_file)
        print(self.data.datetime.date(), file=self.log_file)
        for i, d in enumerate(self.datas):
            pos = self.getposition(d)
            if len(pos):
                print('{}, 持仓:{}, 成本价:{}, 当前价:{}, 盈亏:{:.2f}'.format(
                    d._name, pos.size, pos.price, pos.adjbase, pos.size * (pos.adjbase - pos.price)),
                    file=self.log_file)

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        dt = self.data.datetime.date()
        print('{} {} Closed: PnL Gross(毛收益) {}, Net(净收益) {}'.format(
            dt, trade.data._name, round(trade.pnl, 2), round(trade.pnlcomm, 2)
        ))

    def stop(self):
        self.log_file.close()
        pass

cerebro = bt.Cerebro()  # 创建cerebro
# 读入股票代码
data_path="stk_100"
filelist = []
for (dirpath, dirnames, filenames) in os.walk(data_path):
    filelist.extend(filenames)

for file in filelist:
    datapath = data_path +'\\'+ file
    # 创建价格数据
    data = bt.feeds.GenericCSVData(
        dataname=datapath,
        fromdate=datetime.datetime(2014, 1, 30),
        todate=datetime.datetime(2020, 9, 30),
        nullvalue=0.0,
        dtformat=('%Y-%m-%d'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1
    )
    # 在Cerebro中添加股票数据
    cerebro.adddata(data, name=file[:-4])
# 设置启动资金
cerebro.broker.setcash(100000.0)

# 设置佣金为千分之一
cerebro.broker.setcommission(commission=0.001)
cerebro.addstrategy(SmaCross)  # 添加策略

df00,df0,df1,df2,df3,df4=bt.out_result(cerebro)
print ('绩效指标,普通交易指标,多空交易指标:')
print (df00)

print ('账户收益率')
print (df0)

print ('总的杠杆')
print (df1)

print ('滚动的对数收益率')
print (df2)

print ('每年收益率')
print (df3)

print ('总的持仓价值')
print (df4)

# 设置回测结果中不显示期权K线
for d in cerebro.datas:
    d.plotinfo.plot = False

b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
cerebro.plot(b)

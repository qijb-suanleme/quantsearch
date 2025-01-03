import datetime
import numpy as np
import pandas as pd
import time
from jqdata import *
from pandas import Series, DataFrame
import statsmodels.api as sm
from jqfactor import get_factor_values

'''
================================================================================
总体回测前
================================================================================
'''


# 总体回测前要做的事情（永远不变的）
def initialize(context):
    set_params()  # 1 设置策参数
    set_variables()  # 2 设置中间变量
    set_backtest()  # 3 设置回测条件


# 1 设置策略参数
def set_params():
    # 单因子测试时g.factor不应为空
    g.factor = 'BP'  # 当前回测的单因子
    g.shift = 21  # 设置一个观测天数（天数）按月调仓
    g.precent = 0.10  # 持仓占可选股票池比例
    g.index = '000300.XSHG'  # 定义股票池，沪深300
    # 多因子合并称DataFrame，单因子测试时可以把无用部分删除提升回测速度
    # 定义因子以及排序方式，默认False方式为降序排列，原值越大sort_rank排序越小
    g.factors = {'BP': False, 'net_profit_increase': True, 'inc_net_profit_year_on_year': True,
                 'operating_profit': True,
                 'inc_revenue_year_on_year': True
                 }
    # 设定选取sort_rank： True 为最大，False 为最小
    g.sort_rank = True
    g.quantile = (90, 100)


'''
000906.XSHG
中证800
'''


# 2 设置中间变量
def set_variables():
    g.feasible_stocks = []  # 当前可交易股票池
    g.if_trade = False  # 当天是否交易
    g.num_stocks = 0  # 设置持仓股票数目


# 3 设置回测条件
def set_backtest():
    set_benchmark('000300.XSHG')  # 设置为基准
    set_option('use_real_price', True)  # 用真实价格交易
    log.set_level('order', 'error')  # 设置报错等级


'''
================================================================================
每天开盘前
================================================================================
'''


# 每天开盘前要做的事情
def before_trading_start(context):
    # 获得当前日期
    day = context.current_dt.day
    yesterday = context.previous_date
    rebalance_day = shift_trading_day(yesterday, 1)
    if yesterday.month != rebalance_day.month:
        if yesterday.day > rebalance_day.day:
            g.if_trade = True
            # 5 设置可行股票池：获得当前开盘的股票池并剔除当前或者计算样本期间停牌的股票
            g.feasible_stocks = set_feasible_stocks(get_index_stocks(g.index), g.shift, context)
            # 6 设置滑点与手续费
            set_slip_fee(context)
            # 购买股票为可行股票池对应比例股票
            g.num_stocks = int(len(g.feasible_stocks) * g.precent)


# 4
# 某一日的前shift个交易日日期
# 输入：date为datetime.date对象(是一个date，而不是datetime)；shift为int类型
# 输出：datetime.date对象(是一个date，而不是datetime)
def shift_trading_day(date, shift):
    # 获取所有的交易日，返回一个包含所有交易日的 list,元素值为 datetime.date 类型.
    tradingday = get_all_trade_days()
    # 得到date之后shift天那一天在列表中的行标号 返回一个数
    shiftday_index = list(tradingday).index(date) + shift
    # 根据行号返回该日日期 为datetime.date类型
    return tradingday[shiftday_index]


# 5
# 设置可行股票池
# 过滤掉当日停牌的股票,且筛选出前days天未停牌股票
# 输入：stock_list为list类型,样本天数days为int类型，context（见API）
# 输出：list=g.feasible_stocks
def set_feasible_stocks(stock_list, days, context):
    # 得到是否停牌信息的dataframe，停牌的1，未停牌得0
    suspened_info_df = get_price(list(stock_list),
                                 start_date=context.current_dt,
                                 end_date=context.current_dt,
                                 frequency='daily',
                                 fields='paused'
                                 )['paused'].T
    # 过滤停牌股票 返回dataframe
    unsuspened_index = suspened_info_df.iloc[:, 0] < 1
    # 得到当日未停牌股票的代码list:
    unsuspened_stocks = suspened_info_df[unsuspened_index].index
    # 进一步，筛选出前days天未曾停牌的股票list:
    feasible_stocks = []
    current_data = get_current_data()
    for stock in unsuspened_stocks:
        if sum(attribute_history(stock,
                                 days,
                                 unit='1d',
                                 fields=('paused'),
                                 skip_paused=False
                                 )
               )[0] == 0:
            feasible_stocks.append(stock)
    # 剔除ST股
    st_data = get_extras('is_st', feasible_stocks, end_date=context.previous_date, count=1)
    stockList = [stock for stock in feasible_stocks if not st_data[stock][0]]
    return stockList


# 6 根据不同的时间段设置滑点与手续费(永远不变的函数)
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 根据不同的时间段设置手续费
    dt = context.current_dt

    if dt > datetime.datetime(2013, 1, 1):
        set_commission(PerTrade(buy_cost=0.0003,
                                sell_cost=0.0013,
                                min_cost=5))

    elif dt > datetime.datetime(2011, 1, 1):
        set_commission(PerTrade(buy_cost=0.001,
                                sell_cost=0.002,
                                min_cost=5))

    elif dt > datetime.datetime(2009, 1, 1):
        set_commission(PerTrade(buy_cost=0.002,
                                sell_cost=0.003,
                                min_cost=5))

    else:
        set_commission(PerTrade(buy_cost=0.003,
                                sell_cost=0.004,
                                min_cost=5))


'''
================================================================================
每天交易时
================================================================================
'''


def handle_data(context, data):
    # 如果为交易日
    if g.if_trade == True:
        # 7 获得买入卖出信号，输入context，输出股票列表list
        # 字典中对应默认值为false holding_list筛选为true，则选出因子得分最大的
        holding_list = get_stocks(g.feasible_stocks,
                                  context,
                                  asc=g.sort_rank)
        # 新加入的部分，计算holding_list长度
        total_number = len(holding_list)
        # print 'feasible_stocks is %d, holding is %d' %(len(g.feasible_stocks), total_number)
        # 提取需要的分位信息
        (start_q, end_q) = g.quantile
        # 8 重新调整仓位，输入context,使用信号结果holding_list
        rebalance(context, holding_list, start_q, end_q, total_number)
        g.if_trade = False


# 7 原始数据重提取因子打分排名（核心逻辑）
def get_stocks(stocks_list, context, asc):
    #   构建一个新的字符串，名字叫做 'get_df_'+ 'key'
    tmp = 'get_df' + '_' + g.factor
    # 声明字符串是个方程
    aa = globals()[tmp](stocks_list, context, g.factors[g.factor])
    # 3倍标准差去极值
    # aa = winsorize(aa,g.factor,std = 3,have_negative = True)
    # z标准化
    # aa = standardize(aa,g.factor,ty = 2)
    # 获取市值因子
    # cap_data = get_market_cap(context)
    # 市值中性化
    # factor_residual_data = neutralization(aa,g.factor,cap_data)
    # 删除nan，以备数据中某项没有产生nan
    # aa = aa[pd.notnull(aa['BP'])]
    # 生成排名序数
    # aa['BP_sorted_rank'] = aa['BP'].rank(ascending = asc, method = 'dense')
    score = g.factor + '_' + 'sorted_rank'
    stocks = list(aa.sort(score, ascending=asc).index)
    # print stocks
    return stocks


# 8
# 依本策略的买入信号，得到应该买的股票列表
# 借用买入信号结果，不需额外输入
# 输入：context（见API）
def rebalance(context, holding_list, start_q, end_q, total_number):
    if end_q == 100:
        end_q = 100
    # 每只股票购买金额
    every_stock = context.portfolio.portfolio_value / g.num_stocks
    # 空仓只有买入操作
    if len(list(context.portfolio.positions.keys())) == 0:
        # 原设定重scort始于回报率相关打分计算，回报率是升序排列
        for stock_to_buy in holding_list[start_q * total_number / 100: end_q * total_number / 100]:
            order_target_value(stock_to_buy, every_stock)
    else:
        # 不是空仓先卖出持有但是不在购买名单中的股票
        for stock_to_sell in list(context.portfolio.positions.keys()):
            if stock_to_sell not in holding_list[start_q * total_number / 100: end_q * total_number / 100]:
                order_target_value(stock_to_sell, 0)
        # 因order函数调整为顺序调整，为防止先行调仓股票由于后行调仓股票占金额过大不能一次调整到位，这里运行两次以解决这个问题
        for stock_to_buy in holding_list[start_q * total_number / 100: end_q * total_number / 100]:
            order_target_value(stock_to_buy, every_stock)
        for stock_to_buy in holding_list[start_q * total_number / 100: end_q * total_number / 100]:
            order_target_value(stock_to_buy, every_stock)


# BP
# 得到一个dataframe：包含股票代码、账面市值比BP和对应排名BP_sorted_rank
# 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
# 获得市净率pb_ratio
def get_df_BP(stock_list, context, asc):
    df_BP = get_fundamentals(query(valuation.code, valuation.pb_ratio
                                   ).filter(valuation.code.in_(stock_list)))
    # 获得pb倒数
    df_BP['BP'] = df_BP['pb_ratio'].apply(lambda x: 1 / x)
    # 删除nan，以备数据中某项没有产生nan
    df_BP = df_BP[pd.notnull(df_BP['BP'])]
    # 生成排名序数
    df_BP['BP_sorted_rank'] = df_BP['BP'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_BP.index = df_BP.code
    # 删除无用数据
    del df_BP['code']
    # print df_BP
    return df_BP


# EP
# 得到一个dataframe：包含股票代码、盈利收益率EP和EP_sorted_rank
# 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
# 获得动态市盈率pe_ratio
def get_df_EP(stock_list, context, asc):
    q = query(valuation.code, valuation.pe_ratio).filter(valuation.code.in_(stock_list))
    df_EP = get_fundamentals(q, date=context.current_dt - datetime.timedelta(days=1))
    # 获得pe倒数
    df_EP['EP'] = df_EP['pe_ratio'].apply(lambda x: 1 / x)
    # 删除nan，以备数据中某项没有产生nan
    df_EP = df_EP[pd.notnull(df_EP['EP'])]
    # 复制一个dataframe，按对应项目排序
    df_EP['EP_sorted_rank'] = df_EP['EP'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_EP.index = df_EP.code
    # 删除无用数据
    del df_EP['code']
    return df_EP


# 2、获取净利润增长率inc_net_profit_year_on_year
def get_df_inc_net_profit_year_on_year(stock_list, context, asc):
    q = query(valuation.code, indicator.inc_net_profit_year_on_year).filter(valuation.code.in_(stock_list))
    # 获取inc_net_profit_year_on_year数据
    df_inc_net_profit_year_on_year = get_fundamentals(q, date=context.current_dt - datetime.timedelta(days=1))
    # 删除nan，以备数据中某项没有产生nan
    df_inc_net_profit_year_on_year = df_inc_net_profit_year_on_year[
        pd.notnull(df_inc_net_profit_year_on_year['inc_net_profit_year_on_year'])]
    # 复制一个dataframe，按对应项目排序
    df_inc_net_profit_year_on_year['inc_net_profit_year_on_year_sorted_rank'] = df_inc_net_profit_year_on_year[
        'inc_net_profit_year_on_year'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_inc_net_profit_year_on_year.index = df_inc_net_profit_year_on_year.code
    # 删除无用数据
    del df_inc_net_profit_year_on_year['code']
    return df_inc_net_profit_year_on_year


# 3、获取营业利润数据operating_profit
def get_df_operating_profit(stock_list, context, asc):
    q = query(valuation.code, income.operating_profit).filter(valuation.code.in_(stock_list))
    # 获取营业利润operating_profit数据
    df_operating_profit = get_fundamentals(q, date=context.current_dt - datetime.timedelta(days=1))
    ##删除nan
    df_operating_profit = df_operating_profit[pd.notnull(df_operating_profit['operating_profit'])]
    # 复制一个df，放入排序结果
    df_operating_profit['operating_profit_sorted_rank'] = df_operating_profit['operating_profit'].rank(ascending=asc,
                                                                                                       method='dense')
    # 使用股票代码作为index
    df_operating_profit.index = df_operating_profit['code']
    # 删除无用数据
    del df_operating_profit['code']
    return df_operating_profit


# 4、获取营业收入增长率inc_revenue_year_on_year
def get_df_inc_revenue_year_on_year(stock_list, context, asc):
    q = query(valuation.code, indicator.inc_revenue_year_on_year).filter(valuation.code.in_(stock_list))
    # 获取营业收入增长率inc_revenue_year_on_year数据
    df_inc_revenue_year_on_year = get_fundamentals(q, date=context.current_dt - datetime.timedelta(days=1))
    # 删除nan
    df_inc_revenue_year_on_year = df_inc_revenue_year_on_year[
        pd.notnull(df_inc_revenue_year_on_year['inc_revenue_year_on_year'])]
    # 复制一个df，放入排序结果
    df_inc_revenue_year_on_year['inc_revenue_year_on_year_sorted_rank'] = df_inc_revenue_year_on_year[
        'inc_revenue_year_on_year'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_inc_revenue_year_on_year.index = df_inc_revenue_year_on_year['code']
    # 删除无用数据
    del df_inc_revenue_year_on_year['code']
    return df_inc_revenue_year_on_year


# 3PEG
# 输入：context(见API)；stock_list为list类型，表示股票池
# 输出：df_PEG为dataframe: index为股票代码，data为相应的PEG值
def get_df_PEG(stock_list, context, asc):
    # 查询股票池里股票的市盈率，收益增长率
    q_PE_G = query(valuation.code, valuation.pe_ratio, indicator.inc_net_profit_year_on_year
                   ).filter(valuation.code.in_(stock_list))
    # 得到一个dataframe：包含股票代码、市盈率PE、收益增长率G
    # 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
    df_PE_G = get_fundamentals(q_PE_G)
    # 筛选出成长股：删除市盈率或收益增长率为负值的股票
    df_Growth_PE_G = df_PE_G[(df_PE_G.pe_ratio > 0) & (df_PE_G.inc_net_profit_year_on_year > 0)]
    # 去除PE或G值为非数字的股票所在行
    df_Growth_PE_G.dropna()
    # 得到一个Series：存放股票的市盈率TTM，即PE值
    Series_PE = df_Growth_PE_G.ix[:, 'pe_ratio']
    # 得到一个Series：存放股票的收益增长率，即G值
    Series_G = df_Growth_PE_G.ix[:, 'inc_net_profit_year_on_year']
    # 得到一个Series：存放股票的PEG值
    Series_PEG = Series_PE / Series_G
    # 将股票与其PEG值对应
    Series_PEG.index = df_Growth_PE_G.ix[:, 0]
    # 生成空dataframe
    df_PEG = pd.DataFrame(Series_PEG)
    # 将Series类型转换成dataframe类型
    df_PEG['PEG'] = pd.DataFrame(Series_PEG)
    # 得到一个dataframe：包含股票代码、盈利收益率PEG和PEG_sorted_rank
    # 赋予顺序排列PEG数据序数编号
    df_PEG['PEG_sorted_rank'] = df_PEG['PEG'].rank(ascending=asc, method='dense')
    # 删除不需要列
    df_PEG = df_PEG.drop(0, 1)
    return df_PEG


# 4DP
# 得到一个dataframe：包含股票代码、股息率(DP)和DP_sorted_rank
# 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
def get_df_DP(stock_list, context, asc):
    # 获得dividend_payable和market_cap 应付股利(元)和总市值(亿元)
    df_DP = get_fundamentals(query(balance.code, balance.dividend_payable, valuation.market_cap
                                   ).filter(balance.code.in_(stock_list)))
    # 按公式计算
    df_DP['DP'] = df_DP['dividend_payable'] / (df_DP['market_cap'] * 100000000)
    # 删除nan
    df_DP = df_DP.dropna()
    # 生成排名序数
    df_DP['DP_sorted_rank'] = df_DP['DP'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_DP.index = df_DP.code
    # 删除无用数据
    del df_DP['code']
    del df_DP['dividend_payable']
    del df_DP['market_cap']
    # 改个名字
    df_DP.columns = ['DP', 'DP_sorted_rank']
    return df_DP


# 5CFP
# 得到一个dataframe：包含股票代码、现金收益率CFP和CFP_sorted_rank
# 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
def get_df_CFP(stock_list, context, asc):
    # 获得市现率pcf_ratio cashflow/price
    df_CFP = get_fundamentals(query(valuation.code, valuation.pcf_ratio
                                    ).filter(valuation.code.in_(stock_list)))
    # 获得pcf倒数
    df_CFP['CFP'] = df_CFP['pcf_ratio'].apply(lambda x: 1 / x)
    # 删除nan，以备数据中某项没有产生nan
    df_CFP = df_CFP[pd.notnull(df_CFP['CFP'])]
    # 生成序列数字排名
    df_CFP['CFP_sorted_rank'] = df_CFP['CFP'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_CFP.index = df_CFP.code
    # 删除无用数据
    del df_CFP['code']
    return df_CFP


# 6PS
# 得到一个dataframe：包含股票代码、P/SALES（PS市销率TTM）和PS_sorted_rank
# 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
def get_df_PS(stock_list, context, asc):
    # 获得市销率TTMps_ratio
    df_PS = get_fundamentals(query(valuation.code, valuation.ps_ratio
                                   ).filter(valuation.code.in_(stock_list)))
    # 删除nan，以备数据中某项没有产生nan
    df_PS = df_PS[pd.notnull(df_PS['ps_ratio'])]
    # 生成排名序数
    df_PS['PS_sorted_rank'] = df_PS['ps_ratio'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_PS.index = df_PS.code
    # 删除无用数据
    del df_PS['code']
    # 改个名字
    df_PS.columns = ['PS', 'PS_sorted_rank']
    return df_PS


# 7ALR
# 得到一个dataframe：包含股票代码、资产负债率(asset-liability ratio, ALR)
# 和ALR_sorted_rank
# 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
def get_df_ALR(stock_list, context, asc):
    # 获得total_liability和total_assets 负债合计(元)和资产总计(元)
    df_ALR = get_fundamentals(query(balance.code, balance.total_liability, balance.total_assets
                                    ).filter(balance.code.in_(stock_list)))
    # 复制一个dataframe，按对应项目排序
    df_ALR['ALR'] = df_ALR['total_liability'] / df_ALR['total_assets']
    # 删除nan
    df_ALR = df_ALR.dropna()
    # 生成排名序数
    df_ALR['ALR_sorted_rank'] = df_ALR['ALR'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_ALR.index = df_ALR.code
    # 删除无用数据
    del df_ALR['code']
    del df_ALR['total_liability']
    del df_ALR['total_assets']
    return df_ALR


# 8FACR
# 得到一个dataframe：包含股票代码、固定资产比例(fixed assets to capital ratio, FACR )
# 和FACR_sorted_rank
# 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
def get_df_FACR(stock_list, context, asc):
    # 获得fixed_assets和total_assets 固定资产(元)和资产总计(元)
    df_FACR = get_fundamentals(query(balance.code, balance.fixed_assets, balance.total_assets
                                     ).filter(balance.code.in_(stock_list)))
    # 根据公式计算
    df_FACR['FACR'] = df_FACR['fixed_assets'] / df_FACR['total_assets']
    # 删除nan
    df_FACR = df_FACR.dropna()
    # 生成排名序数
    df_FACR['FACR_sorted_rank'] = df_FACR['FACR'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_FACR.index = df_FACR.code
    # 删除无用数据
    del df_FACR['code']
    del df_FACR['fixed_assets']
    del df_FACR['total_assets']
    # 改个名字
    df_FACR.columns = ['FACR', 'FACR_sorted_rank']
    return df_FACR


# 9CMC
# 得到一个dataframe：包含股票代码、流通市值CMC和CMC_sorted_rank
# 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
def get_df_CMC(stock_list, context, asc):
    # 获得流通市值 circulating_market_cap 流通市值(亿)
    df_CMC = get_fundamentals(query(valuation.code, valuation.circulating_market_cap
                                    ).filter(valuation.code.in_(stock_list)))
    # 删除nan
    df_CMC = df_CMC.dropna()
    # 生成排名序数
    df_CMC['CMC_sorted_rank'] = df_CMC['circulating_market_cap'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_CMC.index = df_CMC.code
    # 删除无用数据
    del df_CMC['code']
    # 改个名字
    df_CMC.columns = ['CMC', 'CMC_sorted_rank']
    return df_CMC


# 10MC
# 得到一个dataframe：包含股票代码、总市值MC和MC_sorted_rank
# 默认date = context.current_dt的前一天,使用默认值，避免未来函数，不建议修改
def get_df_MC(stock_list, context, asc):
    # 获得总市值 circulating_market_cap 流通市值(亿)
    df_MC = get_fundamentals(query(valuation.code, valuation.market_cap
                                   ).filter(valuation.code.in_(stock_list)))
    # 删除nan
    df_MC = df_MC.dropna()
    # 生成排名序数
    df_MC['MC_sorted_rank'] = df_MC['market_cap'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_MC.index = df_MC.code
    # 删除无用数据
    del df_MC['code']
    # 改个名字
    df_MC.columns = ['MC', 'MC_sorted_rank']
    return df_MC


# 11roe_ttm
def get_df_roe_ttm(stock_list, context, asc):
    df_roe_ttm = get_factor_values(stock_list, factors='roe_ttm',
                                   end_date=context.current_dt - datetime.timedelta(days=1), count=1)
    df_roe_ttm = df_roe_ttm['roe_ttm'].T
    df_roe_ttm.columns = ['roe_ttm']
    df_roe_ttm['roe_ttm_sorted_rank'] = df_roe_ttm['roe_ttm'].rank(ascending=asc, method='dense')
    return df_roe_ttm


# 12net_profit_to_total_operate_revenue_ttm
def get_df_net_profit_to_total_operate_revenue_ttm(stock_list, context, asc):
    df_net_profit_to_total_operate_revenue_ttm = get_factor_values(stock_list,
                                                                   factors='net_profit_to_total_operate_revenue_ttm',
                                                                   end_date=context.current_dt - datetime.timedelta(
                                                                       days=1), count=1)
    df_net_profit_to_total_operate_revenue_ttm = df_net_profit_to_total_operate_revenue_ttm[
        'net_profit_to_total_operate_revenue_ttm'].T
    df_net_profit_to_total_operate_revenue_ttm.columns = ['net_profit_to_total_operate_revenue_ttm']
    df_net_profit_to_total_operate_revenue_ttm['net_profit_to_total_operate_revenue_ttm_sorted_rank'] = \
    df_net_profit_to_total_operate_revenue_ttm['net_profit_to_total_operate_revenue_ttm'].rank(ascending=asc,
                                                                                               method='dense')
    return df_net_profit_to_total_operate_revenue_ttm


# 13operating_revenue_growth_rate
def get_df_operating_revenue_growth_rate(stock_list, context, asc):
    df_operating_revenue_growth_rate = get_factor_values(stock_list, factors='operating_revenue_growth_rate',
                                                         end_date=context.current_dt - datetime.timedelta(days=1),
                                                         count=1)
    df_operating_revenue_growth_rate = df_operating_revenue_growth_rate['operating_revenue_growth_rate'].T
    df_operating_revenue_growth_rate.columns = ['operating_revenue_growth_rate']
    df_operating_revenue_growth_rate['operating_revenue_growth_rate_sorted_rank'] = df_operating_revenue_growth_rate[
        'operating_revenue_growth_rate'].rank(ascending=asc, method='dense')
    return df_operating_revenue_growth_rate


# 14获取roa
def get_df_roa(stock_list, context, asc):
    # 获得总市值 circulating_market_cap 流通市值(亿)
    df_roa = get_fundamentals(query(valuation.code, indicator.roa
                                    ).filter(valuation.code.in_(stock_list)))
    # 删除nan
    df_roa = df_roa.dropna()
    # 生成排名序数
    df_roa['roa_sorted_rank'] = df_roa['roa'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_roa.index = df_roa.code
    # 删除无用数据
    del df_roa['code']
    # 改个名字
    df_roa.columns = ['roa', 'roa_sorted_rank']
    return df_roa


# 获取d_a
def get_df_d_a(stock_list, context, asc):
    df_d_a = get_fundamentals(
        query(valuation.code, balance.total_assets, balance.total_liability).filter(valuation.code.in_(stock_list)))
    df_d_a['d_a'] = df_d_a['total_liability'] / df_d_a['total_assets']
    # 删除nan
    df_d_a = df_d_a.dropna()
    # 生成排名序数
    df_d_a['d_a_sorted_rank'] = df_d_a['d_a'].rank(ascending=asc, method='dense')
    # 使用股票代码作为index
    df_d_a.index = df_d_a['code']
    # 删除无用数据
    del df_d_a['code']
    del df_d_a['total_assets']
    del df_d_a['total_liability']
    # 改个名字
    df_d_a.columns = ['d_a', 'd_a_sorted_rank']
    return df_d_a


# 获取经营活动现金流量ttm
def get_df_net_operate_cash_flow_ttm(stock_list, context, asc):
    df_net_operate_cash_flow_ttm = get_factor_values(stock_list, ['net_operate_cash_flow_ttm'],
                                                     end_date=context.current_dt - datetime.timedelta(days=1), count=1)
    df_net_operate_cash_flow_ttm = df_net_operate_cash_flow_ttm['net_operate_cash_flow_ttm'].T
    df_net_operate_cash_flow_ttm.columns = ['net_operate_cash_flow_ttm']
    df_net_operate_cash_flow_ttm['net_operate_cash_flow_ttm_sorted_rank'] = df_net_operate_cash_flow_ttm[
        'net_operate_cash_flow_ttm'].rank(ascending=asc, method='dense')

    return df_net_operate_cash_flow_ttm


# 去极值函数（3倍标准差去极值）
def winsorize(factor_data, factor, std=3, have_negative=True):
    '''
    去极值函数
    factor:以股票code为index，因子值为value的Series
    std为几倍的标准差，have_negative 为布尔值，是否包括负值
    输出Series
    '''
    r = factor_data[factor]
    if have_negative == False:
        r = r[r >= 0]
    else:
        pass
    # 取极值
    edge_up = r.mean() + std * r.std()
    edge_low = r.mean() - std * r.std()
    r[r > edge_up] = edge_up
    r[r < edge_low] = edge_low
    r = pd.DataFrame(r)
    return r


# z－score标准化函数：
def standardize(factor_data, factor, ty=2):
    '''
    s为Series数据
    ty为标准化类型:1 MinMax,2 Standard,3 maxabs
    '''
    temp = factor_data[factor]
    if int(ty) == 1:
        re = (temp - temp.min()) / (temp.max() - temp.min())
    elif ty == 2:
        re = (temp - temp.mean()) / temp.std()
    elif ty == 3:
        re = temp / 10 ** np.ceil(np.log10(temp.abs().max()))
    return pd.DataFrame(re)


# 获取所有的市值
def get_market_cap(context):
    q = query(
        valuation.code,
        valuation.market_cap
    ).filter(
        valuation.code.in_(g.feasible_stocks)
    )
    df = get_fundamentals(q, context.current_dt)
    df = df.set_index('code')
    return df


# 市值中性化函数
def neutralization(data_factor, factor, data_market_cap):
    data_market_cap['market_cap'] = data_market_cap['market_cap'].apply(lambda x: math.log(x))
    df = pd.concat([data_factor, data_market_cap], axis=1, join='inner')
    y = df[factor]
    x = df['market_cap']
    result = sm.OLS(y, x).fit()
    result = pd.DataFrame(result.resid)
    result.columns = [g.factor]
    return result


'''
================================================================================
每天收盘后
================================================================================
'''


# 每日收盘后要做的事情（本策略中不需要）
def after_trading_end(context):
    return
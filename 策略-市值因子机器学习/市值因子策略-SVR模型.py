# 克隆自聚宽文章：https://www.joinquant.com/post/10778
# 标题：【量化课堂】机器学习多因子策略
# 作者：JoinQuant量化课堂
import pandas as pd
import numpy as np
import math
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import jqdata


def initialize(context):
    set_params()
    set_backtest()
    run_daily(trade, 'every_bar')


def set_params():
    g.days = 0
    # 调仓频率:
    g.refresh_rate = 10
    g.invest_by_group = True
    g.quantile = 0.1
    # 分组
    # 第1组：1
    # 第2组：2
    # ... ...
    # 第n组：n
    g.group = 1
    # 持仓数（分组时失效）
    g.stocknum = 10


def set_backtest():
    # set_benchmark('000001.XSHG')
    set_benchmark('000985.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')


# 设置可行股票池：过滤掉当日停牌的股票
# 输入：initial_stocks为list类型,表示初始股票池； context（见API）
# 输出：unsuspened_stocks为list类型，表示当日未停牌的股票池，即：可行股票池
def set_feasible_stocks(initial_stocks, context):
    # 判断初始股票池的股票是否停牌，返回list
    paused_info = []
    # get_current_data() 只有在获取值的时候才能有值:
    current_data = get_current_data()
    # print ('current_data is:', current_data)
    for i in initial_stocks:
        # print ('i is :', i)
        # print ('current_data[i] is:',current_data[i].paused)
        paused_info.append(current_data[i].paused)
    df_paused_info = pd.DataFrame({'paused_info': paused_info}, index=initial_stocks)
    unsuspened_stocks = list(df_paused_info.index[df_paused_info.paused_info == False])
    return unsuspened_stocks


def trade(context):
    if g.days % 10 == 0:
        sample = get_index_stocks('000985.XSHG', date=None)
        # 过滤停牌的股票:
        sample = set_feasible_stocks(sample, context)
        q = query(valuation.code, valuation.market_cap, balance.total_assets - balance.total_liability,
                  balance.total_assets / balance.total_liability, income.net_profit, income.net_profit + 1,
                  indicator.inc_revenue_year_on_year, balance.development_expenditure).filter(
            valuation.code.in_(sample))
        df = get_fundamentals(q, date=None)
        df.columns = ['code', 'log_mcap', 'log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD']

        df['log_mcap'] = np.log(df['log_mcap'])
        df['log_NC'] = np.log(df['log_NC'])
        df['NI_p'] = np.log(np.abs(df['NI_p']))
        # df['NI_n'] = np.log(np.abs(df['NI_n'][df['NI_n']<0])) #选择某列满足某个条件的数据
        df['NI_n'] = np.abs(df['NI_p'][df['NI_p'] < 0])
        df['log_RD'] = np.log(df['log_RD'])
        df.index = df.code.values

        del df['code']  # del d
        df = df.fillna(0)

        # 这个是对哪组数据的操作呢? 是log_mcap还是g?
        df[df > 10000] = 10000
        df[df < -10000] = -10000

        # 申万行业
        industry_set = ['801010', '801020', '801030', '801040', '801050', '801080', '801110', '801120', '801130',
                        '801140', '801150', '801160', '801170', '801180', '801200', '801210', '801230', '801710',
                        '801720', '801730', '801740', '801750', '801760', '801770', '801780', '801790', '801880',
                        '801890']

        for i in range(len(industry_set)):
            industry = get_industry_stocks(industry_set[i], date=None)
            s = pd.Series([0] * len(df), index=df.index)
            s[set(industry) & set(df.index)] = 1  # Series的行索引取值
            df[industry_set[i]] = s  # DataFrame的列索引取值

        # 截面因子数据:自变量
        X = df[['log_NC', 'LEV', 'NI_p', 'NI_n', 'g', 'log_RD', '801010', '801020', '801030', '801040', '801050',
                '801080', '801110', '801120', '801130', '801140', '801150', '801160', '801170', '801180', '801200',
                '801210', '801230', '801710', '801720', '801730', '801740', '801750', '801760', '801770', '801780',
                '801790', '801880', '801890']]

        # 对数市值:因变量
        Y = df[['log_mcap']]

        # 数据缺失值填充:
        X = X.fillna(0)
        Y = Y.fillna(0)

        svr = SVR(kernel='rbf', gamma=0.1)
        model = svr.fit(X, Y)

        # randomforest = RandomForestRegressor(random_state=42,n_estimators=500,n_jobs=-1)
        # model = randomforest.fit(X, Y)

        # 一直都不太明白为什么要拿残差做做因子
        # factor = Y - pd.DataFrame(svr.predict(X), index = Y.index, columns = ['log_mcap'])
        factor = Y - pd.DataFrame(svr.predict(X), index=Y.index, columns=['log_mcap'])
        factor = factor.sort_index(by='log_mcap')
        ###  分组测试用 ##############
        if g.invest_by_group == True:
            len_secCodeList = len(list(factor.index))
            g.stocknum = int(len_secCodeList * g.quantile)

        start = g.stocknum * (g.group - 1)
        end = g.stocknum * g.group
        stockset = list(factor.index[start:end])

        # stockset = list(factor.index[:10])
        sell_list = list(context.portfolio.positions.keys())

        # 先卖出:
        for stock in sell_list:
            # if stock not in stockset[:g.stocknum]:
            if stock not in stockset:
                stock_sell = stock
                order_target_value(stock_sell, 0)

        # 为股票分配资金:
        if len(context.portfolio.positions) < g.stocknum:
            num = g.stocknum - len(context.portfolio.positions)
            cash = context.portfolio.cash / num
        else:
            cash = 0
            num = 0

        for stock in stockset[:g.stocknum]:
            if stock in sell_list:
                pass
            else:
                stock_buy = stock
                order_target_value(stock_buy, cash)
                num = num - 1
                if num == 0:
                    break
        g.days += 1
    else:
        g.days = g.days + 1    

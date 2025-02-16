import pandas as pd
import numpy as np
import datetime
from scipy.stats import percentileofscore
import warnings
warnings.filterwarnings('ignore')
from WindPy import w
w.start()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 设置字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
import quantstats as qs
import os


# 初始化quantstats库
qs.extend_pandas()
class backtest_longshort:
    def __init__(self, start_date, end_date, df_timing, index_code='000300.SH',positive = 1):
        self.start_date = start_date
        self.end_date = end_date
        self.index_code = index_code
        self.positive = positive
        self.df_index = self.load_index_data()  # Load chosen index data
        self.df_timing = df_timing
        self.df_merged = self.merge_data()
        self.calculate_net_value()


    def load_index_data(self):
        """Load and preprocess index data based on the selected code."""
        data = w.wsd(self.index_code, "open, close", self.start_date, self.end_date, "PriceAdj=F")
        index_name = self.get_index_name(self.index_code)

        #检查数据完整性
        if data.ErrorCode != 0 or len(data.Data) < 2:
            raise ValueError(f"Data error for index {self.index_code}.")
        
        df_index = pd.DataFrame({
            f'{index_name}_open': data.Data[0],  # 开盘价
            f'{index_name}_close': data.Data[1]  # 收盘价
            }, index=pd.to_datetime(data.Times))
        

        df_index = self.adj_date(df_index)
        df_index['ret_unadjusted'] = df_index[f'{index_name}_close'].pct_change(1)
        df_index['ret_adjusted'] = df_index[f'{index_name}_close']/df_index[f'{index_name}_open'] - 1
        # df_index['ret_adjusted'] = df_index[f'{index_name}_close'].pct_change(1)
        return df_index

    def get_index_name(self, index_code):
        """Return the name of the index based on the index code."""
        index_names = {
            '930050.CSI': '中证A50',
            '000001.SH': '上证指数',
            '000300.SH': '沪深300',
            '000852.SH': '中证1000',
            '399006.SZ': '创业板',
            '000922.CSI': '中证红利指数',
            '000510.SH':'中证A500'
        }
        return index_names.get(index_code, '指数')
    
    def adj_date(self, df):
        """Adjust DataFrame to the specified date range."""
        return df[(df.index >= self.start_date) & (df.index <= self.end_date)]

    def merge_data(self):
        df_merged = pd.merge(self.df_timing, self.df_index, how='right', left_index=True, right_index=True)
        #初始化position_ret
        df_merged['position_ret'] = 0.0

        # 获取 Position_index 列的前一行值，用于比较
        df_merged['Position_prev'] = df_merged['Position'].shift(1)

        # 条件1: 如果Position没有变后
        condition_no_change = df_merged['Position_prev'] == df_merged['Position']
        df_merged.loc[condition_no_change, 'position_ret'] = \
        df_merged['Position'] * df_merged['ret_unadjusted'] - df_merged['transaction_fee']

        # 条件2: 如果 Position 发生变化
        condition_changed  = df_merged['Position_prev'] != df_merged['Position']
        df_merged.loc[condition_changed, 'position_ret'] = \
        df_merged['Position'] * df_merged['ret_adjusted'] - df_merged['transaction_fee']

        # 删除辅助列
        df_merged.drop(columns=['Position_prev'], inplace=True)

        return df_merged


    def calculate_net_value(self):
        """Calculate net values based on positions and returns, with transaction fees."""
        initial_value = 1.0
        self.df_merged['净值'] = initial_value

        for i in range(1, len(self.df_merged)):
            previous_value = self.df_merged.iloc[i-1]['净值']
            ret = self.df_merged.iloc[i]['position_ret']
            transaction_fee = self.df_merged.iloc[i]['transaction_fee']
            
            # 计算新的净值，扣除交易费用
            new_value = previous_value * (1 + ret - transaction_fee)
            self.df_merged.iloc[i, self.df_merged.columns.get_loc('净值')] = new_value

    def max_drawdown(self, series):
        cumulative = (series + 1).cumprod()
        return -((cumulative / cumulative.cummax()) - 1).min()

    def getRetIndicators(self, df, return_name='', benchmark_ret=None):
        '''使用quantstats计算并返回年化收益率、年化波动率、信息比率、胜率、最大回撤率和超额回撤水平'''
        
        # 获取策略收益率
        strategy_returns = df['strategy_ret']
        
        # 使用QuantStats计算年化收益率和波动率
        ret = qs.stats.cagr(strategy_returns)  # 年化收益率
        vol = qs.stats.volatility(strategy_returns)  # 年化波动率

        # 收益波动比
        ret_vol_ratio = ret / vol if vol != 0 else np.nan

        # 计算信息比率和超额回撤率
        if benchmark_ret is not None:
            ir = qs.stats.information_ratio(strategy_returns, benchmark=benchmark_ret)
            excess_ret_value = (strategy_returns.add(1).cumprod()[-1] - benchmark_ret.add(1).cumprod()[-1])  # 计算超额收益率
            excess_ret = strategy_returns - benchmark_ret
            # #年化超额收益
            excess_ret_ann = qs.stats.cagr(excess_ret)
            
            #超额回撤
            excess_mdd = qs.stats.max_drawdown(strategy_returns - benchmark_ret)  # 计算超额回撤水平


        else:
            raise ValueError("Unsupported method, please provide benchmark returns.")

        # 计算最大回撤率
        mdd = qs.stats.max_drawdown(strategy_returns)

        # 收益回撤比
        ret_mdd_ratio = ret / mdd if mdd != 0 else np.nan

        # 计算胜率
        winrate = (strategy_returns > 0).mean()  # 胜率

        # 最长亏损周期
        value = df['净值']
        loss_days_list = []
        loss_start_day_list = []
        loss_end_day_list = []
        for i in range(len(value)):
            value_list_i = value[i:]
            value_list_i = [1 if x > value_list_i[0] else 0 for x in value_list_i[1:]]
            if 1 in value_list_i:
                loss_days = value_list_i.index(1)
            else:
                loss_days = len(value_list_i)
            loss_days_list.append(loss_days)
            loss_start_day_list.append(i)
            loss_end_day_list.append(i + loss_days)
        max_loss_days = max(loss_days_list)
        max_loss_duration = max_loss_days / 20  # 转换为月

        #亏损时间占比
        loss_time_ratio_list = []
        for i in range(len(value)):
            value_list_i = value[i:]
            value_list_i = [1 if x > value_list_i[0] else 0 for x in value_list_i[1:]]
            if len(value_list_i) < min(240 * 3, len(value_list_i)):  # 修改此处长度检查
                continue
            value_list_i = value_list_i[:min(240 * 3, len(value_list_i))]  # 确保长度足够
            if len(value_list_i) > 0:
                loss_time_ratio = 1 - sum(value_list_i) / len(value_list_i)
                loss_time_ratio_list.append(loss_time_ratio)
        loss_time_ratio = np.mean(loss_time_ratio_list) if loss_time_ratio_list else np.nan

        #  一年投资期末亏损超过10%的概率
        df_rtn_next_year = df['净值'].pct_change(min(252, len(df))).shift(-min(252, len(df))).dropna()
        df_loss = (df_rtn_next_year < -0.1).replace([True, False], [1, 0])
        loss_prob_10pct = sum(df_loss) / len(df_loss) if len(df_loss) > 0 else np.nan

        # 计算自定义指标（胜率、赔率、换手率）
        custom_metrics = self.calculate_custom_metrics(strategy_returns, df['Position'], 252)

        # 返回结果，包括新增的超额回撤率和自定义指标
        index_list = ['年化收益率', '年化波动率', '信息比率', '胜率', '最大回撤率', '超额净值','年化超额收益率', '超额回撤水平', '收益波动比', '收益回撤比', '最长亏损周期(月)', '亏损时间占比', '1年投资期末亏损超过10%的概率', '赔率', '换手率']
        index_list = [return_name + i for i in index_list]
        return pd.Series(index=index_list, data=[ret, vol, ir, winrate, mdd, excess_ret_value ,excess_ret_ann, excess_mdd, ret_vol_ratio, ret_mdd_ratio, max_loss_duration, loss_time_ratio, loss_prob_10pct, custom_metrics['赔率'], custom_metrics['换手率']])

    def calculate_custom_metrics(self, returns, positions, frequency):
        '''
        计算自定义指标，如胜率、赔率和换手率。
        
        输入:
        - returns: pandas.Series，策略的收益率序列。
        - positions: pandas.Series，策略的持仓信号序列。
        - frequency: int，年度交易日数量，用于换手率计算，默认252（日度）。

        返回:
        - custom_metrics: pandas.Series，包含胜率、赔率和换手率的序列。
        '''
        pos_changes = positions.diff().abs()


        turnover_events = pos_changes[pos_changes != 0].count()

        total_days = len(returns.dropna())
        average_win = returns[returns > 0].mean()
        average_loss = -returns[returns < 0].mean()
        odds_ratio = average_win / average_loss if average_loss > 0 else np.nan
        turnover_rate = turnover_events / total_days if total_days > 0 else np.nan
        return pd.Series({
            '赔率': odds_ratio,
            '换手率': turnover_rate
        }, name='Value')

    def analyze_performance(self, strategy_returns, benchmark_returns, risk_free_rate):
        '''
        使用QuantStats分析策略表现。
        
        输入:
        - strategy_returns: pandas.Series，策略的日收益率序列。
        - benchmark_returns: pandas.Series，基准的日收益率序列。
        - risk_free_rate: float，年化无风险利率。
        
        返回:
        - performance_metrics: pandas.DataFrame，策略和基准的性能指标。
        '''
        qs.extend_pandas()
        strategy_performance = qs.reports.metrics(strategy_returns, mode='full', rf=risk_free_rate, display=False)
        benchmark_performance = qs.reports.metrics(benchmark_returns, mode='full', rf=risk_free_rate, display=False)
        qs.plots.returns(strategy_returns, benchmark_returns, cumulative=True)
        performance_metrics = pd.concat([strategy_performance, benchmark_performance], axis=1)
        performance_metrics.columns = ['Strategy', 'Benchmark']
        return performance_metrics

    def calculate_performance(self):
        """Calculate performance metrics by year and plot net value and excess returns."""
        df_performance = self.df_merged.copy()
        df_performance['strategy_ret'] = df_performance['净值'].pct_change(1)

        # 获取基准指数的每日收益率
        benchmark_returns = self.df_merged[f'{self.get_index_name(self.index_code)}_close'].pct_change(1)

        df_performance['Year'] = df_performance.index.year

        # 按年份分组并计算指标
        results = {}
        for year, group in df_performance.groupby('Year'):
            results[year] = self.getRetIndicators(group, benchmark_ret=benchmark_returns.loc[group.index])

        # 全时间区间的指标
        full_period_metrics = self.getRetIndicators(df_performance, benchmark_ret=benchmark_returns)
        result_df = pd.concat([pd.DataFrame(results), full_period_metrics.rename('Full Period')], axis=1)

        # 计算超额收益
        df_performance['超额收益'] = df_performance['strategy_ret'] - benchmark_returns
        df_performance['累计超额收益'] = df_performance['超额收益'].cumsum()

        # 计算上升市（Cluster=1）和下跌市（Cluster=0）的累计超额收益
        df_performance['累计超额收益_上升市'] = df_performance[df_performance['Position'] == 1]['超额收益'].cumsum()
        df_performance['累计超额收益_下跌市'] = df_performance[df_performance['Position'] == 0]['超额收益'].cumsum()


        # 绘制净值曲线和基准曲线
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(self.df_merged.index, self.df_merged['净值'], label='择时策略', color='blue')
        ax1.plot(self.df_merged.index, (self.df_merged[f'{self.get_index_name(self.index_code)}_close'].pct_change(1) + 1).cumprod(), label=f'{self.get_index_name(self.index_code)}（基准）', color='red')
        ax1.set_title(f'择时策略 vs {self.get_index_name(self.index_code)}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper left')
        # 设置日期格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
        plt.gcf().autofmt_xdate()


        # 第二张图：绘制所有时间的累计超额收益
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.fill_between(df_performance.index, 0, df_performance['累计超额收益'], color='green', alpha=0.5, label='累计超额收益')
        ax2.set_ylabel('累计超额收益')
        ax2.legend()
        # 设置日期格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
        plt.gcf().autofmt_xdate()


        # 第三张图：分别绘制上升市和下跌市的累计超额收益
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.fill_between(df_performance.index, 0, df_performance['累计超额收益_上升市'], color='blue', alpha=0.5, label='上升市累计超额收益')
        ax3.fill_between(df_performance.index, 0, df_performance['累计超额收益_下跌市'], color='red', alpha=0.5, label='下跌市累计超额收益')
        ax3.set_ylabel('累计超额收益')
        ax3.legend()
        # 设置日期格式
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
        plt.gcf().autofmt_xdate()

        return result_df



class backtest_stockbond:
    def __init__(self, start_date, end_date, df_timing, index_code='000300.SH', dynamic=True):
        self.start_date = start_date
        self.end_date = end_date
        self.index_code = index_code
        self.dynamic = dynamic
        self.df_index = self.load_index_data()  # Load chosen index data
        self.df_timing = df_timing              # 策略中的Cluster数据
        self.df_merged = self.merge_data()      # 合并数据
        self.calculate_net_value()              # 计算净值

    def load_index_data(self):
        """Load and preprocess index data based on the selected code."""
        data = w.wsd(self.index_code, "open, close", self.start_date, self.end_date, "PriceAdj=F")
        index_name = self.get_index_name(self.index_code)

        #检查数据完整性
        if data.ErrorCode != 0 or len(data.Data) < 2:
            raise ValueError(f"Data error for index {self.index_code}.")
        
        df_index = pd.DataFrame({
            f'{index_name}_open': data.Data[0],  # 开盘价
            f'{index_name}_close': data.Data[1]  # 收盘价
            }, index=pd.to_datetime(data.Times))
        

        df_index = self.adj_date(df_index)
        df_index['index_ret_unadjusted'] = df_index[f'{index_name}_close'].pct_change(1)
        df_index['index_ret_adjusted'] = df_index[f'{index_name}_close']/df_index[f'{index_name}_open'] - 1
        return df_index
    
    def get_index_name(self, index_code):
        """Return the name of the index based on the index code."""
        index_names = {
            '930040.CSI': '中证A50',
            '000001.SH': '上证指数',
            '000300.SH': '沪深300',
            '000852.SH': '中证1000',
            '399006.SZ': '创业板',
            '000922.CSI': '中证红利指数',
            '000510.SH':'中证A500'
        }
        return index_names.get(index_code, '指数')
    
    def adj_date(self, df):
        """Adjust DataFrame to the specified date range."""
        return df[(df.index >= self.start_date) & (df.index <= self.end_date)]


    def merge_data(self):
        """Merge timing indicators with index and bond data."""
        df_merged = pd.merge(self.df_timing, self.df_index, how='right', left_index=True, right_index=True)
    
        # 初始化 position_ret
        df_merged['position_ret'] = 0.0
        
        # 获取 Position_index 列的前一行值，用于比较
        df_merged['Position_index_prev'] = df_merged['Position_index'].shift(1)
        
        # 条件1: 如果之前是1，现在也是1
        condition_no_change = (df_merged['Position_index_prev'] == 1) & (df_merged['Position_index'] == 1)
        df_merged.loc[condition_no_change, 'position_ret'] = \
            df_merged['Position_index'] * df_merged['index_ret_unadjusted'] - df_merged['transaction_fee_index']
        
        # 条件2: 如果之前是0，现在变成1
        condition_new_position = (df_merged['Position_index_prev'] == 0) & (df_merged['Position_index'] == 1)
        df_merged.loc[condition_new_position, 'position_ret'] = \
            df_merged['Position_index'] * df_merged['index_ret_adjusted'] - df_merged['transaction_fee_index']
        
        # 加入 Bond 的收益计算
        df_merged['position_ret'] += df_merged['Position_Bond'] * df_merged['Bond_ret'] - df_merged['transaction_fee_Bond']
        
        # 删除辅助列
        df_merged.drop(columns=['Position_index_prev'], inplace=True)
        
        return df_merged


    def calculate_net_value(self):
        """Calculate net values based on positions and returns, with transaction fees."""
        initial_value = 1.0

        self.df_merged['净值'] = initial_value

        for i in range(1, len(self.df_merged)):
            previous_value = self.df_merged.iloc[i-1]['净值']
            ret = self.df_merged.iloc[i]['position_ret']

            # 计算新的净值，扣除交易费用
            new_value = previous_value * (1 + ret)
            self.df_merged.iloc[i, self.df_merged.columns.get_loc('净值')] = new_value

    def max_drawdown(self, series):
        cumulative = (series + 1).cumprod()
        return -((cumulative / cumulative.cummax()) - 1).min()

    def getRetIndicators(self, df, return_name='', benchmark_ret=None):
        '''使用quantstats计算并返回年化收益率、年化波动率、信息比率、胜率和最大回撤率'''

        # 获取策略收益率
        strategy_returns = df['strategy_ret']
        
        # 使用QuantStats计算年化收益率和波动率
        ret = qs.stats.cagr(strategy_returns)  # 年化收益率
        vol = qs.stats.volatility(strategy_returns)  # 年化波动率
        
        # 收益波动比
        ret_vol_ratio = ret / vol if vol != 0 else np.nan

        # 计算信息比率，基准为沪深300的收益率
        if benchmark_ret is not None:
            ir = qs.stats.information_ratio(strategy_returns, benchmark=benchmark_ret)
            excess_ret_value = (strategy_returns.add(1).cumprod()[-1] - benchmark_ret.add(1).cumprod()[-1])  # 计算超额收益率
            excess_ret = strategy_returns - benchmark_ret
            # #年化超额收益
            excess_ret_ann = qs.stats.cagr(excess_ret)
            excess_mdd = qs.stats.max_drawdown(strategy_returns - benchmark_ret)  # 计算超额回撤水平
        else:
            raise ValueError("Unsupported method, please provide benchmark returns.")

        # 计算最大回撤率
        mdd = qs.stats.max_drawdown(strategy_returns)

        # 收益回撤比
        ret_mdd_ratio = ret / mdd if mdd != 0 else np.nan
        # 计算胜率
        winrate = (strategy_returns > 0).mean()  # 胜率

        # 最长亏损周期
        value = df['净值']
        loss_days_list = []
        loss_start_day_list = []
        loss_end_day_list = []
        for i in range(len(value)):
            value_list_i = value[i:]
            value_list_i = [1 if x > value_list_i[0] else 0 for x in value_list_i[1:]]
            if 1 in value_list_i:
                loss_days = value_list_i.index(1)
            else:
                loss_days = len(value_list_i)
            loss_days_list.append(loss_days)
            loss_start_day_list.append(i)
            loss_end_day_list.append(i + loss_days)
        max_loss_days = max(loss_days_list)
        max_loss_duration = max_loss_days / 20  # 转换为月

        #亏损时间占比
        loss_time_ratio_list = []
        for i in range(len(value)):
            value_list_i = value[i:]
            value_list_i = [1 if x > value_list_i[0] else 0 for x in value_list_i[1:]]
            if len(value_list_i) < min(240 * 3, len(value_list_i)):  # 修改此处长度检查
                continue
            value_list_i = value_list_i[:min(240 * 3, len(value_list_i))]  # 确保长度足够
            if len(value_list_i) > 0:
                loss_time_ratio = 1 - sum(value_list_i) / len(value_list_i)
                loss_time_ratio_list.append(loss_time_ratio)
        loss_time_ratio = np.mean(loss_time_ratio_list) if loss_time_ratio_list else np.nan

        #  一年投资期末亏损超过10%的概率
        df_rtn_next_year = df['净值'].pct_change(min(252, len(df))).shift(-min(252, len(df))).dropna()
        df_loss = (df_rtn_next_year < -0.1).replace([True, False], [1, 0])
        loss_prob_10pct = sum(df_loss) / len(df_loss) if len(df_loss) > 0 else np.nan

        # 计算自定义指标（胜率、赔率、换手率）
        custom_metrics = self.calculate_custom_metrics(strategy_returns, df)

        # 返回结果，包括新增的超额回撤率和自定义指标
        index_list = ['年化收益率', '年化波动率', '信息比率', '胜率', '最大回撤率', '超额净值','年化超额收益率', '超额回撤水平', '收益波动比', '收益回撤比', '最长亏损周期(月)', '亏损时间占比', '1年投资期末亏损超过10%的概率', '赔率', '换手率']
        index_list = [return_name + i for i in index_list]
        return pd.Series(index=index_list, data=[ret, vol, ir, winrate, mdd, excess_ret_value ,excess_ret_ann, excess_mdd, ret_vol_ratio, ret_mdd_ratio, max_loss_duration, loss_time_ratio, loss_prob_10pct, custom_metrics['赔率'], custom_metrics['换手率']])

    def calculate_custom_metrics(self, returns, df):
        '''
        计算自定义指标，如胜率、赔率和换手率。
        
        输入:
        - returns: pandas.Series，策略的收益率序列。
        - positions: pandas.Series，策略的持仓信号序列。
        - frequency: int，年度交易日数量，用于换手率计算，默认252（日度）。

        返回:
        - custom_metrics: pandas.Series，包含胜率、赔率和换手率的序列。
        '''
        pos_changes_index = df['Position_index'].diff().abs()
        pos_changes_bond = df['Position_Bond'].diff().abs()

        # 换手率计算
        turnover_events_index = pos_changes_index.sum()  # 股票仓位的变动次数之和

        turnover_events_bond = pos_changes_bond.sum()    # 债券仓位的变动次数之和
        # 总换手率
        total_days = len(self.df_merged)
        turnover_rate_index = turnover_events_index / total_days  # 股票换手率
        turnover_rate_bond = turnover_events_bond / total_days    # 债券换手率
        turnover_rate = turnover_rate_index

        total_days = len(returns.dropna())
        average_win = returns[returns > 0].mean()
        average_loss = -returns[returns < 0].mean()
        odds_ratio = average_win / average_loss if average_loss > 0 else np.nan
        return pd.Series({
            '赔率': odds_ratio,
            '换手率': turnover_rate
        }, name='Value')
    
    def analyze_performance(self, strategy_returns, benchmark_returns, risk_free_rate):
        '''
        使用QuantStats分析策略表现。
        
        输入:
        - strategy_returns: pandas.Series，策略的日收益率序列。
        - benchmark_returns: pandas.Series，基准的日收益率序列。
        - risk_free_rate: float，年化无风险利率。
        
        返回:
        - performance_metrics: pandas.DataFrame，策略和基准的性能指标。
        '''
        qs.extend_pandas()
        strategy_performance = qs.reports.metrics(strategy_returns, mode='full', rf=risk_free_rate, display=False)
        benchmark_performance = qs.reports.metrics(benchmark_returns, mode='full', rf=risk_free_rate, display=False)
        qs.plots.returns(strategy_returns, benchmark_returns, cumulative=True)
        performance_metrics = pd.concat([strategy_performance, benchmark_performance], axis=1)
        performance_metrics.columns = ['Strategy', 'Benchmark']
        return performance_metrics
    
    def calculate_performance(self):
        """Calculate performance metrics by year."""
        df_performance = self.df_merged.copy()
        df_performance['strategy_ret'] = df_performance['净值'].pct_change(1)

        # 获取沪深300的每日收益率作为基准
        benchmark_returns = self.df_merged[f'{self.get_index_name(self.index_code)}_close'].pct_change(1)

        df_performance['Year'] = df_performance.index.year
        # 按年份分组并计算指标
        results = {}
        for year, group in df_performance.groupby('Year'):
            results[year] = self.getRetIndicators(group, benchmark_ret=benchmark_returns.loc[group.index])

          # 全时间区间的指标
        full_period_metrics = self.getRetIndicators(df_performance, benchmark_ret=benchmark_returns)
        result_df = pd.concat([pd.DataFrame(results), full_period_metrics.rename('Full Period')], axis=1)

        # 计算超额收益
        df_performance['超额收益'] = df_performance['strategy_ret'] - benchmark_returns
        df_performance['累计超额收益'] = df_performance['超额收益'].cumsum()

        # 计算上升市（Cluster=1）和下跌市（Cluster=0）的累计超额收益
        df_performance['累计超额收益_上升市'] = df_performance[df_performance['Position_index'] == 1]['超额收益'].cumsum()
        df_performance['累计超额收益_下跌市'] = df_performance[df_performance['Position_index'] == 0]['超额收益'].cumsum()        

        # 绘制净值曲线和基准曲线
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(self.df_merged.index, self.df_merged['净值'], label='择时策略', color='blue')
        ax1.plot(self.df_merged.index, (self.df_merged[f'{self.get_index_name(self.index_code)}_close'].pct_change(1) + 1).cumprod(), label=f'{self.get_index_name(self.index_code)}（基准）', color='red')
        ax1.set_title(f'择时策略 vs {self.get_index_name(self.index_code)}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper left')
        # 设置日期格式
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.close(fig1)

        # 第二张图：绘制所有时间的累计超额收益
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.fill_between(df_performance.index, 0, df_performance['累计超额收益'], color='green', alpha=0.5, label='累计超额收益')
        ax2.set_title('累计超额收益曲线')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('累计超额收益')
        ax2.legend()
        # 设置日期格式
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.close(fig2)

        # 第三张图：分别绘制上升市和下跌市的累计超额收益
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.fill_between(df_performance.index, 0, df_performance['累计超额收益_上升市'], color='blue', alpha=0.5, label='上升市累计超额收益')
        ax3.fill_between(df_performance.index, 0, df_performance['累计超额收益_下跌市'], color='red', alpha=0.5, label='下跌市累计超额收益')
        ax3.set_title('上升市和下跌市的累计超额收益曲线')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('累计超额收益')
        ax3.legend()
        # 设置日期格式
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.close(fig3)

        print(f"图像已成功保存")

        return result_df


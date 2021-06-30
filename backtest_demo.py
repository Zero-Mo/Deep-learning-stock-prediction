import time

from Utils import *

class Backtest:
    def __init__(self, start_cash=300000, fee=0.0003):
        self.index_name = 'hs300'
        self.index_code = 'sh.000300'
        self.base_data_path = './data/'
        self.data_path = './data/stocks_demo/'
        self.fee = fee
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name))
        self.stocks_codes = self.stocks['code']
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.index_code))
        self.trading_dates = self.index['date']
        self.today = (self.trading_dates[0], 0)
        self.start_cash = start_cash
        self.cash = pd.DataFrame(self.trading_dates)
        self.cash['cash'] = start_cash
        self.cash = self.cash.set_index('date')
        self.position = pd.DataFrame()
        self.position['date'] = self.trading_dates
        for stock_code in self.stocks_codes:
            self.position[stock_code] = 0
        self.position = self.position.set_index('date')

    def stocks_data(self, stock_codes, trading_date: str) -> pd.DataFrame():
        if stock_codes.empty:
            return pd.DataFrame(None)
        stocks = pd.DataFrame()
        for stock_code in stock_codes:
            data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            if trading_date in data.index:
                stocks = stocks.append(data.loc[trading_date], ignore_index=True)
        stocks = stocks.set_index('code')
        return stocks

    def buy(self, stocks_codes) -> bool:
        trading_date = self.today[0]
        stocks = self.stocks_data(stocks_codes, trading_date)
        if stocks.empty:
            return False
        n = len(stocks_codes)
        single = self.cash.loc[trading_date, 'cash'] // n
        for stock_code in stocks_codes:
            open_price = stocks.loc[stock_code, 'open']
            quantity = ((single / (1+self.fee)) / open_price) // 100
            self.position.loc[trading_date, stock_code] += quantity * 100
            self.cash.loc[trading_date, 'cash'] -= open_price * quantity * 100
        del stocks
        return True

    def sell(self, stocks_codes):
        trading_date = self.today[0]
        stocks = self.stocks_data(self.stocks_codes, trading_date)
        cash = 0
        sell_num = 0
        for stock_code in self.stocks_codes:
            position = self.position.loc[trading_date, stock_code]
            if position != 0:
                if stock_code not in stocks_codes:
                    cash += position * stocks.loc[stock_code, 'open'] * (1-self.fee)
                    self.position.loc[trading_date, stock_code] = 0
                    sell_num += 1
                else:
                    stocks_codes = stocks_codes.drop(stock_code)
        self.cash.loc[trading_date, 'cash'] += cash
        del stocks
        return stocks_codes, sell_num

    def next_day(self) -> str:
        cash = self.cash.loc[self.today[0], 'cash']
        position = self.position.loc[self.today[0]]
        today = self.today[1]
        if today < len(self.trading_dates) - 1:
            self.today = (self.trading_dates[today+1], today+1)
            self.cash.loc[self.today[0], 'cash'] = cash
            self.position.loc[self.today[0]] = position
            return self.today[0]

    def calculate(self):
        basic_position = self.start_cash
        position = []
        for i, trading_date in enumerate(self.trading_dates):
            if i % 5 != 0:
                continue
            print('当前计算交易日: '+trading_date, end='\r')
            stock_data = self.stocks_data(self.stocks_codes, trading_date)
            daily_position = self.cash.loc[trading_date, 'cash']
            for stock_code in self.stocks_codes:
                quantity = self.position.loc[trading_date, stock_code]
                if quantity != 0:
                    daily_position += quantity * stock_data.loc[stock_code, 'close']
            position.append(daily_position)
        position = pd.Series(position)
        position /= basic_position
        return position
if __name__ == '__main__':
    start_time = time.time()
    bt1 = Backtest(start_cash=10000000, fee=0.0003)
    bt2 = Backtest(start_cash=10000000, fee=0.0003)
    bt3 = Backtest(start_cash=10000000, fee=0.0003)
    bt4 = Backtest(start_cash=10000000, fee=0.0003)
    # bt5 = Backtest(start_cash=10000000, fee=0.0003)
    # bt6 = Backtest(start_cash=10000000, fee=0.0003)
    # bt7 = Backtest(start_cash=10000000, fee=0.0003)
    # bt8 = Backtest(start_cash=10000000, fee=0.0003)
    # bt9 = Backtest(start_cash=10000000, fee=0.0003)
    # bt10 = Backtest(start_cash=10000000, fee=0.0003)
    # bt11 = Backtest(start_cash=10000000, fee=0.0003)
    # bt12 = Backtest(start_cash=10000000, fee=0.0003)

    strategy = Strategy(data_days=10)
    for date_key, date in bt1.trading_dates.items():
        if date_key % 10 == 0:
            print('当前交易日: ' + date)
            # print('因子模型选股中...')
            # chosen1 = strategy.choose_by_bm(bt1.today, 90)
            # chosen2 = strategy.choose_by_mf(bt2.today, 90)
            # chosen3 = strategy.choose_by_tr(bt3.today, 90)
            print('神经网络模型选股中...')
            # chosen4 = strategy.choose_by_cnn(bt4.today, 90)
            # chosen5 = strategy.choose_by_rnn_relu(bt5.today, 90)
            # chosen6 = strategy.choose_by_resnet18(bt6.today, 90)
            chosen1 = strategy.choose_by_densenet(bt1.today, 90)
            # chosen8 = strategy.choose_by_gru(bt8.today, 90)
            chosen3 = strategy.choose_by_lstm(bt3.today, 90)
            chosen4 = strategy.choose_by_bilstm(bt4.today, 90)
            chosen2 = strategy.choose_by_lstm2(bt2.today, 90)

            # print('集成学习模型选股中...')
            # chosen12 = strategy.choose_by_ensemble(bt12.today)
            # to_buy1, sell1 = bt1.sell(chosen1)
            # print('价值因子(BM)选股模型卖出', sell1, '只股票')
            # to_buy2, sell2 = bt2.sell(chosen2)
            # print('动量因子(MF)选股模型卖出', sell2, '只股票')
            # to_buy3, sell3 = bt3.sell(chosen3)
            # print('换手率因子(TR)选股模型卖出', sell3, '只股票')
            # to_buy4, sell4 = bt4.sell(chosen4)
            # print('CNN选股模型卖出', sell4, '只股票')
            # to_buy5, sell5 = bt5.sell(chosen5)
            # print('RNN_relu选股模型卖出', sell5, '只股票')
            # to_buy6, sell6 = bt6.sell(chosen6)
            # print('ResNet18选股模型卖出', sell6, '只股票')
            to_buy1, sell1 = bt1.sell(chosen1)
            print('DenseNet选股模型卖出', sell1, '只股票')
            # to_buy8, sell8 = bt8.sell(chosen8)
            # print('GRU选股模型卖出', sell7, '只股票')
            to_buy3, sell3 = bt3.sell(chosen3)
            print('LSTM选股模型卖出', sell3, '只股票')
            to_buy4, sell4 = bt4.sell(chosen4)
            print('BILSTM选股模型卖出', sell4, '只股票')
            to_buy2, sell2 = bt2.sell(chosen2)
            print('LSTM2选股模型卖出', sell2, '只股票')
            # to_buy12, sell12 = bt12.sell(chosen12)
            # print('集成学习选股模型卖出', sell12, '只股票')

            # bt1.buy(to_buy1)
            # print('价值因子(BM)选股模型买入', len(to_buy1), '只股票')
            # bt2.buy(to_buy2)
            # print('动量因子(MF)选股模型买入', len(to_buy2), '只股票')
            # bt3.buy(to_buy3)
            # print('换手率因子(TR)选股模型买入', len(to_buy3), '只股票')
            # bt4.buy(to_buy4)
            # print('CNN选股模型买入', len(to_buy4), '只股票')
            # bt5.buy(to_buy5)
            # print('RNN_relu选股模型买入', len(to_buy5), '只股票')
            # bt6.buy(to_buy6)
            # print('ResNet18选股模型买入', len(to_buy6), '只股票')
            bt1.buy(to_buy1)
            print('DenseNet选股模型买入', len(to_buy1), '只股票')
            # bt8.buy(to_buy8)
            # print('GRU选股模型买入', len(to_buy8), '只股票')
            bt3.buy(to_buy3)
            print('LSTM选股模型买入', len(to_buy3), '只股票')
            bt4.buy(to_buy4)
            print('BILSTM选股模型买入', len(to_buy4), '只股票')
            bt2.buy(to_buy2)
            print('LSTM2选股模型买入', len(to_buy2), '只股票')

            # bt12.buy(to_buy12)
            # print('集成学习选股模型买入', len(to_buy12), '只股票')
        # bt1.next_day()
        # bt2.next_day()
        bt3.next_day()
        bt4.next_day()
        # bt5.next_day()
        # bt6.next_day()
        bt1.next_day()
        # bt8.next_day()
        # bt9.next_day()
        # bt10.next_day()
        bt2.next_day()
        # bt12.next_day()

    mid_time = time.time()
    span = mid_time - start_time
    print('回测模拟交易用时 {} 分 {} 秒'.format(int(span // 60), span % 60))
    # print('\n计算价值因子(BM)选股模型收益中')
    # bm_position = bt1.calculate()
    # print('\n计算动量因子(MF)选股模型收益中')
    # mf_position = bt2.calculate()
    # print('\n计算换手率因子(TR)选股模型收益中')
    # tr_position = bt3.calculate()
    # print('\n计算CNN选股模型收益中')
    # cnn_position = bt4.calculate()
    # print('\n计算RNN_relu选股模型收益中')
    # rnn_relu_position = bt5.calculate()
    # print('\n计算ResNet18选股模型收益中')
    # resnet18_position = bt6.calculate()
    print('\n计算DenseNet选股模型收益中')
    densenet_position = bt1.calculate()
    # print('\n计算GRU选股模型收益中')
    # gru_position = bt8.calculate()
    print('\n计算LSTM选股模型收益中')
    lstm_position = bt3.calculate()
    print('\n计算BILSTM选股模型收益中')
    bilstm_position = bt4.calculate()
    print('\n计算LSTM2选股模型收益中')
    lstm2_position = bt2.calculate()
    # print('\n计算集成学习选股模型收益中')
    # ensemble_position = bt12.calculate()
    basic_index_price = bt1.index['close'][0]
    index_price = bt1.index['close'][::5] / basic_index_price
    x = range(0, len(bt1.trading_dates), 5)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=[20, 15], dpi=160)
    plt.subplot(211)
    # plt.plot(x, index_price, 'k:', label='沪深300指数收益率')
    # plt.plot(x, bm_position, 'r:', label='价值因子(BM)选股模型持仓收益率')
    # plt.plot(x, mf_position, 'g:', label='动量因子(MF)选股模型持仓收益率')
    # plt.plot(x, tr_position, 'b:', label='换手率因子(TR)选股模型持仓收益率')
    # plt.plot(x, cnn_position, label='CNN选股模型持仓收益率')
    plt.plot(x, lstm_position, label='LSTM选股模型持仓收益率')
    plt.plot(x, bilstm_position, label='BILSTM选股模型持仓收益率')
    plt.plot(x, lstm2_position, label='LSTM2选股模型持仓收益率')
    # plt.plot(x, rnn_relu_position, label='RNN_relu选股模型持仓收益率')
    # plt.plot(x, resnet18_position, label='ResNet18选股模型持仓收益率')
    # plt.plot(x, gru_position, label='GRU选股模型持仓收益率')
    plt.plot(x, densenet_position, label='DenseNet选股模型持仓收益率')
    # plt.plot(x, ensemble_position, label='集成学习选股模型持仓收益率')
    plt.ylabel('收益率/%')
    x_ticks = list(x[::len(x) // 4])
    x_ticks.append(x[-1])
    x_labels = [bt1.trading_dates[i] for i in x_ticks]
    plt.xticks(x_ticks, x_labels)
    plt.legend()
    plt.subplot(212)

    # plt.plot(x, [0] * len(x), 'k:', label='基准市场收益率(沪深300)', )
    # plt.plot(x, bm_position.values - index_price.values, 'r:', label='价值因子(BM)选股模型持仓超额收益率')
    # plt.plot(x, cnn_position.values - index_price.values, label='CNN选股模型持仓超额收益率')
    # plt.plot(x, mf_position.values - index_price.values, 'g:', label='动量因子(MF)选股模型持仓超额收益率')
    # plt.plot(x, tr_position.values - index_price.values, 'b:', label='换手率因子(TR)选股模型持仓超额收益率')
    plt.plot(x, lstm_position.values - index_price.values, label='LSTM选股模型持仓超额收益率')
    plt.plot(x, bilstm_position.values - index_price.values, label='BILSTM选股模型持仓超额收益率')
    plt.plot(x, lstm2_position.values - index_price.values, label='LSTM2选股模型持仓超额收益率')
    # plt.plot(x, gru_position.values - index_price.values, label='GRU选股模型持仓超额收益率')
    # plt.plot(x, rnn_relu_position.values - index_price.values, label='RNN_relu选股模型持仓超额收益率')
    # plt.plot(x, resnet18_position.values - index_price.values, label='ResNet18选股模型持仓超额收益率')
    plt.plot(x, densenet_position.values - index_price.values, label='DenseNet选股模型持仓超额收益率')
    # plt.plot(x, ensemble_position.values - index_price.values, label='集成学习选股模型持仓超额收益率')
    plt.ylabel('超额收益率/%')
    plt.xticks(x_ticks, x_labels)
    plt.legend()
    plt.savefig('result_demo.jpg')
    end_time = time.time()
    span = end_time - mid_time
    print('计算持仓收益用时 {} 分 {} 秒'.format(int(span // 60), span % 60))
    span = end_time - start_time
    print('总计用时 {} 分 {:.2f} 秒'.format(int(span // 60), span % 60))

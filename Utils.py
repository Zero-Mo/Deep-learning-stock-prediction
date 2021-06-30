from Model import *

class Strategy:
    def __init__(self, data_days=10):

        self.data_days = data_days
        self.index_name = 'hs300'
        self.index_code = 'sh.000300'
        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        self.train_data_path = './data/train_data/'
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name))
        self.stocks_codes = self.stocks['code']
        # 指数日线数据
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.index_code))
        self.trading_dates = self.index['date']
        self.dataset = StockDataset(data_days=data_days)
        self.prediction = Prediction(data_days=data_days, batch_size=50)
        self.prediction.train_cnn(self.dataset, retrain=False, epochs=2)
        self.prediction.train_lstm(self.dataset, retrain=False, epochs=2)
        self.prediction.train_bilstm(self.dataset, retrain=False, epochs=2)
        self.prediction.train_lstm2(self.dataset, retrain=False, epochs=2)
        self.prediction.train_gru(self.dataset, retrain=False, epochs=2)
        self.prediction.train_rnn_relu(self.dataset, retrain=False, epochs=2)
        self.prediction.train_resnet18(self.dataset, retrain=False, epochs=2)
        self.prediction.train_densenet(self.dataset, retrain=False, epochs=2)

    def choose_by_bm(self, today: tuple, number: int):
        if today[1] < self.data_days:
            return pd.Series(None)
        stocks_data = pd.DataFrame(self.stocks_codes)
        stocks_data['aver_BM'] = 0
        stocks_data = stocks_data.set_index('code')
        days = range(today[1] - self.data_days, today[1])
        for stock_code in self.stocks_codes:
            sum_BM = 0
            valid_days = self.data_days
            stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            for day in days:
                if self.trading_dates[day] in stock_data.index:
                    pb = stock_data.loc[self.trading_dates[day], 'pbMRQ']
                    if pb != 0:
                        sum_BM += 1.0 / pb
                    else:
                        sum_BM += 0
                else:
                    valid_days -= 1
            if valid_days != 0:
                aver_BM = sum_BM / valid_days
            else:
                aver_BM = 0
            if aver_BM > 0:
                stocks_data.loc[stock_code, 'aver_BM'] = aver_BM
        stocks_data.sort_values(by='aver_BM', ascending=False, inplace=True)
        if len(stocks_data.index) > number:
            return stocks_data.index[0:number]
        else:
            return stocks_data.index[:]

    def choose_by_mf(self, today: tuple, number: int):
        if today[1] < self.data_days:
            return pd.Series(None)
        stocks_data = pd.DataFrame(self.stocks_codes)
        stocks_data['aver_MF'] = 0
        stocks_data = stocks_data.set_index('code')
        days = range(today[1] - self.data_days, today[1])
        for stock_code in self.stocks_codes:
            sum_MF = 0
            valid_days = self.data_days
            stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            for day in days:
                if self.trading_dates[day] in stock_data.index:
                    pc = stock_data.loc[self.trading_dates[day], 'preclose']
                    close = stock_data.loc[self.trading_dates[day], 'close']
                    sum_MF += close / pc
                else:
                    valid_days -= 1
            if valid_days != 0:
                aver_MF = sum_MF / valid_days
            else:
                aver_MF = 0
            if aver_MF > 0:
                stocks_data.loc[stock_code, 'aver_MF'] = aver_MF
        stocks_data.sort_values(by='aver_MF', ascending=False, inplace=True)
        if len(stocks_data.index) > number:
            return stocks_data.index[0:number]
        else:
            return stocks_data.index[:]

    def choose_by_tr(self, today: tuple, number: int):
        if today[1] < self.data_days:
            return pd.Series(None)
        stocks_data = pd.DataFrame(self.stocks_codes)
        stocks_data['aver_TR'] = 0
        stocks_data = stocks_data.set_index('code')
        days = range(today[1] - self.data_days, today[1])
        for stock_code in self.stocks_codes:
            sum_TR = 0
            valid_days = self.data_days
            stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            for day in days:
                if self.trading_dates[day] in stock_data.index:
                    tr = stock_data.loc[self.trading_dates[day], 'turn']
                    sum_TR += tr
                else:
                    valid_days -= 1
            if valid_days > 2:
                aver_TR = sum_TR / valid_days
                tr1 = stock_data.loc[self.trading_dates[days[-1]], 'turn']
                tr2 = stock_data.loc[self.trading_dates[days[-1] - 1], 'turn']
                ratio = (tr1 + tr2) / (aver_TR * 2)
            else:
                ratio = 0
            if ratio > 0:
                stocks_data.loc[stock_code, 'ratio'] = ratio

        stocks_data.sort_values(by='ratio', ascending=False, inplace=True)
        if len(stocks_data.index) > number:
            return stocks_data.index[0:number]
        else:
            return stocks_data.index[:]

    def __nn_choose(self, model_type: str, today: tuple, number: int):
        if today[1] < self.data_days:
            return pd.Series(None)
        stocks_data = pd.DataFrame(self.stocks_codes)
        stocks_data['change'] = 0
        stocks_data = stocks_data.set_index('code')
        avail_num = 0
        for stock_code in self.stocks_codes:
            change = getattr(self.prediction, 'predict_' + model_type)(stock_code, today)
            if type(change) != int:
                change = change[0, 0].item()
            if change > 0:
                stocks_data.loc[stock_code, 'change'] = change
                avail_num += 1
        stocks_data.sort_values(by='change', ascending=False, inplace=True)
        if avail_num > number:
            return stocks_data.index[0:number]
        else:
            return stocks_data.index[:avail_num]

    def choose_by_cnn(self, today: tuple, number: int):
        return self.__nn_choose('cnn', today, number)

    def choose_by_lstm(self, today: tuple, number: int):
        return self.__nn_choose('lstm', today, number)

    def choose_by_bilstm(self, today: tuple, number: int):
        return self.__nn_choose('bilstm', today, number)

    def choose_by_lstm2(self, today: tuple, number: int):
        return self.__nn_choose('lstm2', today, number)

    def choose_by_gru(self, today: tuple, number: int):
        return self.__nn_choose('gru', today, number)

    def choose_by_rnn_relu(self, today: tuple, number: int):
        return self.__nn_choose('rnn_relu', today, number)

    def choose_by_resnet18(self, today: tuple, number: int):
        return self.__nn_choose('resnet18', today, number)

    def choose_by_densenet(self, today: tuple, number: int):
        return self.__nn_choose('densenet', today, number)

    def choose_by_ensemble(self, today: tuple):
        chosen_num = pd.DataFrame()
        chosen_num['code'] = self.stocks_codes
        chosen_num['num'] = 0
        chosen_num.set_index('code', inplace=True)
        number = 300
        for chosen in (self.__nn_choose('cnn', today, number),
                       self.__nn_choose('lstm', today, number),
                       self.__nn_choose('bilstm', today, number),
                       self.__nn_choose('lstm2', today, number),
                       self.__nn_choose('rnn_relu', today, number),
                       self.__nn_choose('resnet18', today, number),
                       self.__nn_choose('densenet', today, number)):
            for stock_code in self.stocks_codes:
                if stock_code in chosen:
                    chosen_num.loc[stock_code, 'num'] += 1
        return chosen_num[chosen_num['num'] >= 3].index

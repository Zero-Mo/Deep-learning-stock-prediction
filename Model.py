
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def return_rate_transform(return_rate):
    if return_rate < -0.093:
        return -1.0
    elif return_rate < -0.053:
        return -0.8
    elif return_rate < -0.030:
        return -0.6
    elif return_rate < -0.014:
        return -0.4
    elif return_rate < 0.000:
        return -0.2
    elif return_rate < 0.016:
        return 0.2
    elif return_rate < 0.034:
        return 0.4
    elif return_rate < 0.058:
        return 0.6
    elif return_rate < 0.100:
        return 0.8
    elif return_rate >= 0.100:
        return 1.0


class StockDataset(Dataset):
    def __init__(self, data_days=10, remake_data=False):
        super(StockDataset, self).__init__()

        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        self.train_data_path = './data/train_data/'
        self.data_days = data_days
        self.index_name = 'hs300'
        self.index_code = 'sh.000300'
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name))
        self.stocks_codes = self.stocks['code']
        self.input_columns = ('open', 'high', 'low', 'close', 'preclose',
                              'turn', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ',)

        if not os.path.exists('{}{}.pkl'.format(self.base_data_path, self.index_name)):
            remake_data = True
        if remake_data:
            data = []
            for stock_code in tqdm(self.stocks_codes):

                stock_data = pd.read_csv('{}{}.csv'.format(self.train_data_path, stock_code))
                stock_data = pd.DataFrame(stock_data, columns=self.input_columns)

                batches = len(stock_data.index) - 2 * self.data_days
                if batches <= 0:
                    continue
                for i in range(batches):
                    if 0 in stock_data[i:i + self.data_days].values:
                        continue
                    # data_days后收盘价
                    next_price = stock_data.loc[2 * data_days + i, 'close']
                    # 当前日期收盘价
                    this_price = stock_data.loc[data_days + i, 'close']
                    close_change = this_price / stock_data.loc[data_days + i - 1, 'close'] - 1
                    predict_change = (next_price / this_price - 1)
                    # 当前日期前一天到前data_days天 共data_days天数据
                    data.append({'data': stock_data[i:i + self.data_days].values,
                                 'label': [predict_change, close_change]})
            self.data = pd.DataFrame(data)
            self.data.to_pickle('{}{}.pkl'.format(self.base_data_path, self.index_name))
        else:
            self.data = pd.read_pickle('{}{}.pkl'.format(self.base_data_path, self.index_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data.loc[idx, 'data'], dtype=torch.float32, device=device)
        label = torch.tensor(self.data.loc[idx, 'label'], dtype=torch.float32, device=device)
        return data, label


class CNNModel(nn.Module):
    def __init__(self, input_size, data_days=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3) 
        self.fc1 = nn.Linear((data_days - 4) * (input_size - 4) * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, n_layers):
        super(RNNModel, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, n_layers, batch_first=True)
        elif rnn_type == 'BILSTM':
            # self.rnn = getattr(nn, 'LSTM')(input_size, hidden_size, n_layers, batch_first=True)
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        elif rnn_type =='LSTM2':
            self.rnn=nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        else:
            try:
                non_linearity = { 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""非可选RNN类型,可选参数:['LSTM', 'GRU',  'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, nonlinearity=non_linearity, batch_first=True)
        if rnn_type=='LSTM2':
            cell_l0=nn.LSTMCell(input_size=120,hidden_size=60)
            cell_l1=nn.LSTMCell(input_size=60,hidden_size=30)
            h_l0=torch.zeros(3,60)
            c_l0=torch.zeros(3,60)
            h_l1=torch.zeros(3,30)
            c_l1=torch.zeros(3,30)
            xs=[torch.randn(3,120)for _ in range(10)]
            for xt in xs:
                h_l0,c_l0=cell_l0(xt,(h_l0,c_l0))
                h_l1,c_l1=cell_l1(h_l0,(h_l1,c_l1))


        self.fc1 = nn.Linear(hidden_size, 120)
        if rnn_type == 'BILSTM':
            self.fc1 = nn.Linear(hidden_size * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(self, x):
        x, _ = self.rnn(x)
        x = F.relu(self.fc1(x[:, -1, :]))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """实现将ResNet迁移应用于股票预测"""
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(4 * 512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            DenseLayer(in_channels + growth_rate * i,
                                       growth_rate, bn_size))


class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNetBC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 bn_size=4, theta=0.5, num_classes=2):
        super(DenseNetBC, self).__init__()

        # 初始的卷积为filter:2倍的growth_rate
        num_init_feature = 2 * growth_rate

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_feature,
                                kernel_size=3, stride=1,
                                padding=1, bias=False)),
        ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1),
                                     DenseBlock(num_layers, num_feature,
                                                bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1),
                                         Transition(num_feature,
                                                    int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        self.linear = nn.Linear(num_feature, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 增加一个1的维度
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.linear(out)
        return out


def dense_net_BC_100():
    return DenseNetBC(growth_rate=12, block_config=(16, 16, 16))


class Prediction:
    def __init__(self, data_days=10, batch_size=50):

        self.data_days = data_days
        self.index_name = 'hs300'
        self.index_code = 'sh.000300'
        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name))
        self.stocks_codes = self.stocks['code']
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.index_code))
        self.trading_dates = self.index['date']
        self.batch_size = batch_size
        self.input_columns = ('open', 'high', 'low', 'close', 'preclose',
                              'turn', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ',)
        input_size = len(self.input_columns)

        self.cnn = CNNModel(data_days, input_size).to(device)

        hidden_size = 20
        n_layers = 2

        self.lstm2= RNNModel('LSTM2', input_size, hidden_size, n_layers).to(device)
        self.bilstm = RNNModel('BILSTM', input_size, hidden_size, n_layers).to(device)
        self.lstm = RNNModel('LSTM', input_size, hidden_size, n_layers).to(device)
        self.gru = RNNModel('GRU', input_size, hidden_size, n_layers).to(device)
        self.rnn_relu = RNNModel('RNN_RELU', input_size, hidden_size, n_layers).to(device)
        self.resnet18 = ResNet18().to(device)
        self.densenet = dense_net_BC_100().to(device)

        self.criterion = nn.MSELoss()
        self.cnn_optimizer = torch.optim.AdamW(self.cnn.parameters())
        self.lstm_optimizer = torch.optim.AdamW(self.lstm.parameters())
        self.bilstm_optimizer = torch.optim.AdamW(self.bilstm.parameters())
        self.lstm2_optimizer=torch.optim.AdamW(self.lstm2.parameters())
        self.gru_optimizer = torch.optim.AdamW(self.gru.parameters())
        self.rnn_relu_optimizer = torch.optim.AdamW(self.rnn_relu.parameters())
        self.rn18_optimizer = torch.optim.AdamW(self.resnet18.parameters())
        self.densenet_optimizer = torch.optim.AdamW(self.densenet.parameters())

    def __train(self, model_name, model, optim, train_dataset, epochs=2):
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        model.train()
        print('*' * 20, '\n', model_name, '模型训练中')
        for epoch in range(epochs):
            for data, label in train_data:
                output = model.forward(data)
                loss = self.criterion(output, label)
                optim.zero_grad()
                loss.backward()
                optim.step()
                print('Train_loss:', loss.item(), end='\r')
                if loss.item() < 1e-3:
                    break
        torch.save(model, '{}{}.pt'.format(self.base_data_path, model_name))
        print('\n', model_name, '模型训练完成')

    def train_cnn(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}CNN.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            self.cnn = torch.load('{}CNN.pt'.format(self.base_data_path, self.index_name))
            return
        self.__train('CNN', self.cnn, self.cnn_optimizer, train_dataset, epochs)

    def train_lstm2(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}LSTM2.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            self.lstm2 = torch.load('{}LSTM2.pt'.format(self.base_data_path))
            return
        self.__train('LSTM2', self.lstm2, self.lstm2_optimizer, train_dataset, epochs)

    def train_bilstm(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}BILSTM.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            self.bilstm = torch.load('{}BILSTM.pt'.format(self.base_data_path))
            return
        self.__train('BILSTM', self.bilstm, self.bilstm_optimizer, train_dataset, epochs)

    def train_lstm(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}LSTM.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            self.lstm = torch.load('{}LSTM.pt'.format(self.base_data_path))
            return
        self.__train('LSTM', self.lstm, self.lstm_optimizer, train_dataset, epochs)

    def train_gru(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}GRU.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            self.rnn_relu = torch.load('{}GRU.pt'.format(self.base_data_path))
            return
        self.__train('GRU', self.gru, self.gru_optimizer, train_dataset, epochs)

    def train_rnn_relu(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}RNN_relu.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            self.rnn_relu = torch.load('{}RNN_relu.pt'.format(self.base_data_path))
            return
        self.__train('RNN_relu', self.rnn_relu, self.rnn_relu_optimizer, train_dataset, epochs)

    def train_resnet18(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}resnet18.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            self.resnet18 = torch.load('{}resnet18.pt'.format(self.base_data_path))
            return
        self.__train('resnet18', self.resnet18, self.rn18_optimizer, train_dataset, epochs)

    def train_densenet(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}densenet.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            self.densenet = torch.load('{}densenet.pt'.format(self.base_data_path))
            return
        self.__train('densenet', self.densenet, self.densenet_optimizer, train_dataset, epochs)

    def __predict_data(self, stock_code: str, today: tuple, abs_date=False):
        stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code))
        date_index = today[1] if abs_date else len(stock_data) - len(self.trading_dates) + today[1]
        if date_index < self.data_days:
            return 0
        stock_data = pd.DataFrame(stock_data, columns=self.input_columns)
        stock_data = stock_data.replace(0, None)
        stock_data = stock_data[date_index - self.data_days:date_index]
        stock_data = np.reshape(stock_data.values, (1, self.data_days, len(self.input_columns)))
        stock_data = torch.tensor(stock_data, dtype=torch.float32, device=device)
        return stock_data

    def __predict(self, model, stock_code: str, today: tuple):
        model.eval()
        stock_data = self.__predict_data(stock_code, today)
        if type(stock_data) == int:
            return 0
        with torch.no_grad():
            output = model.forward(stock_data)
            return output

    def predict_cnn(self, stock_code: str, today: tuple):
        return self.__predict(self.cnn, stock_code, today)

    def predict_lstm2(self, stock_code: str, today: tuple):
        return self.__predict(self.lstm2, stock_code, today)

    def predict_lstm(self, stock_code: str, today: tuple):
        return self.__predict(self.lstm, stock_code, today)

    def predict_bilstm(self, stock_code: str, today: tuple):
        return self.__predict(self.bilstm, stock_code, today)

    def predict_gru(self, stock_code: str, today: tuple):
        return self.__predict(self.gru, stock_code, today)

    def predict_rnn_tanh(self, stock_code: str, today: tuple):
        return self.__predict(self.rnn_tanh, stock_code, today)

    def predict_rnn_relu(self, stock_code: str, today: tuple):
        return self.__predict(self.rnn_relu, stock_code, today)

    def predict_resnet18(self, stock_code: str, today: tuple):
        return self.__predict(self.resnet18, stock_code, today)

    def predict_densenet(self, stock_code: str, today: tuple):
        return self.__predict(self.densenet, stock_code, today)


if __name__ == '__main__':
    dataset = StockDataset(data_days=10, remake_data=False)
    print('训练集大小:', len(dataset))

    prediction = Prediction(data_days=10, batch_size=200)

    prediction.train_cnn(dataset, retrain=False, epochs=2)
    prediction.train_rnn_relu(dataset, retrain=False, epochs=2)
    prediction.train_resnet18(dataset, retrain=False, epochs=2)
    prediction.train_densenet(dataset, retrain=False, epochs=2)
    prediction.train_gru(dataset, retrain=False, epochs=2)
    prediction.train_lstm(dataset, retrain=False, epochs=2)
    prediction.train_bilstm(dataset, retrain=False, epochs=2)
    prediction.train_lstm2(dataset, retrain=False, epochs=2)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=[30, 15], dpi=160)
    for code in dataset.stocks_codes[:5]:
        print('正在绘制'+code+'预测图像')
        plt.clf()
        df = pd.read_csv('./data/stocks/' + code + '.csv')
        trading_dates = df['date']
        x_r = range(0, len(trading_dates))
        x_ticks = list(x_r[::100])
        x_ticks.append(x_r[-1])
        x_labels = [trading_dates[i] for i in x_ticks]
        true_close = df['close'].values

        def close_p(x):
            if type(x) == int:
                return x
            x = x[0, 1].item()
            return x if 0.2 > x > -0.2 else 0.0

        print('计算CNN')
        cnn_close = [true_close[j]*(1+close_p(prediction.predict_cnn(code, (0, j))))
                     for j in range(len(trading_dates))]
        print('计算RNN_relu')
        rnn_relu_close = [true_close[j] * (1 + close_p(prediction.predict_rnn_relu(code, (0, j))))
                          for j in range(len(trading_dates))]
        print('计算ResNet18')
        rn18_close = [true_close[j] * (1 + close_p(prediction.predict_resnet18(code, (0, j))))
                      for j in range(len(trading_dates))]
        print('计算DenseNet')
        densenet_close = [true_close[j]*(1+close_p(prediction.predict_densenet(code, (0, j))))
                          for j in range(len(trading_dates))]
        print('计算GRU')
        gru_close = [true_close[j]*(1+close_p(prediction.predict_gru(code, (0, j))))
                     for j in range(len(trading_dates))]
        print('计算LSTM')
        lstm_close = [true_close[j]*(1+close_p(prediction.predict_lstm(code, (0, j))))
                      for j in range(len(trading_dates))]

        print('计算BILSTM')
        bilstm_close = [true_close[j] * (1 + close_p(prediction.predict_bilstm(code, (0, j))))
                        for j in range(len(trading_dates))]
        print('计算LSTM2')
        lstm2_close = [true_close[j] * (1 + close_p(prediction.predict_lstm2(code, (0, j))))
                       for j in range(len(trading_dates))]


        def sp(i, predict_close, label_name):
            plt.subplot(3, 3, i)
            plt.plot(x_r, true_close, label='真实值')
            plt.plot(x_r, predict_close, label=label_name)
            plt.ylabel('收盘价')
            plt.xticks(x_ticks, x_labels)
            plt.legend()

        sp(1, cnn_close, 'CNN模型预测值')
        sp(2, rnn_relu_close, 'RNN_relu模型预测值')
        sp(3, rn18_close, 'ResNet18模型预测值')
        sp(4, densenet_close, 'DenseNet模型预测值')
        sp(5, gru_close, 'GRU模型预测值')
        sp(6, lstm_close, 'LSTM模型预测值')
        sp(7, bilstm_close, 'BILSTM模型预测值')
        sp(8, lstm2_close, 'LSTM2模型预测值')


        plt.savefig(code+'_predict.jpg')

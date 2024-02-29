import argparse
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
import os

class EOD_Preprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def _read_EOD_data(self):
        self.data_EOD = []
        for index, ticker in enumerate(self.tickers):
            single_EOD = np.genfromtxt(
                os.path.join(self.data_path, self.market_name + '_' + ticker +
                             '_30Y.csv'), dtype=str, delimiter=',',
                skip_header=True
            )
            self.data_EOD.append(single_EOD)
        print('#stocks\' EOD data readin:', len(self.data_EOD))
        assert len(self.tickers) == len(self.data_EOD), 'length of tickers ' \
                                                        'and stocks not match'


    def _transfer_EOD_str(self, selected_EOD_str, tra_date_index):
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=np.float32)
        for row, daily_EOD in enumerate(selected_EOD_str):
            date_str = daily_EOD[0].replace('-05:00', '')
            date_str = date_str.replace('-04:00', '')
            if date_str in tra_date_index:
                selected_EOD[row, 0] = tra_date_index[date_str]
            for col in range(1, selected_EOD_str.shape[1]):
                selected_EOD[row, col] = float(daily_EOD[col])
        return selected_EOD

    '''
        Transform the original EOD data collected from Google Finance to a
        friendly format to fit machine learning model via the following steps:
            Calculate moving average (5-days, 10-days, 20-days, 30-days),
            ignoring suspension days (market open, only suspend this stock)
            Normalize features by (feature - min) / (max - min)
    '''
    def generate_feature(self, selected_tickers_fname, begin_date, end_date, 
                         opath, pad_begin=60):
        trading_dates = np.genfromtxt(
            os.path.join(self.data_path, '..',
                         self.market_name + '_aver_line_dates.csv'),
            dtype=str, delimiter=',', skip_header=False
        )
        factors = np.genfromtxt(os.path.join(self.data_path, '..', 'F-F_5_Factors.csv'),
            dtype=np.float32, delimiter=',', skip_header=False)
        factors[:, 0] = 1.0

        print('#trading dates:', len(trading_dates))
        print('begin date:', begin_date)
        print('end dates:', end_date)
        # transform the trading dates into a dictionary with index
        index_tra_dates = {}
        tra_dates_index = {}
        for index, date in enumerate(trading_dates):
            tra_dates_index[date] = index
            index_tra_dates[index] = date
        self.tickers = np.genfromtxt(
            os.path.join(self.data_path, '..', selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        #self.tickers = ['AABA', 'AAON', 'AAPL', 'AAWW', 'AAXJ', 'AAXN', 'ABAX', 'ABCB', 'ABCO', 'ABMD']
        #self.tickers = ['CGO', 'CHI', 'CHW', 'CHY', 'CSQ', 'IEP', 'UCBI']
        #self.tickers = ['RGLD']
        print('#tickers selected:', len(self.tickers))
        self._read_EOD_data()
        for stock_index, single_EOD in enumerate(self.data_EOD):
            # select data within the begin_date and the end_date
            begin_date_row = -1
            for date_index, daily_EOD in enumerate(single_EOD):
                date_str = daily_EOD[0].replace('-05:00', '')
                date_str = date_str.replace('-04:00', '')
                cur_date = datetime.strptime(date_str, self.date_format)
                if cur_date > begin_date:
                    begin_date_row = date_index
                    break
            end_date_row = single_EOD.shape[0]
            for date_index, daily_EOD in enumerate(reversed(single_EOD)):
                date_str = daily_EOD[0].replace('-05:00', '')
                date_str = date_str.replace('-04:00', '')
                cur_date = datetime.strptime(date_str, self.date_format)
                if cur_date <= end_date:
                    end_date_row = single_EOD.shape[0] - date_index
                    break
            
            selected_EOD_str = single_EOD[begin_date_row:end_date_row]
            selected_EOD = self._transfer_EOD_str(selected_EOD_str, tra_dates_index)

            dates = selected_EOD[:, 0]
            dates = dates.astype(int)
            l_date = dates[0]
            r_date = dates[-1]
            
            ## if l_date != 0:
            ##     print(self.tickers[stock_index] + ":" + str(l_date))
            ## if r_date != 1304:
            ##     print(self.tickers[stock_index] + ":" + str(r_date))

            log_returns = np.zeros(r_date - l_date, dtype=np.float32)
            row_r = 0
            for row in range(len(dates) - 1):
                log_difference = np.log(selected_EOD[row + 1, 4]) - \
                                 np.log(selected_EOD[row, 4])
                period = dates[row + 1] - dates[row]
                log_returns[row_r : row_r + period] = log_difference / period
                row_r = row_r + period

            features = np.zeros([len(trading_dates) - pad_begin, 12], dtype=np.float32)
            features[:, 0] = 0.0
            l_bound = max(l_date - pad_begin + 30, 0)
            r_bound = r_date - pad_begin + 1
            adjust = pad_begin - l_date
            for row in range(l_bound, r_bound):
                features[row, 0] = 1.0
                features[row, 1] = log_returns[row + adjust - 1]
                if row == l_bound:
                    features[row, 2] = np.mean(log_returns[row + adjust - 5 : row + adjust])
                    features[row, 3] = np.mean(log_returns[row + adjust - 10 : row + adjust])
                    features[row, 4] = np.mean(log_returns[row + adjust - 20 : row + adjust])
                    features[row, 5] = np.mean(log_returns[row + adjust - 30 : row + adjust])
                else:
                    features[row, 2] = features[row - 1, 2] + \
                        (log_returns[row + adjust - 1] - log_returns[row + adjust - 6]) / 5
                    features[row, 3] = features[row - 1, 3] + \
                        (log_returns[row + adjust - 1] - log_returns[row + adjust - 11]) / 10
                    features[row, 4] = features[row - 1, 4] + \
                        (log_returns[row + adjust - 1] - log_returns[row + adjust - 21]) / 20
                    features[row, 5] = features[row - 1, 5] + \
                        (log_returns[row + adjust - 1] - log_returns[row + adjust - 31]) / 30
                features[row, -1] = factors[row + pad_begin, -1]
                if row < l_date:
                    y = log_returns[0 : row + adjust] - \
                        factors[l_date + 1 : row + pad_begin + 1, 6]
                    X = factors[l_date + 1 : row + pad_begin + 1, 1 : 6]
                else:
                    y = log_returns[row - l_date : row + adjust] - \
                        factors[row + 1 : row + pad_begin + 1, 6]
                    X = factors[row + 1 : row + pad_begin + 1, 1 : 6]
                reg = LinearRegression().fit(X, y)
                features[row, 6 : -1] = reg.coef_

            # write out
            np.savetxt(os.path.join(opath, self.market_name + '_' +
                                    self.tickers[stock_index] + '_' +
                                    '2_new.csv'), features,
                       fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    desc = "pre-process EOD data market by market, including listing all " \
           "trading days, all satisfied stocks (5 years & high price), " \
           "normalizing and compansating data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-path', help='path of EOD data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = 'data/google_finance'
    if args.market is None:
        args.market = 'NASDAQ'

    processor = EOD_Preprocessor(args.path, args.market)
    processor.generate_feature(
        processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv',
        datetime.strptime('2012-10-03 00:00:00', processor.date_format),
        datetime.strptime('2017-12-08 23:59:59', processor.date_format),
        os.path.join(processor.data_path, '..', '2013-01-01'), 
        pad_begin=60
    )
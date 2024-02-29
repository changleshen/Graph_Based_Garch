import argparse
from datetime import datetime
import numpy as np
import os

class EOD_Preprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def _read_EOD_data(self):
        self.data_EOD = []
        for index, ticker in enumerate(self.tickers):
            #if ticker not in ['AABA', 'AAON', 'AAPL', 'AAWW', 'AAXJ', 'AAXN', 'ABAX', 'ABCB', 'ABCO', 'ABMD']:
            #    break
            #if ticker not in ['CGO', 'CHI', 'CHW', 'CHY', 'CSQ', 'IEP', 'UCBI']:
            #    break
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
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
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
    def generate_feature(self, selected_tickers_fname, begin_date, opath,
                         return_days=1, pad_begin=59):
        trading_dates = np.genfromtxt(
            os.path.join(self.data_path, '..',
                         self.market_name + '_aver_line_dates.csv'),
            dtype=str, delimiter=',', skip_header=False
        )
        factors = np.genfromtxt(os.path.join(self.data_path, '..', 'F-F_5_Factors.csv'),
            dtype=str, delimiter=',', skip_header=False)
        print('#trading dates:', len(trading_dates))
        # begin_date = datetime.strptime(trading_dates[29], self.date_format)
        print('begin date:', begin_date)
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
        print('#tickers selected:', len(self.tickers))
        self._read_EOD_data()
        for stock_index, single_EOD in enumerate(self.data_EOD):
            # select data within the begin_date
            begin_date_row = -1
            for date_index, daily_EOD in enumerate(single_EOD):
                date_str = daily_EOD[0].replace('-05:00', '')
                date_str = date_str.replace('-04:00', '')
                cur_date = datetime.strptime(date_str, self.date_format)
                if cur_date > begin_date:
                    begin_date_row = date_index
                    break
            selected_EOD_str = single_EOD[begin_date_row:]
            selected_EOD = self._transfer_EOD_str(selected_EOD_str, tra_dates_index)

            # calculate moving average features
            f_begin_date_row = -1
            dates = selected_EOD[return_days:, 0]
            for row in range(pad_begin, 0, -1):
                if dates[row] < dates[0] + pad_begin:
                    f_begin_date_row = row + 1
                    break

            log_return = np.log(selected_EOD[return_days : , 4]) - np.log(selected_EOD[0 : -return_days, 4])
            mov_aver_features = np.zeros([len(log_return), 4], dtype=float) 
            # 4 columns refers to 5-, 10-, 20-, 30-days average
            for row in range(f_begin_date_row, len(log_return)):
                if dates[row] - dates[row - 1] == 3:
                    print(self.tickers[stock_index] + '!!!')
                if dates[row] - dates[row - 1] == 4:
                    print(self.tickers[stock_index] + '!!!!')
                if dates[row] - dates[row - 1] == 5:
                    print(self.tickers[stock_index] + '!!!!!')
                # date_index = dates[row]
                # aver_5 = 0.0
                # aver_10 = 0.0
                # aver_20 = 0.0
                # aver_30 = 0.0
                # count_5 = 0
                # count_10 = 0
                # count_20 = 0
                # count_30 = 0
                # for offset in range(30):
                #     date_gap = date_index - dates[row - offset]
                #     if date_gap < 5:
                #         count_5 += 1
                #         aver_5 += log_return[row - offset]
                #     if date_gap < 10:
                #         count_10 += 1
                #         aver_10 += log_return[row - offset]
                #     if date_gap < 20:
                #         count_20 += 1
                #         aver_20 += log_return[row - offset]
                #     if date_gap < 30:
                #         count_30 += 1
                #         aver_30 += log_return[row - offset]
                # mov_aver_features[row, 0] = aver_5 / count_5
                # mov_aver_features[row, 1] = aver_10 / count_10
                # mov_aver_features[row, 2] = aver_20 / count_20
                # mov_aver_features[row, 3] = aver_30 / count_30

            #factor_loadings = np.zeros([len(log_return), 5], dtype=float) 
            #for r_row in range(f_begin_date_row, len(log_return)):
            #    for l_row in range(r_row - pad_begin, r_row - 4):
            #        if dates[l_row] + pad_begin >= dates[r_row]:
            #            break
            #    y = log_return[l_row : r_row + 1]
            #    time_period = dates[l_row : r_row + 1]
            #    X = factors[time_period]


            '''
                generate feature and ground truth in the following format:
                date_index, 5-day, 10-day, 20-day, 30-day, close price
                two ways to pad missing dates:
                for dates without record, pad a row [date_index, -1234 * 5]
            '''
            ahead_days = return_days + pad_begin
            features = np.ones([len(trading_dates) - ahead_days, 6],
                               dtype=float) * -1234
            # data missed at the beginning
            for row in range(len(trading_dates) - ahead_days):
                features[row, 0] = row
            #if int(dates[f_begin_date_row]) > ahead_days:
            #    print(self.tickers[stock_index] + '!!')
            for row in range(f_begin_date_row, len(log_return)):
                cur_index = int(dates[row])
                features[cur_index - ahead_days, 1 : 5] = mov_aver_features[row]
                if cur_index - int(dates[row - return_days]) == return_days:
                    features[cur_index - ahead_days, 5] = log_return[row]
            #print(self.tickers[stock_index])
            # write out
            ##np.savetxt(os.path.join(opath, self.market_name + '_' +
            ##                        self.tickers[stock_index] + '_' +
            ##                        str(return_days) + '_new.csv'), features,
            ##           fmt='%.6f', delimiter=',')


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
        os.path.join(processor.data_path, '..', '2013-01-01'), 
        return_days=1, pad_begin=59
    )
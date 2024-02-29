import argparse
import json
import numpy as np
import scipy.sparse as sp
import os

class SectorPreprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def generate_sector_relation(self, industry_ticker_file,
                                 selected_tickers_fname):
        selected_tickers = np.genfromtxt(
            os.path.join(self.data_path, selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print('#tickers selected:', len(selected_tickers))
        ticker_index = {}
        for index, ticker in enumerate(selected_tickers):
            ticker_index[ticker] = index
        with open(industry_ticker_file, 'r') as fin:
            industry_tickers = json.load(fin)
        print('#industries:', len(industry_tickers))
        valid_industry_count = 0
        valid_industry_index = {}
        edge_count = 0
        for industry in industry_tickers.keys():
            cur_industry_count = len(industry_tickers[industry])
            if cur_industry_count > 1 and industry != 'n/a':
                valid_industry_index[industry] = valid_industry_count
                valid_industry_count += 1
                edge_count += cur_industry_count * (cur_industry_count - 1)
        print('#valid industries:', valid_industry_count)
        print('#edges:', edge_count)
        ticker_edge_index = np.zeros([len(selected_tickers), len(selected_tickers)], dtype=bool)
        print(ticker_edge_index.shape)
        for industry in valid_industry_index.keys():
            cur_ind_tickers = industry_tickers[industry]
            if len(cur_ind_tickers) <= 1:
                print('shit industry:', industry)
                continue
            for i in range(len(cur_ind_tickers)):
                left_tic_ind = ticker_index[cur_ind_tickers[i]]
                for j in range(i + 1, len(cur_ind_tickers)):
                    right_tic_ind = ticker_index[cur_ind_tickers[j]]
                    ticker_edge_index[left_tic_ind][right_tic_ind] = True
                    ticker_edge_index[right_tic_ind][left_tic_ind] = True

        coo_ticker_edge_index = sp.coo_matrix(ticker_edge_index)
        ticker_edge_index = np.array([coo_ticker_edge_index.row, coo_ticker_edge_index.col], dtype=int)
        print(ticker_edge_index.shape)
        np.save(os.path.join('data/relation/sector_industry/', 
                             processor.market_name + '_industry_relation'), 
                ticker_edge_index)


if __name__ == '__main__':
    desc = "pre-process sector data market by market, including listing all " \
           "trading days, all satisfied stocks (5 years & high price), " \
           "normalizing and compansating data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-path', help='path of EOD data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = 'data'
    if args.market is None:
        args.market = 'NASDAQ'

    processor = SectorPreprocessor(args.path, args.market)

    processor.generate_sector_relation(
        os.path.join('data/relation/sector_industry/',
                     processor.market_name + '_industry_ticker.json'),
        processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    )
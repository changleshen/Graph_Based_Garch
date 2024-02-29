import numpy as np
import os


def load_EOD_data(data_path, market_name, tickers):
    features = []
    excess_return = []
    factors = []
    masks = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_2_new.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            features = np.zeros([len(tickers), single_EOD.shape[0] - 1, 
                                 single_EOD.shape[1] - 2], dtype=np.float32)
            mul_ind = np.ones(single_EOD.shape[0], dtype=int)
            excess_return = np.zeros([len(tickers), single_EOD.shape[0] - 1], dtype=np.float32)
        features[index] = single_EOD[:-1, 1:-1]
        excess_return[index] = single_EOD[1:, 1]
        #excess_return[index] = single_EOD[1:, 1] - single_EOD[1:, -1]
        mul_ind = np.multiply(mul_ind, single_EOD[:, 0])
    mul_ind_n = mul_ind[:-1] * mul_ind[1:]
    masks = (mul_ind_n > 0.5)
    factors_in = np.genfromtxt(os.path.join(data_path, '..', 'F-F_5_Factors.csv'),
            dtype=np.float32, delimiter=',', skip_header=False)
    factors = factors_in[- features.shape[1]:, 1 : ]
    excess_return = excess_return - np.expand_dims(factors_in[- features.shape[1]:, -1], axis=0)

    return features, excess_return, masks, factors


if __name__ == '__main__':
    tickers = np.genfromtxt('data/NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv',
                                     dtype=str, delimiter='\t', skip_header=False)
    features, excess_return, masks, factors = load_EOD_data('data/2013-01-01', 'NASDAQ', tickers)

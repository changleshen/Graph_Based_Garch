import argparse
import json
import numpy as np
import scipy.sparse as sp
import os


class EdgePreprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.market_name = market_name

    def generate_sector_relation(self, industry_ticker_file, selected_tickers_file):
        selected_tickers = np.genfromtxt(selected_tickers_file,
                                         dtype=str, delimiter='\t', skip_header=False)
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
        return ticker_edge_index
    
    def generate_wiki_relation(self, connection_file, ticker_wiki_file, selected_path_file):
        # readin tickers
        tickers = np.genfromtxt(ticker_wiki_file, dtype=str, delimiter=',',
                                skip_header=False)
        print('#tickers selected:', tickers.shape)
        wikiid_ticind_dic = {}
        for ind, tw in enumerate(tickers):
            if not tw[-1] == 'unknown':
                wikiid_ticind_dic[tw[-1]] = ind
        print('#tickers aligned:', len(wikiid_ticind_dic))
        # readin selected paths/connections
        sel_paths = np.genfromtxt(selected_path_file, dtype=str, delimiter=' ',
                                  skip_header=False)
        print('#paths selected:', len(sel_paths))
        sel_paths = set(sel_paths[:, 0])
        # readin connections
        with open(connection_file, 'r') as fin:
            connections = json.load(fin)
        print('#connection items:', len(connections))
        # get occured paths
        occur_paths = set()
        for sou_item, conns in connections.items():
            for tar_item, paths in conns.items():
                for p in paths:
                    path_key = '_'.join(p)
                    if path_key in sel_paths:
                        occur_paths.add(path_key)
        # generate
        valid_path_index = {}
        for ind, path in enumerate(occur_paths):
            valid_path_index[path] = ind
        print('#valid paths:', len(valid_path_index))
        # for path, ind in valid_path_index.items():
        #     print(path, ind)
        wiki_relation_embedding = np.zeros([tickers.shape[0], tickers.shape[0]], dtype=bool)
        conn_count = 0
        for sou_item, conns in connections.items():
            for tar_item, paths in conns.items():
                for p in paths:
                    path_key = '_'.join(p)
                    if path_key in valid_path_index.keys():
                        wiki_relation_embedding[wikiid_ticind_dic[sou_item]][wikiid_ticind_dic[tar_item]] = True
                        wiki_relation_embedding[wikiid_ticind_dic[tar_item]][wikiid_ticind_dic[sou_item]] = True
                        conn_count += 1
        print('connections count:', conn_count, 'ratio:', conn_count / float(tickers.shape[0] * tickers.shape[0]))
        print(wiki_relation_embedding.shape)
        return wiki_relation_embedding
    
    def build_edge_index(self, industry_ticker_file, selected_tickers_file, 
                         connection_file, ticker_wiki_file, selected_path_file):
        ticker_edge_index = self.generate_sector_relation(industry_ticker_file, selected_tickers_file)
        wiki_relation_embedding = self.generate_wiki_relation(
            connection_file, ticker_wiki_file, selected_path_file)
        edge_index = np.logical_or(ticker_edge_index, wiki_relation_embedding)
        coo_edge_index = sp.coo_matrix(edge_index)
        edge_index = np.array([coo_edge_index.row, coo_edge_index.col], dtype=int)
        print(edge_index.shape)
        np.save(os.path.join(self.data_path, self.market_name + '_edge_index'), edge_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', help='path of data')
    parser.add_argument('-sec', help='path of sector-industry relation data')
    parser.add_argument('-wiki', help='path of wiki relation data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = 'data/relation'
    if args.market is None:
        args.market = 'NASDAQ'
    
    sec_path = 'data/relation/sector_industry/'
    wiki_path = 'data/relation/wikidata/'
    processor = EdgePreprocessor(args.path, args.market)
    processor.build_edge_index(
        os.path.join(sec_path, processor.market_name + '_industry_ticker.json'),
        os.path.join(processor.data_path, '..', 
                     processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'),
        os.path.join(wiki_path, processor.market_name + '_connections.json'),
        os.path.join(wiki_path, processor.market_name + '_wiki.csv'),
        os.path.join(wiki_path, 'selected_wiki_connections.csv')
    )
import json
import numpy as np
import scipy.sparse as sp
import os

def build_wiki_relation(market_name, connection_file, tic_wiki_file,
                        sel_path_file):
    # readin tickers
    tickers = np.genfromtxt(tic_wiki_file, dtype=str, delimiter=',',
                            skip_header=False)
    print('#tickers selected:', tickers.shape)
    wikiid_ticind_dic = {}
    for ind, tw in enumerate(tickers):
        if not tw[-1] == 'unknown':
            wikiid_ticind_dic[tw[-1]] = ind
    print('#tickers aligned:', len(wikiid_ticind_dic))

    # readin selected paths/connections
    sel_paths = np.genfromtxt(sel_path_file, dtype=str, delimiter=' ',
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
    for path, ind in valid_path_index.items():
        print(path, ind)

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
    coo_wiki_relation_embedding = sp.coo_matrix(wiki_relation_embedding)
    ticker_edge_index = np.array([coo_wiki_relation_embedding.row, coo_wiki_relation_embedding.col], dtype=int)
    print(ticker_edge_index.shape)
    np.save(os.path.join('data/relation/wikidata/', market_name + '_wiki_relation'), 
                ticker_edge_index)


# single thread version
if __name__ == '__main__':
    path = 'data/relation/wikidata/'
    build_wiki_relation('NASDAQ',
                        os.path.join(path, 'NASDAQ_connections.json'),
                        os.path.join(path, 'NASDAQ_wiki.csv'),
                        os.path.join(path, 'selected_wiki_connections.csv'))
    print('----------')
    build_wiki_relation('NYSE',
                        os.path.join(path, 'NYSE_connections.json'),
                        os.path.join(path, 'NYSE_wiki.csv'),
                        os.path.join(path, 'selected_wiki_connections.csv'))

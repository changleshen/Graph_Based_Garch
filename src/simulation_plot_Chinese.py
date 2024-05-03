import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
from scipy import stats
import scipy.sparse as sp
import statsmodels.api as sm
import math

file_name = 'data_simulation/S200T400/'

theta_0 = np.load(file_name + 'theta_0.npy')
theta_mean = np.load(file_name + 'theta_mean.npy')


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['times_simsun']
# plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

s = 0

fig, axs = plt.subplots(4, 2, figsize=(15, 8))
axs[0, 0].plot(0.01 * theta_0[s, :, 0], label='真实值', color='black')
axs[0, 0].plot(0.01 * theta_mean[s, :, 0], label='估计值', color='dodgerblue')
axs[0, 0].set_title(r'$\hat{\alpha}_{t}$', fontdict={'fontsize': 16})
axs[0, 0].legend(fontsize=12)
axs[0, 0].tick_params(labelsize=12)

axs[0, 1].plot(theta_0[s, :, 1], label='真实值', color='black')
axs[0, 1].plot(theta_mean[s, :, 1], label='估计值', color='dodgerblue')
axs[0, 1].set_title(r'$\hat{\beta}_{1t}$', fontdict={'fontsize': 16})
axs[0, 1].legend(fontsize=12)
axs[0, 1].tick_params(labelsize=12)

axs[1, 0].plot(theta_0[s, :, 2], label='真实值', color='black')
axs[1, 0].plot(theta_mean[s, :, 2], label='估计值', color='dodgerblue')
axs[1, 0].set_title(r'$\hat{\beta}_{2t}$', fontdict={'fontsize': 16})
axs[1, 0].legend(fontsize=12)
axs[1, 0].tick_params(labelsize=12)

axs[1, 1].plot(theta_0[s, :, 3], label='真实值', color='black')
axs[1, 1].plot(theta_mean[s, :, 3], label='估计值', color='dodgerblue')
axs[1, 1].set_title(r'$\hat{\beta}_{3t}$', fontdict={'fontsize': 16})
axs[1, 1].legend(fontsize=12)
axs[1, 1].tick_params(labelsize=12)

axs[2, 0].plot(theta_0[s, :, 4], label='真实值', color='black')
axs[2, 0].plot(theta_mean[s, :, 4], label='估计值', color='dodgerblue')
axs[2, 0].set_title(r'$\hat{a}_{t}$', fontdict={'fontsize': 16})
axs[2, 0].legend(fontsize=12)
axs[2, 0].tick_params(labelsize=12)

axs[2, 1].plot(theta_0[s, :, 5], label='真实值', color='black')
axs[2, 1].plot(theta_mean[s, :, 5], label='估计值', color='dodgerblue')
axs[2, 1].set_title(r'$\hat{b}_{t}$', fontdict={'fontsize': 16})
axs[2, 1].legend(fontsize=12)
axs[2, 1].tick_params(labelsize=12)

axs[3, 0].plot(0.01 * 0.01 * theta_0[s, :, 6], label='真实值', color='black')
axs[3, 0].plot(0.01 * 0.01 * theta_mean[s, :, 6], label='估计值', color='dodgerblue')
axs[3, 0].set_title(r'$\hat{c}_{t}$', fontdict={'fontsize': 16})
axs[3, 0].legend(fontsize=12)
axs[3, 0].tick_params(labelsize=12)

fig.delaxes(axs[3, 1])

plt.tight_layout()
plt.savefig('plot/simulation/S200T400/theta.pdf')
plt.close()


Stocks = 50
np.random.seed(42)
edge_index = sp.random(Stocks, Stocks, density=0.5 - 1 / np.sqrt(5), data_rvs=stats.binom(n=1, p=1).rvs)
edge_index = edge_index - np.transpose(edge_index)
coo_edge_index = sp.coo_matrix(edge_index)
edge_index = torch.tensor(np.array([coo_edge_index.row, coo_edge_index.col]), dtype=torch.long)
data = Data(edge_index=edge_index, num_nodes=Stocks)
graph = torch_geometric.utils.to_networkx(data)
graph = nx.Graph(graph)
# Plot the graph
pos = nx.spring_layout(graph, k=5/math.sqrt(Stocks))  # Layout algorithm for node positioning
plt.figure(figsize=(6, 6))
nx.draw(graph, pos, with_labels=False, node_color='crimson', node_size=100, 
        edge_color='lightskyblue', width=0.8, alpha=0.8)
plt.savefig('plot/simulation/graph.pdf', bbox_inches='tight')
plt.close()
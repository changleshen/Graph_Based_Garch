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
import pandas as pd
from arch import arch_model

from model import GNN, training, LossFunction, activation, InitialTheta


device = torch.device('cuda:1')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

file_name = 'checkpoints/real_data_50_16.pth'

return_data = pd.read_csv('data/return.csv', index_col=0)
factor_data = pd.read_csv('data/factors.csv', index_col=0)
market_portfolio_data = pd.read_csv('data/sp500.csv', index_col=0)
label_dict = {i: col for i, col in enumerate(return_data.columns)}
start_date = 2012
end_date = start_date + 1509
T_training = 1008
T_validation = 1259
y_data = np.transpose(return_data.values) - factor_data.values[:, 5]
Stocks = y_data.shape[0]

f = factor_data.values[start_date + 1 : end_date + 1, 0 : 5]
f_intercept = sm.add_constant(factor_data.values[:, 0 : 5])
f_intercept_t = np.transpose(f_intercept)
y_reg = np.transpose(y_data)

y_sum_sqaures = np.square(y_data)
for t in range(1, y_data.shape[1]):
    y_sum_sqaures[:, t] = y_sum_sqaures[:, t] + y_sum_sqaures[:, t - 1]
y = y_data[:, start_date + 1 : end_date + 1]
x = np.zeros([y.shape[0], end_date - start_date, 10])
x[:, :, 0] = np.abs(y_data)[:, start_date : end_date]
x[:, :, 1] = np.sqrt((y_sum_sqaures[:, start_date : end_date] - y_sum_sqaures[:, start_date - 5 : end_date - 5]) / 5)
x[:, :, 2] = np.sqrt((y_sum_sqaures[:, start_date : end_date] - y_sum_sqaures[:, start_date - 10 : end_date - 10]) / 10)
x[:, :, 3] = np.sqrt((y_sum_sqaures[:, start_date : end_date] - y_sum_sqaures[:, start_date - 20 : end_date - 20]) / 20)
x[:, :, 4] = np.sqrt((y_sum_sqaures[:, start_date : end_date] - y_sum_sqaures[:, start_date - 30 : end_date - 30]) / 30)

for t in range(start_date, end_date):
    X_reg = f_intercept[t - 119 : t + 1, :]
    X_reg_t = f_intercept_t[: , t - 119 : t + 1]
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_reg_t, X_reg)), X_reg_t), y_reg[t - 119 : t + 1, :])
    x[:, t - start_date, 5 :] = np.transpose(beta[1 : , :])

corr_res = stats.spearmanr(y[:, : T_training], axis=1)
corr_mat = corr_res.correlation
edge_index = (np.abs(corr_mat) > 0.5) - np.eye(Stocks)
print(np.mean(edge_index))
coo_edge_index = sp.coo_matrix(edge_index)
edge_index = torch.tensor(np.array([coo_edge_index.row, coo_edge_index.col]), device=device, dtype=torch.long)
# data = Data(edge_index=edge_index, num_nodes=Stocks)
# graph = torch_geometric.utils.to_networkx(data)
# graph = nx.Graph(graph)
# pos = nx.spring_layout(graph, k=100/math.sqrt(Stocks))  # Layout algorithm for node positioning
# plt.figure(figsize=(12, 16))
# nx.draw(graph, labels=label_dict, font_size=12, font_family='Times New Roman',
#         node_color='pink', node_size=150, edge_color='lightblue', width=0.8, alpha=0.7)
# plt.savefig('plot/graph.pdf', bbox_inches='tight')
# plt.close()

x = torch.tensor(x, dtype=torch.float, device=device)
y = torch.tensor(y, dtype=torch.float, device=device)
f = torch.tensor(f, dtype=torch.float, device=device)
x_training = x[:, : T_training, :]
y_training = y[:, : T_training]
f_training = f[: T_training, :]
x_validation = x[:, T_training : T_validation, :]
y_validation = y[:, T_training : T_validation]
f_validation = f[T_training - 1 : T_validation - 1, :]
x_test = x[:, T_validation :, :]
y_test = y[:, T_validation :]
f_test = f[T_validation - 1 : -1, :]
x_training_mean = torch.mean(x_training, dim=1, keepdim=True)
x_training_std = torch.std(x_training, dim=1, keepdim=True)
x_norm = (x - x_training_mean) / x_training_std

theta_tilde_0 = InitialTheta(y_training, f_training, device)
theta_hat_0 = activation(torch.unsqueeze(theta_tilde_0, dim=1)).expand(-1, y.shape[1] + 1, -1)


model = GNN(19, 16, 9, 4, device).to(device)
model.eval()
state_dict = torch.load(file_name)
model.load_state_dict(state_dict)
theta_hat = model.sequential(x_norm, edge_index, theta_tilde_0)
theta_hat = activation(theta_hat)

loss_fn = LossFunction().to(device)
loss, epsilon, sigma2 = loss_fn(y, f, theta_hat)
l, epsilon_garch, sigma2_garch = loss_fn(y, f, theta_hat_0)
epsilon_training = epsilon[:, : T_training]
epsilon_validation = epsilon[:, T_training : T_validation]
epsilon_test = epsilon[:, T_validation :]
sigma2_training = sigma2[:, : T_training]
sigma2_validation = sigma2[:, T_training : T_validation]
sigma2_test = sigma2[:, T_validation :]
epsilon_garch_training = epsilon_garch[:, : T_training]
epsilon_garch_validation = epsilon_garch[:, T_training : T_validation]
epsilon_garch_test = epsilon_garch[:, T_validation :]
sigma2_garch_training = sigma2_garch[:, : T_training]
sigma2_garch_validation = sigma2_garch[:, T_training : T_validation]
sigma2_garch_test = sigma2_garch[:, T_validation :]


def comp(epsilon, sigma2):
    h = torch.abs(epsilon)
    sigma = torch.sqrt(sigma2)
    h2 = torch.square(epsilon)
    print(torch.mean(torch.square(sigma - h)))
    print(torch.mean(torch.square(sigma2 - h2)))
    print(torch.mean(torch.log(sigma2) + h2 / sigma2))
    print(torch.mean(torch.square(torch.log(sigma2) - torch.log(h2))))
    print(torch.mean(torch.abs(sigma - h)))
    print(torch.mean(torch.abs(sigma2 - h2)))

comp(epsilon_garch_training, sigma2_garch_training)
comp(epsilon_training, sigma2_training)
comp(epsilon_garch_test, sigma2_garch_test)
comp(epsilon_test, sigma2_test)
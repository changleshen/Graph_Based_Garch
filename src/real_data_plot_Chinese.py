from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


device = torch.device('cpu')
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

dates = return_data.index
datelist = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
date_training = datelist[start_date + 1 : start_date + T_training + 1]
date_test = datelist[start_date + T_validation + 1 : end_date + 1]
 
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

edge_degree = np.mean(edge_index, axis=0)
plot_ind = np.where((edge_degree > 0.1) & (edge_degree < 0.25))[0]
print(plot_ind)
edge_index_plot = edge_index[plot_ind, :][:, plot_ind]
coo_edge_index = sp.coo_matrix(edge_index)
edge_index = torch.tensor(np.array([coo_edge_index.row, coo_edge_index.col]), device=device, dtype=torch.long)
coo_edge_index_plot = sp.coo_matrix(edge_index_plot)
edge_index_plot = torch.tensor(np.array([coo_edge_index_plot.row, coo_edge_index_plot.col]), device=device, dtype=torch.long)

return_plot = return_data.iloc[:, plot_ind]
label_dict = {i: col for i, col in enumerate(return_plot.columns)}
data = Data(edge_index=edge_index_plot, num_nodes=len(plot_ind))
graph = torch_geometric.utils.to_networkx(data)
graph = nx.Graph(graph)
pos = nx.spring_layout(graph, k=200/math.sqrt(Stocks))  # Layout algorithm for node positioning
plt.figure(figsize=(10, 10))
nx.draw(graph, labels=label_dict, font_size=14, font_family='Times New Roman',
        node_color='crimson', node_size=100, edge_color='lightskyblue', width=0.8, alpha=0.8)
plt.savefig('plot/real_data/graph.pdf', bbox_inches='tight')
plt.close()


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
state_dict = torch.load(file_name, map_location=device)
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

theta_hat = theta_hat[:, 1 :, :]
theta_training = theta_hat[:, : T_training, :]
theta_validation = theta_hat[:, T_training : T_validation, :]
theta_test = theta_hat[:, T_validation :, :]
theta_hat_0 = theta_hat_0[:, 1 :, :]
theta_hat_0_training = theta_hat_0[:, : T_training, :]
theta_hat_0_validation = theta_hat_0[:, T_training : T_validation, :]
theta_hat_0_test = theta_hat_0[:, T_validation :, :]

y_hat_training = theta_training[:, :, 0] + torch.sum(torch.mul(theta_training[:, :, 1:6], torch.unsqueeze(f_training, dim=0)), dim=-1)
y_hat_test = theta_test[:, :, 0] + torch.sum(torch.mul(theta_test[:, :, 1:6], torch.unsqueeze(f_test, dim=0)), dim=-1)
y_garch_training = theta_tilde_0[:, 0 : 1] + torch.matmul(theta_tilde_0[:, 1 : -3], torch.transpose(f_training, 0, 1))
y_garch_test = theta_tilde_0[:, 0 : 1] + torch.matmul(theta_tilde_0[:, 1 : -3], torch.transpose(f_test, 0, 1))

rf_test = factor_data.values[start_date + T_validation + 1 : end_date + 1, 5]

portfolio_size = int(0.1 * Stocks)
weight = torch.concat((torch.ones(portfolio_size, device=device), -1 * torch.ones(portfolio_size, device=device))) / portfolio_size

long_return, long_portfolio = torch.topk(y_hat_test, portfolio_size, dim=0)
short_return, short_portfolio = torch.topk(y_hat_test, portfolio_size, dim=0, largest=False)
portfolio = torch.concat((long_portfolio, short_portfolio), dim=0)
daily_return = torch.zeros(y_test.shape[1], device=device)
daily_profit = torch.zeros(y_test.shape[1], device=device)
new_high = torch.tensor(1)
max_DD = torch.tensor(0)
for t in range(y_test.shape[1]):
    daily_return[t] = torch.dot(weight, y_test[portfolio[:, t], t])
    if t == 0:
        daily_profit[t] = 1 + 0.01 * daily_return[t]
        new_high = daily_profit[t]
    else:
        daily_profit[t] = daily_profit[t - 1] * (1 + 0.01 * daily_return[t] + 0.01 * rf_test[t])
        if daily_profit[t] > new_high:
            new_high = daily_profit[t]
        if new_high - daily_profit[t] > max_DD:
            max_DD = (new_high - daily_profit[t]) / new_high
mean_return = torch.mean(daily_return) * 252
std_return = torch.std(daily_return) * math.sqrt(252)
print(mean_return)
print(std_return)
print(mean_return / std_return)
print(max_DD * 100)
print(torch.min(daily_return + torch.tensor(rf_test, device=device)))

long_return, long_portfolio = torch.topk(y_garch_test, portfolio_size, dim=0)
short_return, short_portfolio = torch.topk(y_garch_test, portfolio_size, dim=0, largest=False)
portfolio = torch.concat((long_portfolio, short_portfolio), dim=0)
daily_return_garch = torch.zeros(y_test.shape[1], device=device)
daily_profit_garch = torch.zeros(y_test.shape[1], device=device)
new_high = torch.tensor(1)
max_DD = torch.tensor(0)
for t in range(y_test.shape[1]):
    daily_return_garch[t] = torch.dot(weight, y_test[portfolio[:, t], t])
    if t == 0:
        daily_profit_garch[t] = 1 + 0.01 * daily_return_garch[t]
        new_high = daily_profit_garch[t]
    else:
        daily_profit_garch[t] = daily_profit_garch[t - 1] * (1 + 0.01 * daily_return_garch[t] + 0.01 * rf_test[t])
        if daily_profit_garch[t] > new_high:
            new_high = daily_profit_garch[t]
        if new_high - daily_profit_garch[t] > max_DD:
            max_DD = (new_high - daily_profit_garch[t]) / new_high
mean_return = torch.mean(daily_return_garch) * 252
std_return = torch.std(daily_return_garch) * math.sqrt(252)
print(mean_return)
print(std_return)
print(mean_return / std_return)
print(max_DD * 100)
print(torch.min(daily_return_garch + torch.tensor(rf_test, device=device)))

market_portfolio_return = np.squeeze(market_portfolio_data.values) - rf_test
market_portfolio_profit = np.zeros_like(market_portfolio_return)
new_high = 1
max_DD = 0
for t in range(market_portfolio_return.shape[0]):
    if t == 0:
        market_portfolio_profit[t] = 1 + 0.01 * market_portfolio_return[t]
        new_high = market_portfolio_profit[t]
    else:
        market_portfolio_profit[t] = market_portfolio_profit[t - 1] * (1 + 0.01 * market_portfolio_return[t] + 0.01 * rf_test[t])
        if market_portfolio_profit[t] > new_high:
            new_high = market_portfolio_profit[t]
        if new_high - market_portfolio_profit[t] > max_DD:
            max_DD = (new_high - market_portfolio_profit[t]) / new_high
mean_return = np.mean(market_portfolio_return) * 252
std_return = np.std(market_portfolio_return) * math.sqrt(252)
market_portfolio_profit = market_portfolio_profit - 1.0
print(mean_return)
print(std_return)
print(mean_return / std_return)
print(max_DD * 100)
print(np.min(market_portfolio_return + rf_test))


theta_training_np = theta_training.cpu().detach().numpy()
theta_test_np = theta_test.cpu().detach().numpy()
theta_hat_0_training_np = theta_hat_0_training.cpu().detach().numpy()
theta_hat_0_test_np = theta_hat_0_test.cpu().detach().numpy()

epsilon_training_np = epsilon_training.cpu().detach().numpy()
epsilon_test_np = epsilon_test.cpu().detach().numpy()
sigma2_training_np = sigma2_training.cpu().detach().numpy()
sigma2_test_np = sigma2_test.cpu().detach().numpy()
epsilon_garch_training_np = epsilon_garch_training.cpu().detach().numpy()
epsilon_garch_test_np = epsilon_garch_test.cpu().detach().numpy()
sigma2_garch_training_np = sigma2_garch_training.cpu().detach().numpy()
sigma2_garch_test_np = sigma2_garch_test.cpu().detach().numpy()

y_hat_training_np = y_hat_training.cpu().detach().numpy()
y_hat_test_np = y_hat_test.cpu().detach().numpy()
y_garch_training_np = y_garch_training.cpu().detach().numpy()
y_garch_test_np = y_garch_test.cpu().detach().numpy()
y_training_np = y_training.cpu().detach().numpy()
y_test_np = y_test.cpu().detach().numpy()

daily_return_np = daily_return.cpu().detach().numpy()
daily_profit_np = daily_profit.cpu().detach().numpy() - 1.0
daily_return_garch_np = daily_return_garch.cpu().detach().numpy()
daily_profit_garch_np = daily_profit_garch.cpu().detach().numpy() - 1.0



plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['times_simsun']
# plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

s = stats.randint.rvs(0, Stocks, size=1)[0]
s = stats.randint.rvs(0, Stocks, size=1)[0]


def plot_stock(s, folder_name):
    fig, axs = plt.subplots(9, 1, figsize=(15, 20))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[0].plot(date_training, 0.01 * theta_hat_0_training_np[s, :, 0], label='GARCH(1,1)', linestyle='dotted', 
                color='lightskyblue', linewidth=1.2, alpha=0.7)
    axs[0].plot(date_training, 0.01 * theta_training_np[s, :, 0], label='GNN-GARCH', color='dodgerblue')
    axs[0].set_title(r'$\hat{\alpha}_{t}$', fontdict={'fontsize': 16})
    axs[0].legend(fontsize=12)
    axs[0].tick_params(labelsize=12)

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[1].plot(date_training, theta_hat_0_training_np[s, :, 1], label='GARCH(1,1)', linestyle='dotted', 
                color='lightskyblue', linewidth=1.2, alpha=0.7)
    axs[1].plot(date_training, theta_training_np[s, :, 1], label='GNN-GARCH', color='dodgerblue')
    axs[1].set_title(r'$\hat{\beta}_{1t}$', fontdict={'fontsize': 16})
    axs[1].legend(fontsize=12)
    axs[1].tick_params(labelsize=12)

    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[2].plot(date_training, theta_hat_0_training_np[s, :, 2], label='GARCH(1,1)', linestyle='dotted', 
                color='lightskyblue', linewidth=1.2, alpha=0.7)
    axs[2].plot(date_training, theta_training_np[s, :, 2], label='GNN-GARCH', color='dodgerblue')
    axs[2].set_title(r'$\hat{\beta}_{2t}$', fontdict={'fontsize': 16})
    axs[2].legend(fontsize=12)
    axs[2].tick_params(labelsize=12)

    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[3].plot(date_training, theta_hat_0_training_np[s, :, 3], label='GARCH(1,1)', linestyle='dotted', 
                color='lightskyblue', linewidth=1.2, alpha=0.7)
    axs[3].plot(date_training, theta_training_np[s, :, 3], label='GNN-GARCH', color='dodgerblue')
    axs[3].set_title(r'$\hat{\beta}_{3t}$', fontdict={'fontsize': 16})
    axs[3].legend(fontsize=12)
    axs[3].tick_params(labelsize=12)

    axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[4].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[4].plot(date_training, theta_hat_0_training_np[s, :, 4], label='GARCH(1,1)', linestyle='dotted', 
                color='lightskyblue', linewidth=1.2, alpha=0.7)
    axs[4].plot(date_training, theta_training_np[s, :, 4], label='GNN-GARCH', color='dodgerblue')
    axs[4].set_title(r'$\hat{\beta}_{4t}$', fontdict={'fontsize': 16})
    axs[4].legend(fontsize=12)
    axs[4].tick_params(labelsize=12)

    axs[5].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[5].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[5].plot(date_training, theta_hat_0_training_np[s, :, 5], label='GARCH(1,1)', linestyle='dotted', 
                color='lightskyblue', linewidth=1.2, alpha=0.7)
    axs[5].plot(date_training, theta_training_np[s, :, 5], label='GNN-GARCH', color='dodgerblue')
    axs[5].set_title(r'$\hat{\beta}_{5t}$', fontdict={'fontsize': 16})
    axs[5].legend(fontsize=12)
    axs[5].tick_params(labelsize=12)

    axs[6].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[6].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[6].plot(date_training, theta_hat_0_training_np[s, :, 6], label='GARCH(1,1)', linestyle='dotted', 
                color='lightskyblue', linewidth=1.2, alpha=0.7)
    axs[6].plot(date_training, theta_training_np[s, :, 6], label='GNN-GARCH', color='dodgerblue')
    axs[6].set_title(r'$\hat{a}_{t}$', fontdict={'fontsize': 16})
    axs[6].legend(fontsize=12)
    axs[6].tick_params(labelsize=12)

    axs[7].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[7].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[7].plot(date_training, theta_hat_0_training_np[s, :, 7], label='GARCH(1,1)', linestyle='dotted', 
                color='lightskyblue', linewidth=1.2, alpha=0.7)
    axs[7].plot(date_training, theta_training_np[s, :, 7], label='GNN-GARCH', color='dodgerblue')
    axs[7].set_title(r'$\hat{b}_{t}$', fontdict={'fontsize': 16})
    axs[7].legend(fontsize=12)
    axs[7].tick_params(labelsize=12)

    axs[8].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[8].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[8].plot(date_training, 0.01 * 0.01 * theta_hat_0_training_np[s, :, 8], label='GARCH(1,1)', linestyle='dotted', 
                color='lightskyblue', linewidth=1.2, alpha=0.7)
    axs[8].plot(date_training, 0.01 * 0.01 * theta_training_np[s, :, 8], label='GNN-GARCH', color='dodgerblue')
    axs[8].set_title(r'$\hat{c}_{t}$', fontdict={'fontsize': 16})
    axs[8].legend(fontsize=12)
    axs[8].tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(folder_name + '/theta_training.pdf')
    plt.close()


    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[0].plot(date_training, y_training_np[s, :], label='真实值', color='orange')
    axs[0].plot(date_training, y_garch_training_np[s, :], label='GARCH(1,1)', color='lime')
    axs[0].plot(date_training, y_hat_training_np[s, :], label='GNN-GARCH', color='dodgerblue')
    axs[0].set_title('$\hat{y}_t$（单位：%）', fontdict={'fontsize': 16})
    axs[0].legend(fontsize=12)
    axs[0].tick_params(labelsize=12)

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[1].plot(date_training, epsilon_garch_training_np[s, :], label='GARCH(1,1)', color='lime')
    axs[1].plot(date_training, epsilon_training_np[s, :], label='GNN-GARCH', color='dodgerblue')
    axs[1].set_title('$\hat{\epsilon}_t$（单位：%）', fontdict={'fontsize': 16})
    axs[1].legend(fontsize=12)
    axs[1].tick_params(labelsize=12)

    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[2].plot(date_training, np.sqrt(sigma2_garch_training_np)[s, :], label='GARCH(1,1)', color='lime')
    axs[2].plot(date_training, np.sqrt(sigma2_training_np)[s, :], label='GNN-GARCH', color='dodgerblue')
    axs[2].set_title('$\hat{\sigma}_t$（单位：%）', fontdict={'fontsize': 16})
    axs[2].legend(fontsize=12)
    axs[2].tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(folder_name + '/y_training.pdf')
    plt.close()


    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sm.qqplot(epsilon_training_np[s, :], line ='s', ax=axs[0], markersize=5, markerfacecolor='dodgerblue')
    axs[0].set_title('GNN-GARCH 拟合得到的 $\hat{\epsilon}_{t}$ 的 Q-Q 图', fontdict={'fontsize': 16})
    axs[0].tick_params(labelsize=12)
    axs[0].set_xlabel('标准正态分布分位数', fontdict={'fontsize': 14})
    axs[0].set_ylabel('样本分位数', fontdict={'fontsize': 14})

    sm.qqplot(epsilon_garch_training_np[s, :], line ='s', ax=axs[1], markersize=5, markerfacecolor='dodgerblue')
    axs[1].set_title('GARCH(1,1) 拟合得到的 $\hat{\epsilon}_{t}$ 的 Q-Q 图', fontdict={'fontsize': 16})
    axs[1].tick_params(labelsize=12)
    axs[1].set_xlabel('标准正态分布分位数', fontdict={'fontsize': 14})
    axs[1].set_ylabel('样本分位数', fontdict={'fontsize': 14})

    plt.tight_layout()
    plt.savefig(folder_name + '/QQ.pdf')
    plt.close()


    plt.figure(figsize=(5, 5))
    sm.qqplot(epsilon_garch_training_np[s, :], line ='s', markersize=5, markerfacecolor='dodgerblue')
    # plt.title('$\hat{\epsilon}_{t}$ 的 QQ 图', fontdict={'fontsize': 16})
    plt.tick_params(labelsize=12)
    plt.xlabel('标准正态分布分位数', fontdict={'fontsize': 14})
    plt.ylabel('样本分位数', fontdict={'fontsize': 14})
    plt.tight_layout()
    plt.savefig(folder_name + '/epsilon_garch_qq.pdf')
    plt.close()

    plt.figure(figsize=(5, 5))
    sm.qqplot(epsilon_training_np[s, :], line ='s', markersize=5, markerfacecolor='dodgerblue')
    # plt.title('$\hat{\epsilon}_{t}$ 的 QQ 图', fontdict={'fontsize': 16})
    plt.tick_params(labelsize=12)
    plt.xlabel('标准正态分布分位数', fontdict={'fontsize': 14})
    plt.ylabel('样本分位数', fontdict={'fontsize': 14})
    plt.tight_layout()
    plt.savefig(folder_name + '/epsilon_qq.pdf')
    plt.close()


    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[0, 0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[0, 0].plot(date_test, 0.01 * theta_test_np[s, :, 0], color='dodgerblue')
    axs[0, 0].set_title(r'$\hat{\alpha}_{t}$', fontdict={'fontsize': 16})
    axs[0, 0].tick_params(labelsize=12)
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[0, 1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[0, 1].plot(date_test, theta_test_np[s, :, 1], color='dodgerblue')
    axs[0, 1].set_title(r'$\hat{\beta}_{1t}$', fontdict={'fontsize': 16})
    axs[0, 1].tick_params(labelsize=12)
    axs[0, 2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[0, 2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[0, 2].plot(date_test, theta_test_np[s, :, 2], color='dodgerblue')
    axs[0, 2].set_title(r'$\hat{\beta}_{2t}$', fontdict={'fontsize': 16})
    axs[0, 2].tick_params(labelsize=12)
    axs[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[1, 0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[1, 0].plot(date_test, theta_test_np[s, :, 3], color='dodgerblue')
    axs[1, 0].set_title(r'$\hat{\beta}_{3t}$', fontdict={'fontsize': 16})
    axs[1, 0].tick_params(labelsize=12)
    axs[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[1, 1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[1, 1].plot(date_test, theta_test_np[s, :, 4], color='dodgerblue')
    axs[1, 1].set_title(r'$\hat{\beta}_{4t}$', fontdict={'fontsize': 16})
    axs[1, 1].tick_params(labelsize=12)
    axs[1, 2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[1, 2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[1, 2].plot(date_test, theta_test_np[s, :, 5], color='dodgerblue')
    axs[1, 2].set_title(r'$\hat{\beta}_{5t}$', fontdict={'fontsize': 16})
    axs[1, 2].tick_params(labelsize=12)
    axs[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[2, 0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[2, 0].plot(date_test, theta_test_np[s, :, 6], color='dodgerblue')
    axs[2, 0].set_title(r'$\hat{a}_{t}$', fontdict={'fontsize': 16})
    axs[2, 0].tick_params(labelsize=12)
    axs[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[2, 1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[2, 1].plot(date_test, theta_test_np[s, :, 7], color='dodgerblue')
    axs[2, 1].set_title(r'$\hat{b}_{t}$', fontdict={'fontsize': 16})
    axs[2, 1].tick_params(labelsize=12)
    axs[2, 2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[2, 2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[2, 2].plot(date_test, 0.01 * 0.01 * theta_test_np[s, :, 8], color='dodgerblue')
    axs[2, 2].set_title(r'$\hat{c}_{t}$', fontdict={'fontsize': 16})
    axs[2, 2].tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(folder_name + '/theta_test.pdf')
    plt.close()



    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[0].plot(date_test, y_test_np[s, :], label='真实值', color='orange')
    axs[0].plot(date_test, y_garch_test_np[s, :], label='GARCH(1,1)', color='lime')
    axs[0].plot(date_test, y_hat_test_np[s, :], label='GNN-GARCH', color='dodgerblue')
    axs[0].set_title('$\hat{y}_t$（单位：%）', fontdict={'fontsize': 16})
    axs[0].legend(fontsize=12)
    axs[0].tick_params(labelsize=12)

    # axs[1].plot(epsilon_garch_test_np[s, :], label=r'$\tilde{\epsilon}_t$', color='lime')
    # axs[1].plot(epsilon_test_np[s, :], label=r'$\hat{\epsilon}_t$', color='dodgerblue')
    # axs[1].set_title(r'$\hat{\epsilon}_t$', fontdict={'fontsize': 16})
    # axs[1].legend(fontsize=12)
    # axs[1].tick_params(labelsize=12)

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axs[1].plot(date_test, np.sqrt(sigma2_garch_test_np)[s, :], label='GARCH(1,1)', color='lime')
    axs[1].plot(date_test, np.sqrt(sigma2_test_np)[s, :], label='GNN-GARCH', color='dodgerblue')
    axs[1].set_title('$\hat{\sigma}_t$（单位：%）', fontdict={'fontsize': 16})
    axs[1].legend(fontsize=12)
    axs[1].tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(folder_name + '/y_test.pdf')
    plt.close()



fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axs[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axs[0].plot(date_test, market_portfolio_return, label='S&P 500', color='lightskyblue')
axs[0].plot(date_test, daily_return_garch_np, label='GARCH(1,1)', color='lime')
axs[0].plot(date_test, daily_return_np, label='GNN-GARCH', color='dodgerblue')
axs[0].set_title('投资组合超额收益率（单位：%）', fontdict={'fontsize': 16})
axs[0].legend(fontsize=12)
axs[0].tick_params(labelsize=12)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axs[1].plot(date_test, 100 * market_portfolio_profit, label='S&P 500', color='lightskyblue')
axs[1].plot(date_test, 100 * daily_profit_garch_np, label='GARCH(1,1)', color='lime')
axs[1].plot(date_test, 100 * daily_profit_np, label='GNN-GARCH', color='dodgerblue')
axs[1].set_title('投资组合累计收益率（单位：%）', fontdict={'fontsize': 16})
axs[1].legend(fontsize=12)
axs[1].tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('plot/real_data/portfolio.pdf')
plt.close()


s = 1
ticker = return_data.columns[s]
folder_name = 'plot/real_data/' + ticker
plot_stock(s, folder_name)
s = 187
ticker = return_data.columns[s]
folder_name = 'plot/real_data/' + ticker
plot_stock(s, folder_name)
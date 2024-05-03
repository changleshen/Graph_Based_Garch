import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
from scipy import stats
import scipy.sparse as sp
from scipy import stats
import statsmodels.api as sm
import math
import pandas as pd
import gc


from model import GNN, training, LossFunction, SequentialGNN, activation, InitialTheta

device = torch.device('cuda:0')
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

return_data = pd.read_csv('data/return.csv', index_col=0)
factor_data = pd.read_csv('data/factors.csv', index_col=0)
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

mean_training = np.zeros((4, 5), dtype=float)
std_training = np.zeros((4, 5), dtype=float)
sharpe_training = np.zeros((4, 5), dtype=float)
R2_training = np.zeros((4, 5), dtype=float)

mean_validation = np.zeros((4, 5), dtype=float)
std_validation = np.zeros((4, 5), dtype=float)
sharpe_validation = np.zeros((4, 5), dtype=float)
R2_validation = np.zeros((4, 5), dtype=float)


mean_test = np.zeros((4, 5), dtype=float)
std_test = np.zeros((4, 5), dtype=float)
sharpe_test = np.zeros((4, 5), dtype=float)
R2_test = np.zeros((4, 5), dtype=float)
maxDD_test = np.zeros((4, 5), dtype=float)
maxLoss_test = np.zeros((4, 5), dtype=float)

i = 0
j = 0

for wd in [1, 2, 5]:
    j = 0
    for bs in [8, 16, 32, 64, 128]:
        # print(i)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        file_name = 'checkpoints/real_data_' + str(wd) + '_' + str(bs) + '.pth'
        model = training(x_training, y_training, f_training, theta_tilde_0, edge_index, 
                         model_name=file_name, 
                         device=device, hidden_dim=16, heads=4, 
                         learning_rate=5e-4, weight_decay=wd, num_epochs=100, batch_size=bs, 
                         sche=False, milestones=[300], gamma=0.1)
        # model = GNN(19, 16, 9, 4, device).to(device)
        # model.eval()
        # state_dict = torch.load(file_name)
        # model.load_state_dict(state_dict)
        theta_hat = model.sequential(x_norm, edge_index, theta_tilde_0)
        theta_hat = activation(theta_hat)

        loss_fn = LossFunction().to(device)
        loss, epsilon, sigma2 = loss_fn(y, f, theta_hat)
        sigma2_training = sigma2[:, : T_training]
        sigma2_validation = sigma2[:, T_training : T_validation]
        sigma2_test = sigma2[:, T_validation :]

        theta_hat = theta_hat[:, 1 :, :]
        theta_training = theta_hat[:, : T_training, :]
        theta_validation = theta_hat[:, T_training : T_validation, :]
        theta_test = theta_hat[:, T_validation :, :]
        y_hat_training = theta_training[:, :, 0] + torch.sum(torch.mul(theta_training[:, :, 1:6], torch.unsqueeze(f_training, dim=0)), dim=-1)
        y_hat_validation = theta_validation[:, :, 0] + torch.sum(torch.mul(theta_validation[:, :, 1:6], torch.unsqueeze(f_validation, dim=0)), dim=-1)
        y_hat_test = theta_test[:, :, 0] + torch.sum(torch.mul(theta_test[:, :, 1:6], torch.unsqueeze(f_test, dim=0)), dim=-1)

        portfolio_size = int(0.1 * Stocks)
        weight = torch.concat((torch.ones(portfolio_size, device=device), -1 * torch.ones(portfolio_size, device=device))) / portfolio_size

        long_return, long_portfolio = torch.topk(y_hat_training, portfolio_size, dim=0)
        short_return, short_portfolio = torch.topk(y_hat_training, portfolio_size, dim=0, largest=False)
        portfolio = torch.concat((long_portfolio, short_portfolio), dim=0)
        daily_return = torch.zeros(y_training.shape[1], device=device)
        daily_profit = torch.zeros(y_training.shape[1], device=device)
        for t in range(y_training.shape[1]):
            daily_return[t] = torch.dot(weight, y_training[portfolio[:, t], t])
            if t == 0:
                daily_profit[t] = daily_return[t]
            else:
                daily_profit[t] = daily_profit[t - 1] + daily_return[t]
        mean_return = torch.mean(daily_return) * 252
        std_return = torch.std(daily_return) * math.sqrt(252)
        mean_training[i, j] = mean_return.item()
        std_training[i, j] = std_return.item()
        R2 = torch.sum(torch.square(y_hat_training - y_training)) / torch.sum(torch.square(y_training))
        R2_training[i, j] = R2.item()

        long_return, long_portfolio = torch.topk(y_hat_validation, portfolio_size, dim=0)
        short_return, short_portfolio = torch.topk(y_hat_validation, portfolio_size, dim=0, largest=False)
        portfolio = torch.concat((long_portfolio, short_portfolio), dim=0)
        daily_return = torch.zeros(y_validation.shape[1], device=device)
        daily_profit = torch.zeros(y_validation.shape[1], device=device)
        for t in range(y_validation.shape[1]):
            daily_return[t] = torch.dot(weight, y_validation[portfolio[:, t], t])
            if t == 0:
                daily_profit[t] = daily_return[t]
            else:
                daily_profit[t] = daily_profit[t - 1] + daily_return[t]
        mean_return = torch.mean(daily_return) * 252
        std_return = torch.std(daily_return) * math.sqrt(252)
        mean_validation[i, j] = mean_return.item()
        std_validation[i, j] = std_return.item()
        R2 = torch.sum(torch.square(y_hat_validation - y_validation)) / torch.sum(torch.square(y_validation))
        R2_validation[i, j] = R2.item()

        long_return, long_portfolio = torch.topk(y_hat_test, portfolio_size, dim=0)
        short_return, short_portfolio = torch.topk(y_hat_test, portfolio_size, dim=0, largest=False)
        portfolio = torch.concat((long_portfolio, short_portfolio), dim=0)
        daily_return = torch.zeros(y_test.shape[1], device=device)
        daily_profit = torch.zeros(y_test.shape[1], device=device)
        new_high = torch.tensor(0)
        max_DD = torch.tensor(0)
        for t in range(y_test.shape[1]):
            daily_return[t] = torch.dot(weight, y_test[portfolio[:, t], t])
            if t == 0:
                daily_profit[t] = daily_return[t]
                new_high = daily_return[t]
            else:
                daily_profit[t] = daily_profit[t - 1] + daily_return[t]
                if daily_profit[t] > new_high:
                    new_high = daily_profit[t]
                if new_high - daily_profit[t] > max_DD:
                    max_DD = new_high - daily_profit[t]
        mean_return = torch.mean(daily_return) * 252
        std_return = torch.std(daily_return) * math.sqrt(252)
        mean_test[i, j] = mean_return.item()
        std_test[i, j] = std_return.item()
        R2 = torch.sum(torch.square(y_hat_test - y_test)) / torch.sum(torch.square(y_test))
        R2_test[i, j] = R2.item()
        maxDD_test[i, j] = max_DD.item()
        maxLoss_test[i, j] = torch.min(daily_return).item()

        j = j + 1

    i = i + 1 

sharpe_training = mean_training / std_training
sharpe_validation = mean_validation / std_validation
sharpe_test = mean_test / std_test

print(sharpe_training)
print(sharpe_validation)
print(sharpe_test)
print(maxDD_test)
print(maxLoss_test)

print(R2_training)
print(R2_validation)
print(R2_test)
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


from model import GNN, training, LossFunction, SequentialGNN, activation, InitialTheta

device = torch.device('cuda:0')

def generate_simulation(x, f, edge_index, device, Tdist=False, df=None, hidden_dim=16, heads=2, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_stocks = x.shape[0]
    num_times = x.shape[1]
    output_dim = f.shape[-1] + 4
    input_dim = x.shape[2] + output_dim

    model = GNN(input_dim, hidden_dim, output_dim, heads, device).to(device)
    x_norm = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)
    sim_theta_tilde = model.sequential(x_norm, edge_index)
    sim_theta = activation(sim_theta_tilde)
    sim_theta = sim_theta[:, 1 :, :]
    sim_a_plus_b = sim_theta[:, :, -3] + sim_theta[:, :, -2]
    sim_error = torch.zeros((num_stocks, num_times), device=device)
    sim_volatility = torch.zeros_like(sim_error, device=device)
    if Tdist:
        t_4 = torch.distributions.StudentT(df=df)
        sim_stdnorm = math.sqrt(0.5) * t_4.sample((num_stocks, num_times)).to(device)
    else:
        sim_stdnorm = torch.randn_like(sim_error, device=device)
    
    for t in range(num_times):
        if t == 0:
            sim_volatility[:, t] = sim_theta[:, t, -1]  + \
                torch.mul(sim_a_plus_b[:, t], torch.mean(sim_theta[:, :, -1], dim=1) / (1.0 - torch.mean(sim_a_plus_b, dim=1)))
        else:
            sim_volatility[:, t] = sim_theta[:, t, -1] + \
                torch.mul(sim_theta[:, t, -3], sim_volatility[:, t - 1]) + \
                    torch.mul(sim_theta[:, t, -2], torch.square(sim_error[:, t - 1]))
        sim_error[:, t] = torch.mul(torch.sqrt(sim_volatility[:, t]), sim_stdnorm[:, t])
    sim_y = sim_theta[:, :, 0] + \
        torch.sum(torch.mul(sim_theta[:, :, 1:-3], torch.unsqueeze(f, dim=0)), dim=-1) + sim_error
    sim_theta = sim_theta.detach()
    sim_y = sim_y.detach()
    return model, sim_theta, sim_y, sim_error, sim_volatility

np.random.seed(42)
Stocks = 400
Times = 200
hidden_dim = 12

edge_index = sp.random(Stocks, Stocks, density=0.5 - 1 / np.sqrt(5), data_rvs=stats.binom(n=1, p=1).rvs)
edge_index = edge_index - np.transpose(edge_index)
coo_edge_index = sp.coo_matrix(edge_index)
edge_index = np.array([coo_edge_index.row, coo_edge_index.col])
edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
# data = Data(edge_index=edge_index, num_nodes=Stocks)
# graph = torch_geometric.utils.to_networkx(data)
# graph = nx.Graph(graph)
# Plot the graph
# # # pos = nx.spring_layout(graph)  # Layout algorithm for node positioning
# nx.draw(graph, pos, with_labels=False, node_color='lightblue', node_size=10, edge_color='gray', width=1.0, alpha=0.7)
# Show the plot
# plt.savefig('plot/t_S=200_T=200_new/graph.png')
# plt.close()
x = np.zeros((Stocks, Times, 5))
f = np.zeros((Times, 3))
Phit_f = np.array([[0.4, -0.1, 0.0], 
                   [0.0, 0.1, 0.2], 
                   [-0.1, 0.3, 0.3]])
Lt_f = np.array([[0.5, 0.2, -0.1], 
                 [0.0, 0.3, 0.2], 
                 [0.0, 0.0, 0.4]])

Phit_x = np.array([[0.7, 0.0, 0.1, -0.2, 0.3], 
                   [-0.1, 0.4, 0.5, 0.2, -0.3], 
                   [0.3, -0.1, 0.3, 0.0, 0.1], 
                   [0.0, 0.2, -0.1, 0.5, 0.3], 
                   [0.2, -0.3, 0.1, -0.2, 0.2]])
Lt_x = np.array([[0.5, 0.2, 0.1, -0.2, 0.3], 
                 [0.0, 0.8, -0.5, 0.2, 0.3], 
                 [0.0, 0.0, 0.4, 0.2, 0.3], 
                 [0.0, 0.0, 0.0, 0.6, 0.1], 
                 [0.0, 0.0, 0.0, 0.0, 0.3]])
x_error = np.dot(np.random.randn(Stocks, Times, 5), Lt_x)
f_error = np.dot(np.random.randn(Times, 3), Lt_f)
x_0_mean = np.random.randn(Stocks, 5)
f_0_mean = np.random.randn(3) + np.array([0.0, 0.0, 0.0])
for t in range(Times):
    if t == 0:
        x[:, t, :] = x_0_mean + x_error[:, t, :]
        f[t, :] = f_0_mean + f_error[t, :]
    else:
        x[:, t, :] = x_0_mean + np.dot((x[:, t - 1, :] - x_0_mean), Phit_x) + x_error[:, t, :]
        f[t, :] = f_0_mean + np.dot((f[t - 1, :] - f_0_mean), Phit_f) + f_error[t, :]

x = torch.tensor(x, dtype=torch.float, device=device)
f = torch.tensor(f, dtype=torch.float, device=device)

torch.manual_seed(0)
torch.cuda.manual_seed(0)
model = GNN(12, hidden_dim, 7, 4, device).to(device)
x_norm = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)
sim_theta_tilde = model.sequential(x_norm, edge_index)
sim_theta = activation(sim_theta_tilde)

torch.save(x, 'data_simulation/S400T200/x.pt')
torch.save(f, 'data_simulation/S400T200/f.pt')
torch.save(edge_index, 'data_simulation/S400T200/edge_index.pt')
torch.save(model.state_dict(), 'checkpoints_simulation/S400T200/0.pth')
torch.save(sim_theta, 'data_simulation/S400T200/theta.pt')
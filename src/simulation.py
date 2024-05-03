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
import argparse


from model import GNN, training, LossFunction, SequentialGNN, activation, InitialTheta

device = torch.device('cuda:1')

def generate_simulation(x, f, sim_theta, device, Tdist=False, df=None, hidden_dim=16, heads=2, seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_stocks = x.shape[0]
    num_times = x.shape[1]

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
    sim_y = sim_y.detach()
    return sim_y, sim_error, sim_volatility


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

Stocks = 200
Times = 200

x = torch.load('data_simulation/S200T200/x.pt', map_location=device)
f = torch.load('data_simulation/S200T200/f.pt', map_location=device)
edge_index = torch.load('data_simulation/S200T200/edge_index.pt', map_location=device)
sim_theta = torch.load('data_simulation/S200T200/theta.pt', map_location=device)

sim_y, sim_error, sim_vol = \
    generate_simulation(x, f, sim_theta, device, 
                        Tdist=False, df=4, hidden_dim=12, heads=4, seed=args.seed)
loss_fn = LossFunction().to(device)
loss, __, __ = loss_fn(sim_y, f, sim_theta)
print(loss.item())

theta_tilde_0 = torch.zeros((Stocks, 7), dtype=torch.float, device=device)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

file_name = 'checkpoints_simulation/S200T200/' + str(args.seed) + '.pth'

model = training(x, sim_y, f, theta_tilde_0, edge_index, 
                 model_name=file_name, device=device, hidden_dim=12, heads=4, 
                 learning_rate=1e-3, weight_decay=2, num_epochs=100, batch_size=8)

x_norm = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)
theta_tilde = model.sequential(x_norm, edge_index)
theta = activation(theta_tilde)
file_name_theta = 'data_simulation/S200T200/theta_' + str(args.seed) + '.pt'
torch.save(theta, file_name_theta)
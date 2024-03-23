import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
from scipy import stats
import scipy.sparse as sp


from model import GNN, training, LossFunction

def generate_simulation(x, f, edge_index, device, 
                        input_dim=10, feature_dim=15, hidden_dim=16, output_dim=9, heads=2):
    model = GNN(input_dim, feature_dim, hidden_dim, output_dim, heads, device).to(device)

    x_norm = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)
    sim_theta = model(x_norm, edge_index)
    sim_error = torch.zeros((x.shape[0], x.shape[1]), device=device)
    sim_volatility = torch.zeros_like(sim_error, device=device)
    sim_stdnorm = torch.randn_like(sim_error, device=device)
    for t in range(x.shape[1]):
        if t == 0:
            sim_volatility[:, t] = sim_theta[:, t, -1]
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


def plot_theta(sim_theta, real_theta, stock):
    # Plot model parameters, with both simulated values and fitted values.
    # Factor model parameters alpha and beta 
    # Garch parameters a, b, and c

    fig, axs = plt.subplots(5, 3, figsize=(20, 20))

    # Factor model parameters alpha
    axs[0, 0].plot(sim_theta[stock, :, 0], label='simulated alpha')
    axs[0, 0].plot(real_theta[stock, :, 0], label='fitted alpha')
    axs[0, 0].legend()
    axs[0, 1].scatter(range(sim_theta.shape[1]), sim_theta[stock, :, 0]-real_theta[stock, :, 0], s=5, label='differences')
    axs[0, 1].legend()

    # Factor model parameters beta
    axs[1, 0].plot(sim_theta[stock, :, 1], label='simulated beta_1')
    axs[1, 0].plot(real_theta[stock, :, 1], label='fitted beta_1')
    axs[1, 0].legend()
    axs[1, 1].plot(sim_theta[stock, :, 2], label='simulated beta_2')
    axs[1, 1].plot(real_theta[stock, :, 2], label='fitted beta_2')
    axs[1, 1].legend()
    axs[1, 2].plot(sim_theta[stock, :, 3], label='simulated beta_3')
    axs[1, 2].plot(real_theta[stock, :, 3], label='fitted beta_3')
    axs[1, 2].legend()

    axs[2, 0].scatter(range(sim_theta.shape[1]), sim_theta[stock, :, 1]-real_theta[stock, :, 1], s=5, label='differences')
    axs[2, 0].legend()
    axs[2, 1].scatter(range(sim_theta.shape[1]), sim_theta[stock, :, 2]-real_theta[stock, :, 2], s=5, label='differences')
    axs[2, 1].legend()
    axs[2, 2].scatter(range(sim_theta.shape[1]), sim_theta[stock, :, 3]-real_theta[stock, :, 3], s=5, label='differences')
    axs[2, 2].legend()

    # Garch parameters a
    axs[3, 0].plot(sim_theta[stock, 1:, -3], label='simulated a')
    axs[3, 0].plot(real_theta[stock, 1:, -3], label='fitted a')
    axs[3, 0].legend()
    axs[4, 0].scatter(range(sim_theta.shape[1]-1), sim_theta[stock, 1:, -3]-real_theta[stock, 1:, -3], s=5, label='differences')
    axs[4, 0].legend()

    # Garch parameters b
    axs[3, 1].plot(sim_theta[stock, 1:, -2], label='simulated b')
    axs[3, 1].plot(real_theta[stock, 1:, -2], label='fitted b')
    axs[3, 1].legend()
    axs[4, 1].scatter(range(sim_theta.shape[1]-1), sim_theta[stock, 1:, -2] - real_theta[stock, 1:, -2], s=5, label='differences')
    axs[4, 1].legend()

    # Garch parameters c
    axs[3, 2].plot(sim_theta[stock, 1:, -1], label='simulated c')
    axs[3, 2].plot(real_theta[stock, 1:, -1], label='fitted c')
    axs[3, 2].legend()
    axs[4, 2].scatter(range(sim_theta.shape[1]-1), sim_theta[stock, 1:, -1] - real_theta[stock, 1:, -1], s=5, label='differences')
    axs[4, 2].legend()


    plt.tight_layout()
    plt.savefig('plot/S=200_T=200_new/parameters/stock_' + str(stock) + '.png')
    plt.close()


def plot_x_y(x, y, sim_e, e, sim_vol, stock):
    
    fig, axs = plt.subplots(3, 3, figsize=(20, 15))

    # Plot simulated input features x, and target y
    axs[0, 0].plot(x[stock, :, 0], label='x_1')
    axs[0, 0].legend()
    axs[0, 1].plot(x[stock, :, 1], label='x_2')
    axs[0, 1].legend()
    axs[0, 2].plot(x[stock, :, 2], label='x_3')
    axs[0, 2].legend()
    axs[1, 0].plot(x[stock, :, 3], label='x_4')
    axs[1, 0].legend()
    axs[1, 1].plot(x[stock, :, 4], label='x_5')
    axs[1, 1].legend()
    axs[1, 2].plot(y[stock, :], label='y')
    axs[1, 2].legend()

    # Plot simulated errors against fitted errors.
    axs[2, 0].plot(sim_e[stock, 1:], label='simulated error')
    axs[2, 0].plot(y[stock, 1:] - sim_e[stock, 1:], label='simulated factors')
    axs[2, 0].legend()
    axs[2, 1].plot(sim_e[stock, 1:], label='simulated error')
    axs[2, 1].plot(e[stock, :], label='fitted error')
    axs[2, 1].legend()
    # axs[2, 2].scatter(range(e.shape[1]), e[stock, :] - sim_e[stock, 1:], s=5, label='diffenrences')
    # axs[2, 2].legend()
    axs[2, 2].plot(np.sqrt(sim_vol[stock, 1:]), label='simulated volatility')
    axs[2, 2].legend()

    plt.tight_layout()
    plt.savefig('plot/S=200_T=200_new/x_and_y/stock_' + str(stock) + '.png')
    plt.close()


def plot_factor(f):
    # Plot factor f.

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    axs[0].plot(f[:, 0], label='f_1')
    axs[0].legend()
    axs[1].plot(f[:, 1], label='f_2')
    axs[1].legend()
    axs[2].plot(f[:, 2], label='f_3')
    axs[2].legend()
    plt.tight_layout()
    plt.savefig('plot/S=200_T=200_new/factor.png')
    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    np.random.seed(42)

    Stocks = 200
    Times = 200

    edge_index = sp.random(Stocks, Stocks, density=0.5 - 1 / np.sqrt(5), data_rvs=stats.binom(n=1, p=1).rvs)
    edge_index = edge_index - np.transpose(edge_index)
    coo_edge_index = sp.coo_matrix(edge_index)
    edge_index = torch.tensor(np.array([coo_edge_index.row, coo_edge_index.col]), device=device, dtype=torch.long)
    data = Data(edge_index=edge_index)
    graph = torch_geometric.utils.to_networkx(data)
    graph = nx.Graph(graph)
    # Plot the graph
    pos = nx.spring_layout(graph)  # Layout algorithm for node positioning
    nx.draw(graph, pos, with_labels=False, node_color='lightblue', node_size=10, edge_color='gray', width=1.0, alpha=0.7)
    # Show the plot
    plt.savefig('plot/S=200_T=200_new/graph.png')
    plt.close()

    x = torch.zeros((Stocks, Times, 5), device=device, dtype=torch.float)
    f = torch.zeros((Times, 3), device=device, dtype=torch.float)
    Phit_f = torch.tensor(np.array([[0.4, 0.1, 0.0], 
                                    [0.0, 0.1, 0.2], 
                                    [0.1, 0.3, 0.3]]), device=device, dtype=torch.float)
    Lt_f = torch.tensor(np.array([[0.5, 0.2, 0.1], 
                                  [0.0, 0.3, 0.2], 
                                  [0.0, 0.0, 0.4]]), device=device, dtype=torch.float)

    x_error = 0.5 * torch.randn((Stocks, Times, 5), device=device, dtype=torch.float)
    f_error = torch.matmul(torch.randn((Times, 3), device=device, dtype=torch.float), Lt_f)
    x_0_mean = torch.randn((Stocks, 5), device=device, dtype=torch.float)
    f_0_mean = torch.randn(3, device=device, dtype=torch.float) + \
        torch.tensor(np.array([0, 0, 0]), device=device, dtype=torch.float)
    for t in range(Times):
        if t == 0:
            x[:, t, :] = x_0_mean + x_error[:, t, :]
            f[t, :] = f_0_mean + f_error[t, :]
        else:
            x[:, t, :] = x_0_mean + 0.866 * (x[:, t - 1, :] - x_0_mean) + x_error[:, t, :]
            f[t, :] = f_0_mean + torch.matmul((f[t - 1, :] - f_0_mean), Phit_f) + f_error[t, :]
    x = x.detach()
    f = f.detach()

    model_0, sim_theta, sim_y, sim_error, sim_vol = generate_simulation(x, f, edge_index, device, 
                                                    input_dim=5, hidden_dim=12, output_dim=7, heads=4)
    
    x_np = x.cpu().numpy()
    y_np = sim_y.cpu().numpy()
    sim_e_np = sim_error.detach().cpu().numpy()
    sim_vol_np = sim_vol.detach().cpu().numpy()
    
    f_np = f.cpu().numpy()
    plot_factor(f_np)

   
    loss_fn = LossFunction().to(device)
    loss, __, __ = loss_fn(sim_y, f, sim_theta, 0)
    print(f"Original Loss: {loss.item()}\n")

    model, theta, e, s2 = training(x, sim_y, f, edge_index, model_name='checkpoints/simulation_S=200_T=200_new.pth',
                        device=device, input_dim=5, hidden_dim=12, output_dim=7, heads=4, 
                        learning_rate=1e-2,  weight_decay=0.01, num_epochs=100, sche=False)

    theta_np = theta.cpu().detach().numpy()
    sim_theta_np = sim_theta.cpu().detach().numpy()
    e_np = e.cpu().detach().numpy()
    s2_np = s2.cpu().detach().numpy()

    for stock in range(3):
        plot_theta(sim_theta_np, theta_np, stock)
        plot_x_y(x_np, y_np, sim_e_np, e_np, sim_vol_np, stock)

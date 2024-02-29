import copy
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch_geometric.nn import TransformerConv, SAGEConv
from arch.univariate import arch_model

from load_data import load_EOD_data



class GATCell(nn.Module):
    # Input is the embedding features, output is a hidden layer that needs a transform to be the Garch parameters.
    def __init__(self, input_dim, hidden_dim, output_dim, heads):
        super().__init__()
        self.conv = TransformerConv(in_channels=input_dim, out_channels=hidden_dim, heads=heads, concat=False)
        #self.conv = SAGEConv(in_channels=input_dim, out_channels=hidden_dim, bias=False)
        self.mlp = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x, theta, edge_index):
        x_in = torch.concat((x, theta), dim=-1)
        hidden = self.conv(x_in, edge_index)
        hidden = self.tanh(hidden)
        theta_out = self.mlp(hidden)
        return theta_out


class GNN(nn.Module):
    # x is a S*T*N tensor
    # x[t] refers to x_t, y[t] refers to y_{t+1}, theta[t] refers to \theta_{t+1}.
    # output is the factor model parameters and the Garch model parameters
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, heads, device):
        super().__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=feature_dim, batch_first=True)
        self.gat_cell = GATCell(feature_dim + output_dim, hidden_dim, output_dim, heads)
        #self.gat_cell = GATCell(input_dim + output_dim, hidden_dim, output_dim, heads)
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x, edge_index, theta_t=None):
        embedding, _ = self.lstm(x)
        num_stock = x.size(dim=0)
        num_time = x.size(dim=1)
        theta_list = []
        
        # recursively getting the hidden layers
        if theta_t is None:
            theta_t = torch.zeros((num_stock, self.output_dim), requires_grad=True).to(self.device)
        for t in range(num_time):
            theta_t = self.gat_cell(embedding[:, t, :], theta_t, edge_index)
            #theta_t = self.gat_cell(x[:, t, :], theta_t, edge_index)
            theta_list.append(theta_t)
        theta = torch.stack(theta_list, dim=1)

        # transform to get the Garch parameters
        alpha_beta = theta[:, :, 0:-3]
        a_b = self.sigmoid(theta[:, :, -3:-1])
        a = torch.mul(a_b[:, :, 0:1], a_b[:, :, 1:])
        b = a_b[:, :, 0:1] - a
        c = self.softplus(theta[:, :, -1:])
        output = torch.concat((alpha_beta, a, b, c), dim=-1)
        return output


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, f, theta):
        num_time = y.size(dim=1)
        epsilon = []
        sigma2 = []
        for t in range(num_time):
            if t == 0:
                sigma2.append(theta[:, 0, -1])
            else:
                sigma2.append(theta[:, t, -1] + \
                              torch.mul(theta[:, t, -3], sigma2[-1]) + \
                                torch.mul(theta[:, t, -2], torch.square(epsilon[-1])))
            epsilon.append(y[:, t] - theta[:, t, 0] - torch.matmul(theta[:, t, 1:-3], f[t]))
        # when calculating the loss function, iterate from t=2
        epsilon.pop(0)
        sigma2.pop(0)
        epsilon = torch.stack(epsilon, dim=1)
        epsilon2 = torch.square(epsilon)
        sigma2 = torch.stack(sigma2, dim=1)
        loss = torch.mean(epsilon2/sigma2 + torch.log(sigma2))
        return loss, epsilon, sigma2


def training(x, y, f, edge_index, model_name, 
             device, input_dim, feature_dim, hidden_dim, output_dim, heads, 
             learning_rate, num_epochs, sche=False, milestones=None, gamma=None):

    model = GNN(input_dim, feature_dim, hidden_dim, output_dim, heads, device).to(device)
    loss_fn = LossFunction().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if sche:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    x_norm = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)
    loss_per_item = np.zeros(num_epochs, dtype=float)
    
    for epoch in range(num_epochs):
        model.train()
        theta = model(x_norm, edge_index)
        loss, e, s2 = loss_fn(y, f, theta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if sche:
            scheduler.step()
        loss_per_item[epoch] = loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_per_item[epoch]}")
    torch.save(model, model_name)
    theta = model(x_norm, edge_index)
    return model, theta, e, s2



if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    validation_ind = 756
    validation_ind = 400
    test_ind = 1008
    tickers =  np.genfromtxt('data/NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv', 
                             dtype=str, delimiter='\t', skip_header=False)
    features, excess_return, masks, factors = load_EOD_data('data/2013-01-01', 'NASDAQ', tickers)
    #np.savetxt('data/excess_return.csv', 1000 * excess_return[:, 100:600], fmt='%.6f', delimiter=',')
    x = torch.tensor(features, device=device, dtype=torch.float)
    y = torch.tensor(excess_return, device=device, dtype=torch.float)
    f = torch.tensor(factors, device=device, dtype=torch.float)
    training_ind = copy.copy(masks)
    training_ind[validation_ind:] = False
    training_x = x[:, training_ind, :]
    training_y = y[:, training_ind] * 1000
    training_f = f[training_ind, :-1]

    edge_index = np.load("data/relation/NASDAQ_edge_index.npy")
    edge_index = torch.tensor(edge_index, device=device, dtype=torch.long)

    model, theta, es, s = training(training_x, training_y, training_f, edge_index, 
                                   model_name='checkpoints/observation_0-400/model_xx.pth',
             device=device, input_dim=10, feature_dim=15, output_dim=9, heads=2, 
             learning_rate=5*1e-2, num_epochs=20, sche=True, milestones=[20,40,80], gamma=0.2)

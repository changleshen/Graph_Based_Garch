import copy
import torch
import numpy as np
from torch import nn
from torch.utils.data import RandomSampler, BatchSampler
from torch.nn.utils.parametrizations import spectral_norm
from torch_geometric.nn import TransformerConv
from arch import arch_model
import time


class GNN(nn.Module):
    # Input is the embedding features, output is a hidden layer that needs a transform to be the Garch parameters.
    def __init__(self, input_dim, hidden_dim, output_dim, heads, device):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self.conv = TransformerConv(in_channels=input_dim, out_channels=hidden_dim, heads=heads, concat=False)
        self.mlp = spectral_norm(nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=False))
        self.tanh = nn.Tanh()

    def forward(self, input, edge_index):
        hidden = self.conv(input, edge_index)
        hidden = self.tanh(hidden)
        theta_out = self.mlp(hidden)
        return theta_out
    
    def sequential(self, x_in, edge_index, theta_t=None):
        theta_tilde_list = []
        num_stocks = x_in.shape[0]
        num_times = x_in.shape[1]
        if theta_t is None:
            theta_t = torch.zeros((num_stocks, self.output_dim), requires_grad=True, device=self.device)
        theta_tilde_list.append(theta_t)
        for t in range(num_times):
            theta_t = self.forward(torch.concat((x_in[:, t, :], theta_t), dim=-1), edge_index)
            theta_tilde_list.append(theta_t)
        theta_tilde = torch.stack(theta_tilde_list, dim=1)
        return theta_tilde
  

def SequentialGNN(model, x_norm, edge_index, output_dim, device, theta_t=None):
    theta_tilde_list = []
    num_stocks = x_norm.shape[0]
    num_times = x_norm.shape[1]
    if theta_t is None:
        theta_t = torch.zeros((num_stocks, output_dim), requires_grad=True, device=device)
    for t in range(num_times):
        theta_t = model(torch.concat((x_norm[:, t, :], theta_t), dim=-1), edge_index)
        theta_tilde_list.append(theta_t)
    theta_tilde = torch.stack(theta_tilde_list, dim=1)
    return theta_tilde


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, f, theta):
        theta = theta[:, 1 :, :]
        num_time = y.size(dim=1)
        a_plus_b = theta[:, :, -3] + theta[:, :, -2]
        epsilon = y - theta[:, :, 0] - torch.sum(torch.mul(theta[:, :, 1:-3], torch.unsqueeze(f, dim=0)), dim=-1)
        epsilon2 = torch.square(epsilon)
        sigma2 = []
        for t in range(num_time):
            if t == 0:
                sigma2.append(theta[:, t, -1] + \
                              torch.mul(a_plus_b[:, t], torch.mean(theta[:, :, -1], dim=1) / (1.0 - torch.mean(a_plus_b, dim=1))))
            else:
                sigma2.append(theta[:, t, -1] + \
                              torch.mul(theta[:, t, -3], sigma2[-1]) + \
                                torch.mul(theta[:, t, -2], epsilon2[:, t - 1]))
        sigma2 = torch.stack(sigma2, dim=1)
        loss = torch.mean(epsilon2/sigma2 + torch.log(sigma2))
        return loss, epsilon, sigma2


class ApproxLossFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, f, theta, sigma2old):
        theta = theta[:, 1 :, :]
        a_plus_b = theta[:, :, -3] + theta[:, :, -2]
        epsilon = y - theta[:, :, 0] - torch.sum(torch.mul(theta[:, :, 1:-3], torch.unsqueeze(f, dim=0)), dim=-1)
        epsilon2 = torch.square(epsilon)
        init_value = torch.mean(theta[:, :, -1], dim=1, keepdim=True) / (1.0 - torch.mean(a_plus_b, dim=1, keepdim=True))
        epsilon2new = torch.concat((init_value, epsilon2[:, : -1]), dim=1)
        sigma2new = torch.concat((init_value, sigma2old[:, : -1]), dim=1)
        sigma2 = theta[:, :, -1] + torch.mul(theta[:, :, -2], epsilon2new) + torch.mul(theta[:, :, -3], sigma2new)
        loss = torch.mean(epsilon2/sigma2 + torch.log(sigma2))
        return loss, epsilon, sigma2


def activation(theta_tilde):
    alpha_beta = theta_tilde[:, :, 0 : -3]
    a_b = nn.Sigmoid()(theta_tilde[:, :, -3 : -1])
    a = torch.mul(a_b[:, :, 0 : 1], a_b[:, :, 1 :])
    b = a_b[:, :, 0 : 1] - a
    c = nn.Softplus()(theta_tilde[:, :, -1 :])
    theta = torch.concat((alpha_beta, a, b, c), dim=-1)
    return theta


def inv_activation(theta):
    alpha_beta = theta[:, 0 : -3]
    theta[:, -2 :] = torch.where(theta[:, -2 :] == 0, 1e-4, theta[:, -2 :])
    a_b = theta[:, -2 : -1] + theta[:, -1 :]
    a_b = torch.where(a_b >= 1.0, 1.0 - 1e-4, a_b)
    x = torch.log(a_b) - torch.log(1.0 - a_b)
    y = torch.log(theta[:, -1 :]) - torch.log(theta[:, -2 : -1])
    z = torch.log(torch.exp(theta[:, -3 : -2] - 1.0))
    theta_tilde = torch.concat((alpha_beta, x, y, z), dim=-1)
    return theta_tilde


def GarchEstimation(y, f):
    Garch_model = arch_model(y=y, x=f, mean='LS', vol='GARCH', p=1, q=1)
    result = Garch_model.fit(disp='off')
    return result.params


def InitialTheta(y, f, device):
    y_np = y.detach().cpu().numpy()
    f_np = f.detach().cpu().numpy()
    vec_GarchEstimation = np.vectorize(GarchEstimation, excluded=['f'], signature='(n)->(m)')
    theta_0 = vec_GarchEstimation(y=y_np, f=f_np)
    theta_0 = torch.tensor(theta_0, dtype=torch.float, device=device)
    theta_tilde_0 = inv_activation(theta_0)
    return theta_tilde_0


def training(x, y, f, theta_tilde_0, edge_index, model_name, device, hidden_dim, heads, 
             learning_rate, weight_decay, num_epochs, batch_size, sche=False, milestones=None, gamma=None):

    num_stocks = x.shape[0]
    num_times = x.shape[1]
    output_dim = f.shape[-1] + 4
    input_dim = x.shape[2] + output_dim

    model = GNN(input_dim, hidden_dim, output_dim, heads, device).to(device)
    loss_fn = LossFunction().to(device)
    approxloss_fn = ApproxLossFunction().to(device)
    sampler = BatchSampler(RandomSampler(range(num_times)), batch_size=batch_size, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if sche:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*0.5)
    x_norm = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)
    
    model.train()
    model_v = torch.vmap(model, in_dims=(1, None), out_dims=1)
    theta_tilde = model.sequential(x_norm, edge_index, theta_tilde_0)
    theta = activation(theta_tilde)
    loss, epsilon, sigma2 = loss_fn(y, f, theta)
    
    times = np.zeros(num_epochs)

    num_epochs_fast = int(num_epochs / 10)

    for epoch in range(num_epochs_fast):
        start_time = time.time()

        for b, indices in enumerate(sampler):
            theta_tilde = theta_tilde.detach()
            sigma2 = sigma2.detach()
            input = torch.concat((x_norm, theta_tilde[:, 1 :, :]), dim=-1)
            input = input[:, indices, :]
            indices_1 = [i + 1 for i in indices]
            theta_tilde[:, indices_1, :] = model_v(input, edge_index)
            theta = activation(theta_tilde)
            loss, epsilon, sigma2 = approxloss_fn(y, f, theta, sigma2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if sche:
                scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

        end_time = time.time()
        times[epoch] = end_time - start_time

    for epoch in range(num_epochs_fast, num_epochs):
        start_time = time.time()
        theta_tilde = model.sequential(x_norm, edge_index, theta_tilde_0)
        theta = activation(theta_tilde)
        loss, epsilon, sigma2 = loss_fn(y, f, theta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if sche:
            scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        end_time = time.time()
        times[epoch] = end_time - start_time

    model.eval()
    torch.save(model.state_dict(), model_name)

    times = np.mean(times)
    print(times)

    return model


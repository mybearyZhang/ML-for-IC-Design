import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import wandb


class GCN_Q(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_actions):
        super(GCN_Q, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.q1 = torch.nn.Linear(hidden_channels, 64)
        self.q2 = torch.nn.Linear(64, num_actions)

        self.i1 = torch.nn.Linear(hidden_channels, 64)
        self.i2 = torch.nn.Linear(64, num_actions)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        
        q = F.relu(self.q1(x))
        q = self.q2(q)

        i = F.relu(self.i1(x))
        i = self.i2(i)

        return q, F.log_softmax(i, dim=1), i


class discrete_BCQ(object):
    def __init__(
        self, 
        num_actions,
        device,
        BCQ_threshold=0.3,
        discount=0.99,
        optimizer="Adam",
        optimizer_parameters={},
        polyak_target_update=False,
        target_update_frequency=8e3,
        tau=0.005,
        initial_eps = 1,
        end_eps = 0.001,
        eps_decay_period = 25e4,
        eval_eps=0.001,
    ):
    
        self.device = device

        # Determine network type
        self.Q = GCN_Q(num_node_features=2, hidden_channels=16, num_actions=num_actions).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Threshold for "unlikely" actions
        self.threshold = BCQ_threshold

        # Number of training iterations
        self.iterations = 0


    def select_action(self, state, eval=False):
        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > self.eval_eps:
            with torch.no_grad():
                # state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
                state = state.to(self.device)
                q, imt, i = self.Q(state)
                imt = imt.exp()
                imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
                # Use large negative number to mask actions from argmax
                return int((imt * q + (1. - imt) * -1e8).argmax(1))
        else:
            return np.random.randint(self.num_actions)


    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            q, imt, i = self.Q(next_state)
            imt = imt.exp()
            imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

            q, imt, i = self.Q_target(next_state)
            target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate
        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action)

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt, action.reshape(-1))

        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        # Log Q values and loss to wandb
        wandb.log({"Q_value": current_Q.mean().item(), "target_Q_value": target_Q.mean().item(), "Q_loss": Q_loss.item()})

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()
    

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
           target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())


    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")


    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
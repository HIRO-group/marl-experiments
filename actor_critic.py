
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: use this file across all MARL experiment files
class QNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim):
        super(QNetwork, self).__init__()
        hidden_size = 64    # TODO: should we make this a config parameter?
        self.fc1 = nn.Linear(np.array(observation_space_shape).prod(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space_dim)

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class Actor(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim):
        super(Actor, self).__init__()
        hidden_size = 64
        self.fc1 = nn.Linear(np.array(observation_space_shape).prod(), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space_dim)

    def forward(self, x):
        # print(f">>> forward device: {device}")
        # x = torch.Tensor(x).to(device)
        # print(f">>> X device: {x.device}")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    def get_action(self, x):
        # print(f">>> get_action device: {device}")
        x = torch.Tensor(x).to(device)
        logits = self.forward(x)
        # Note that this is equivalent to what used to be called multinomial 
        # policy_dist.probs here will produce the same thing as softmax(logits)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()

        # Action probabilities for calculating the adapted loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, log_prob, action_probs
    


def one_hot_q_values(q_values):
    '''
    Convert a tensor of q_values to one-hot encoded tensors
    For example if Q(s,a) = [0.1, 0.5, 0.7] for a in A then
    this function should return [0, 0, 1]
    '''
    # print(" >>>> q_values shape: {}".format(q_values.shape))
    one_hot_values = np.zeros(q_values.shape)

    for idx in range(len(q_values)):

        # Find index of max Q(s,a)
        max_idx = torch.argmax(q_values[idx])

        # Set the value corresponding to this index to 1
        one_hot_values[idx, max_idx] = 1.0

    # Convert np array to tensor before returning
    return torch.from_numpy(one_hot_values)    
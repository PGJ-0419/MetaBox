import collections
import torch
import random
import numpy as np
import pickle 
import os
import math

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        # print(type(obs_batch),type(action_batch),type(reward_batch),type(next_obs_batch),type(done_batch))
        # print(type(action_batch[0]))
        # obs_batch = torch.FloatTensor(np.array(obs_batch))
        obs_batch = torch.stack(obs_batch)
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.Tensor(reward_batch)

        # 兼容操作，满足MOO和SOO等需求
        if isinstance(next_obs_batch, (list, np.ndarray)):
            next_obs_batch = torch.Tensor(np.array(next_obs_batch))
        else:
            next_obs_batch = torch.stack(next_obs_batch)

        # next_obs_batch = torch.FloatTensor(np.array(next_obs_batch))
        done_batch = torch.Tensor(done_batch)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer_torch:
    def __init__(self, capacity, state_dim, device):

        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0  

    
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device, dtype=torch.long)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.dones = torch.zeros(capacity, dtype=torch.long, device=device)

    def append(self, state, action, reward, next_state, done):

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = int(reward)
        self.next_states[self.position] = next_state
        self.dones[self.position] = int(done)


        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

# MOO特有,这个放在这里是需要的吗？
class MultiAgent_ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, transition):
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)

        n_agents, obs_size = len(s_lst[0]), len(s_lst[0][0])
        return torch.tensor(s_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(a_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents), \
               torch.tensor(r_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents), \
               torch.tensor(s_prime_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(done_lst, dtype=torch.float).view(batch_size, chunk_size, 1)

    def __len__(self):
        return len(self.buffer)

def clip_grad_norms(param_groups, max_norm = math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type = 2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def save_class(dir, file_name, saving_class):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(saving_class, f, -1)

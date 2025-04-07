from typing import Tuple
from agent.basic_agent import Basic_Agent
import torch
import math, copy
from typing import Any, Callable, List, Optional, Tuple, Union, Literal

from torch import nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from agent.utils import *
from VectorEnv.great_para_env import ParallelEnv

class Actor(nn.Module):
    def __init__(self, n_state, n_action, hidden_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.mu = nn.Linear(hidden_dim, n_action)
        self.sigma = nn.Linear(hidden_dim, n_action)
        self.tanh = nn.Tanh()
        self.__max_sigma = 0.05
        self.__min_sigma = 0.15

    def forward(self, state, fixed_action = None):
        x = self.tanh(self.linear1(state))
        x = self.tanh(self.linear2(x))
        mu = torch.tanh(self.mu(x) + 1.) / 2.
        sigma = torch.tanh(self.sigma(x) + 1.) / 2. * (self.__max_sigma - self.__min_sigma) + self.__min_sigma
        distribution = torch.distributions.Normal(mu, sigma)

        if fixed_action is None:
            action = distribution.sample()
           
        else:
            action = fixed_action

        log_probs = distribution.log_prob(action)
        log_probs = torch.sum(log_probs, dim=-1)
        action = torch.clamp(action, min=0, max=1)

        return action, log_probs

class Critic(nn.Module):
    def __init__(self, n_state, hidden_dim=64):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.tanh(self.linear1(x))
        out = self.tanh(self.linear2(out))
        baseline_value = self.linear3(out)

        return baseline_value.detach(), baseline_value

# memory for recording transition during training process
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


def clip_grad_norms(param_groups, max_norm=math.inf):
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
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class L2O_Agent_Parallel(Basic_Agent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = 0.99
        self.n_step = 10
        self.K_epochs = 3
        self.eps_clip = 0.2
        self.max_grad_norm = 0.1
        self.device = self.config.device
        self.n_state = self.config.task_cnt * 7 + 1
        self.n_action = self.config.task_cnt * 3
        
        # figure out the actor network
        # self.actor = None
        self.actor = Actor(self.n_state, self.n_action)
        
        # figure out the critic network
        # self.critic = None
        self.critic = Critic(self.n_state)
        assert hasattr(self, 'actor') and hasattr(self, 'critic')

        # figure out the optimizer
        # assert hasattr(torch.optim, self.config.optimizer)
        # self.optimizer = eval('torch.optim.' + self.config.optimizer)(
        #     [{'params': self.actor.parameters(), 'lr': self.config.lr_actor}] +
        #     [{'params': self.critic.parameters(), 'lr': self.config.lr_critic}])
        self.optimizer = torch.optim.Adam(
            [{'params': self.actor.parameters(), 'lr': 1e-5}] +
            [{'params': self.critic.parameters(), 'lr': 1e-5}])
        
        # figure out the lr schedule
        # assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
        # self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer, self.config.lr_decay, last_epoch=-1,)

        # move to device
        self.actor.to(self.device)
        self.critic.to(self.device)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir,'checkpoint'+str(self.cur_checkpoint),self)
        self.cur_checkpoint += 1

    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1

    def train_episode(self, 
                      envs, 
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                      num_cpus: Optional[Union[int, None]]=1,
                      num_gpus: int=0,
                      required_info={}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        env.seed(seeds)
        memory = Memory()

        # params for training
        gamma = self.gamma
        n_step = self.n_step
        
        K_epochs = self.K_epochs
        eps_clip = self.eps_clip
        
        state = env.reset()

        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass
        
        t = 0
        # initial_cost = obj
        _R = torch.zeros(len(env))
        _loss = []
        # sample trajectory
        while not env.all_done():
            t_s = t
            total_cost = 0
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step :  
                
                memory.states.append(state.clone())
                action, log_lh = self.actor(state)
                memory.actions.append(action.clone() if isinstance(action, torch.Tensor) else copy.deepcopy(action))
                memory.logprobs.append(log_lh)

                baseline_val, baseline_val_detached = self.critic(state)
                
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # state transient
                state, rewards, is_end, info = env.step(action)
                memory.rewards.append(torch.FloatTensor(rewards).to(self.device))
                # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
                _R += rewards
                # store info

                # next
                t = t + 1

                try:
                    state = torch.FloatTensor(state).to(self.device)
                except:
                    pass
                
                if np.all(is_end):
                    break
            
            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin update
            old_actions = torch.stack(memory.actions)
            try:
                old_states = torch.stack(memory.states).detach() #.view(t_time, bs, ps, dim_f)
            except:
                pass
            # old_actions = all_actions.view(t_time, bs, ps, -1)
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
            # Optimize PPO policy for K mini-epochs:
            old_value = None
            for _k in range(K_epochs):
                if _k == 0:
                    logprobs = memory.logprobs

                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):

                        # get new action_prob
                        _, log_p = self.actor(old_states[tt], fixed_action = old_actions[tt])

                        logprobs.append(log_p)

                        baseline_val, baseline_val_detached = self.critic(old_states[tt])
                        
                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)

                logprobs = torch.stack(logprobs).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)


                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.critic(state)[0].squeeze(1)

                for r in range(len(reward_reversed)):
                    R = R * gamma + reward_reversed[r]
                    Reward.append(R)
                # clip the target:
            
                Reward = torch.stack(Reward[::-1], 0)

                Reward = Reward.view(-1)
                
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
        
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()

                # check K-L divergence (for logging only)
                approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
                approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
                # calculate loss
                loss = baseline_loss + reinforce_loss

                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()

                _loss.append(loss.item())
                # Clip gradient norm and get (clipped) gradient norms for logging
                # current_step = int(pre_step + t//n_step * K_epochs  + _k)
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.max_grad_norm)

                # perform gradient descent
                self.optimizer.step()
                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                    save_class(self.config.agent_save_dir, 'checkpoint'+str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if self.learning_time >= self.config.max_learning_step:
                    memory.clear_memory()
                    return_info = {'return': _R, 'learn_steps': self.learning_time, 'loss':np.mean(_loss)}
                    # for key in required_info.keys():
                    #     return_info[key] = env.get_env_attr(required_info[key])
                    
                    gbest_list = []
                    optimizer_list = env.get_env_attr('optimizer')
                    for _ in range(len(optimizer_list)):
                        gbest_list.append(optimizer_list[_].gbest)
                    
                    return_info['gbest'] = gbest_list

                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

            memory.clear_memory()
        
        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R, 'learn_steps': self.learning_time, 'loss':np.mean(_loss)}
        # for key in required_info.keys():
        #     return_info[key] = env.get_env_attr(required_info[key])

        gbest_list = []
        optimizer_list = env.get_env_attr('optimizer')
        for _ in range(len(optimizer_list)):
            gbest_list.append(optimizer_list[_].gbest)
                    
        return_info['gbest'] = gbest_list
     
        env.close()
        
        return is_train_ended, return_info
    
    def rollout_episode(self, 
                        env,
                        seed=None,
                        required_info={}):
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.FloatTensor(state).to(self.device)
                except:
                    state = [state]
                action = self.actor(state)[0]
                action = action.cpu().numpy()
                state, reward, is_done = env.step(action)
                R += reward
            results = {'return': R}
            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results
    
    def rollout_batch_episode(self, 
                              envs, 
                              seeds=None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                              asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                              num_cpus: Optional[Union[int, None]]=1,
                              num_gpus: int=0,
                              required_info={}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        if seeds is not None:
            env.seed(seeds)
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass
        
        R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action = self.actor(state)[0]
            
            # state transient
            state, rewards, is_end = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards)
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
        results = {'return': R}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results


from typing import Tuple
from ...rl.ppo import *

import torch
import math, copy
from typing import Any, Callable, List, Optional, Tuple, Union, Literal

from torch import nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from ...rl.utils import *
from ...environment.parallelenv.parallelenv import ParallelEnv

class Actor(nn.Module):
    def __init__(self, n_state, n_action, hidden_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,hidden_dim)
        self.mu = nn.Linear(hidden_dim, n_action)
        self.sigma = nn.Linear(hidden_dim, n_action)
        self.tanh = nn.Tanh()
        self.__max_sigma = 0.15
        self.__min_sigma = 0.05

    def forward(self, state, fixed_action = None):
        x = self.tanh(self.linear1(state))
        x = self.tanh(self.linear2(x))
        mu = (torch.tanh(self.mu(x)) + 1.0) / 2.
        sigma = (torch.tanh(self.sigma(x)) + 1.0) / 2. * (self.__max_sigma - self.__min_sigma) + self.__min_sigma
        distribution = torch.distributions.Normal(mu, sigma)

        if fixed_action is None:
            action = distribution.sample()
           
        else:
            action = fixed_action

        log_probs = distribution.log_prob(action)
        log_probs = torch.sum(log_probs, dim=-1)
        action = torch.clamp(action, min=0, max=1)
        entropy = distribution.entropy()  # for logging only

        return action, log_probs, entropy

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


class L2T(PPO_Agent):
    def __init__(self, config):
        self.config = config

        self.config.optimizer = 'Adam'
        self.config.lr_actor = 1e-4
        self.config.lr_critic = 1e-4
        self.config.lr_scheduler = 'ExponentialLR'

        # define parameters
        self.config.gamma = 0.99
        self.config.n_step = 10
        self.config.K_epochs = 3
        self.config.eps_clip = 0.2
        self.config.max_grad_norm = 0.1
        self.config.device = self.config.device
        self.config.n_state = self.config.task_cnt * 7 + 1
        self.config.n_action = self.config.task_cnt * 3
        
        # figure out the actor network
        # self.actor = None
        actor = Actor(self.config.n_state, self.config.n_action)
        
        # figure out the critic network
        # self.critic = None
        critic = Critic(self.config.n_state)

        super().__init__(self.config, {'actor': actor, 'critic': critic}, [self.config.lr_actor, self.config.lr_critic])

    def __str__(self):
        return "L2T"

    def train_episode(self, 
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                      # asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                      # num_cpus: Optional[Union[int, None]]=1,
                      # num_gpus: int=0,
                      compute_resource = {},
                      tb_logger = None,
                      required_info={}):
        num_cpus = None
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)
        env.seed(seeds)
        memory = Memory()

        # params for training
        gamma = self.gamma
        n_step = self.n_step
        
        K_epochs = self.K_epochs
        eps_clip = self.eps_clip
        
        state = env.reset()

        try:
            state = torch.Tensor(state).to(self.device)
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
            entropy = []
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step :  
                
                memory.states.append(state.clone())
                action, log_lh, entro_p = self.actor(state)
                memory.actions.append(action.clone() if isinstance(action, torch.Tensor) else copy.deepcopy(action))
                memory.logprobs.append(log_lh)

                entropy.append(entro_p.detach().cpu())

                baseline_val, baseline_val_detached = self.critic(state)
                
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # state transient
                state, rewards, is_end, info = env.step(action)
                memory.rewards.append(torch.Tensor(rewards).to(self.device))
                # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
                _R += rewards
                # store info

                # next
                t = t + 1

                try:
                    state = torch.Tensor(state).to(self.device)
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
                    entropy = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):

                        # get new action_prob
                        _, log_p, entro_p = self.actor(old_states[tt], fixed_action = old_actions[tt])

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val, baseline_val_detached = self.critic(old_states[tt])
                        
                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)

                logprobs = torch.stack(logprobs).view(-1)
                entropy = torch.stack(entropy).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)


                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.critic(state)[0].squeeze(1)
                critic_output = R.clone()
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
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                    save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if not self.config.no_tb and self.learning_time % int(self.config.log_step) == 0:
                    self.log_to_tb_train(tb_logger, self.learning_time,
                                         grad_norms,
                                         reinforce_loss, baseline_loss,
                                         _R, Reward, memory.rewards,
                                         critic_output, logprobs, entropy, approx_kl_divergence)

                if self.learning_time >= self.config.max_learning_step:
                    memory.clear_memory()
                    return_info = {'return': _R, 'learn_steps': self.learning_time, 'loss':_loss}
                    return_info['gbest'] = env.get_env_attr('gbest')
                    for key in required_info.keys():
                        return_info[key] = env.get_env_attr(required_info[key])
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

            memory.clear_memory()
        
        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R, 'learn_steps': self.learning_time, 'loss':_loss}
        return_info['gbest'] = env.get_env_attr('gbest')
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
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
                    state = torch.Tensor(state).to(self.device)
                except:
                    state = [state]
                action = self.actor(state)[0]
                action = action.cpu().numpy()
                state, reward, is_done, info = env.step(action)
                R += reward
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            results = {'cost': env_cost, 'fes': env_fes, 'return': R}

            if self.config.full_meta_data:
                meta_X = env.get_env_attr('meta_X')
                meta_Cost = env.get_env_attr('meta_Cost')
                metadata = {'X': meta_X, 'Cost': meta_Cost}
                results['metadata'] = metadata
                
            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results
    
    def rollout_batch_episode(self, 
                              envs, 
                              seeds=None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                              # asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                              # num_cpus: Optional[Union[int, None]]=1,
                              # num_gpus: int=0,
                              compute_resource = {},
                              required_info={}):
        num_cpus = None
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)

        env.seed(seeds)
        state = env.reset()
        try:
            state = torch.Tensor(state).to(self.device)
        except:
            pass
        
        _R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action = self.actor(state)[0]
            
            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            _R += torch.Tensor(rewards)
            # store info
            try:
                state = torch.Tensor(state).to(self.device)
            except:
                pass

        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        results = {'cost': env_cost, 'fes': env_fes, 'return': _R}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results


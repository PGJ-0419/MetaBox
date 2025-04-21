"""
This file is used to train the agent.(for the kind of optimizer that is learnable)
"""
import pickle
import time
import torch
from tqdm import tqdm
from .environment.basic_environment import PBO_Env
from .environment.parallelenv import *
from .logger import *
import copy
from .environment.problem.utils import *
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from .rl.utils import save_class
from tensorboardX import SummaryWriter

from .environment.optimizer import (
    DEDDQN_Optimizer,
    DEDQN_Optimizer,
    RLHPSDE_Optimizer,
    LDE_Optimizer,
    QLPSO_Optimizer,
    RLEPSO_Optimizer,
    RLPSO_Optimizer,
    L2L_Optimizer,
    GLEET_Optimizer,
    RLDAS_Optimizer,
    LES_Optimizer,
    NRLPSO_Optimizer,
    SYMBOL_Optimizer,
    RLDEAFL_Optimizer,
    SurrRLDE_Optimizer,
    RLEMMO_Optimizer,
    madac_optimizer,
    GLHF_Optimizer,
    B2OPT_Optimizer,
    LGA_Optimizer,
    PSORLNS_Optimizer,
)

from .baseline.bbo import (
    DE,
    JDE21,
    MADDE,
    NLSHADELBC,
    PSO,
    GLPSO,
    SDMSPSO,
    SAHLPSO,
    CMAES,
    Random_search,
)

from .baseline.metabbo import (
    GLEET,
    DEDDQN,
    DEDQN,
    QLPSO,
    NRLPSO,
    RLHPSDE,
    RLDEAFL,
    SYMBOL,
    RLDAS,
    SurrRLDE,
    RLEMMO,
    madac,

    GLHF,
    B2OPT,
    LGA,
    LDE,
    RLEPSO,
    RLPSO,
    LES,
    PSORLNS
)


matplotlib.use('Agg')


class Trainer(object):
    def __init__(self, config, user_agent = None, user_optimizer = None, user_datasets = None):
        self.config = config

        if self.config.train_problem in ['bbob-surrogate-10D','bbob-surrogate-5D','bbob-surrogate-2D']:
            self.config.is_train = True

        if user_datasets is None:
            self.train_set, self.test_set = construct_problem_set(config)
        else:
            self.train_set, self.test_set = user_datasets

        self.config.dim = max(self.train_set.maxdim, self.test_set.maxdim)

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if user_agent is None:
            if self.config.resume_dir is None:
                self.agent = eval(self.config.train_agent)(self.config)
            else:
                file_path = self.config.resume_dir + self.config.train_agent + '.pkl'
                with open(file_path, 'rb') as f:
                    self.agent = pickle.load(f)
                self.agent.update_setting(self.config)
        else:
            self.agent = user_agent

        if user_optimizer is None:
            self.optimizer = eval(self.config.train_optimizer)(self.config)
        else:
            self.optimizer = user_optimizer

    def save_log(self, epochs, steps, cost, returns, normalizer):
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return_save = np.stack((steps, returns),  0)
        np.save(log_dir+'return', return_save)
        for problem in self.train_set:
            name = problem.__str__()
            if len(cost[name]) == 0:
                continue
            while len(cost[name]) < len(epochs):
                cost[name].append(cost[name][-1])
                normalizer[name].append(normalizer[name][-1])
            cost_save = np.stack((epochs, cost[name], normalizer[name]),  0)
            np.save(log_dir+name+'_cost', cost_save)
            
    def save_class(dir, file_name, saving_class):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir+file_name+'.pkl', 'wb') as f:
            pickle.dump(saving_class, f, -1)

    def train(self):
        print(f'start training: {self.config.run_time}')
        is_end = False
        # todo tensorboard
        tb_logger = None
        start_time = time.time()
        if not self.config.no_tb:
            if not os.path.exists(os.path.join('output/tensorboard', self.config.run_time)):
                os.makedirs(os.path.join('output/tensorboard', self.config.run_time))
            tb_logger = SummaryWriter(os.path.join('output/tensorboard', self.config.run_time))
            tb_logger.add_scalar("epoch-step", 0, 0)
        train_log = {'loss': [], 'learn_steps': [], 'return': [], 'runtime': [], 'config': copy.deepcopy(self.config)}
        if not os.path.exists(os.path.join('output/train_log', self.config.run_time)):
            os.makedirs(os.path.join('output/train_log', self.config.run_time))
        epoch = 0
        bs = self.config.train_batch_size
        if self.config.train_mode == "single":
            self.train_set.batch_size = 1
            self.train_set.ptr = [i for i in range(0, self.train_set.N)]
        elif self.config.train_mode == "multi":
            self.train_set.batch_size = bs

        epoch_seed = self.config.epoch_seed
        id_seed = self.config.id_seed
        seed = self.config.seed

        checkpoint_time0 = time.time()
        while not is_end:
            learn_step = 0
            self.train_set.shuffle()
            return_record = []
            loss_record = []
            with tqdm(range(int(np.ceil(self.train_set.N / self.train_set.batch_size))), desc = f'Training {self.agent.__class__.__name__} Epoch {epoch}') as pbar:
                for problem_id, problem in enumerate(self.train_set):
                    # set seed
                    seed_list = (epoch * epoch_seed + id_seed * (np.arange(bs) + bs * problem_id) + seed).tolist()

                    # 这里前面已经判断好 train_mode，这里只需要根据 train_mode 构造env就行
                    if self.config.train_mode == "single":
                        env_list = [PBO_Env(copy.deepcopy(problem[0]), copy.deepcopy(self.optimizer)) for _ in range(bs)] # bs
                    elif self.config.train_mode == "multi":
                        env_list = [PBO_Env(copy.deepcopy(p), copy.deepcopy(self.optimizer)) for p in problem] # bs

                    # todo config add para
                    exceed_max_ls, train_meta_data = self.agent.train_episode(envs = env_list,
                                                                              seeds = seed_list,
                                                                              tb_logger = tb_logger,
                                                                              para_mode = self.config.train_parallel_mode,
                                                                              )
                    # train_meta_data {'return': list[], 'loss': list[], 'learn_steps': int}

                    # exceed_max_ls, pbar_info_train = self.agent.train_episode(env)  # pbar_info -> dict
                    postfix_str = (
                        f"loss={np.mean(train_meta_data['loss']):.2e}, "
                        f"learn_steps={train_meta_data['learn_steps']}, "
                        f"return={np.mean(train_meta_data['return']):.2e}"
                    )
                    train_log['loss'].append(train_meta_data['loss'])
                    train_log['learn_steps'].append(train_meta_data['learn_steps'])
                    train_log['return'].append(train_meta_data['return'])
                    train_log['runtime'].append(time.time() - start_time)

                    with open(os.path.join('output/train_log', self.config.run_time, 'train_log.pkl'), 'wb') as f:
                        pickle.dump(train_log, f)
                    
                    pbar.set_postfix_str(postfix_str)
                    pbar.update(self.train_set.batch_size)
                    learn_step = train_meta_data['learn_steps']
                    
                    return_record.append(np.mean(train_meta_data['return']))
                    loss_record.append(np.mean(train_meta_data['loss']))
                    
                    # for id, p in enumerate(problem):
                    #     name = p.__str__()
                    #     cost_record[name].append(train_meta_data['gbest'][id])
                    #     normalizer_record[name].append(train_meta_data['normalizer'][id])
                    #     return_record.append(np.mean(train_meta_data['return']))
                    # learn_steps.append(learn_step)
                    # if learn_step >= (self.config.save_interval * self.agent.cur_checkpoint) and self.config.end_mode == "step":
                    #     save_class(self.config.agent_save_dir, 'checkpoint' + str(self.agent.cur_checkpoint), self.agent)
                    #     # 记录 checkpoint 和 total_step
                    #     with open(self.config.agent_save_dir + "/checkpoint_log.txt", "a") as f:
                    #         f.write(f"Checkpoint {self.agent.cur_checkpoint}: {learn_step}\n")

                    if self.config.end_mode == "step" and exceed_max_ls:
                        is_end = True
                        break
                # self.agent.train_epoch()
            # epoch_steps.append(learn_step)
            checkpoint_time_epoch = time.time() - checkpoint_time0
            epoch += 1

            if not self.config.no_tb:
                tb_logger.add_scalar("epoch-step", learn_step, epoch)
                tb_logger.add_scalar("epoch-avg-return", np.mean(return_record), epoch)
                tb_logger.add_scalar("epoch-avg-loss", np.mean(loss_record), epoch)

            if epoch >= (self.config.save_interval * self.agent.cur_checkpoint) and self.config.end_mode == "epoch":
                save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.agent.cur_checkpoint), self.agent)
                # 记录 checkpoint 和 total_step
                with open(self.config.agent_save_dir + "/checkpoint_log.txt", "a") as f:
                    f.write(f"Checkpoint {self.agent.cur_checkpoint}: {learn_step}; Time: {checkpoint_time_epoch} s\n")

                # todo rollout
                # 保存状态
                # cpu_state = torch.random.get_rng_state()
                # cuda_state = torch.cuda.get_rng_state()
                # np_state = np.random.get_state()
                # self.rollout(self.agent.cur_checkpoint)
                # 载入
                # torch.random.set_rng_state(cpu_state)
                # torch.cuda.set_rng_state(cuda_state)
                # np.random.set_state(np_state)

                self.agent.cur_checkpoint += 1
            if self.config.end_mode == "epoch" and epoch >= self.config.max_epoch:
                is_end = True

    def rollout(self, checkpoint, rollout_run = 10):
        # 读取 agent
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        with open(self.config.agent_save_dir + 'checkpoint' + str(checkpoint) + ".pkl", "rb") as f:
            agent = pickle.load(f)
        rollout_set = copy.deepcopy(self.test_set)
        bs = self.config.train_batch_size

        seed_list = list(range(1, rollout_run + 1)) * bs
        pbar = (rollout_set.N // bs + rollout_set.N % bs)
        with tqdm(range(pbar), desc = f"Rollout{checkpoint}") as pbar:
            for i, problem in enumerate(rollout_set):
                env_list = [PBO_Env(copy.deepcopy(p), copy.deepcopy(self.optimizer))
                            for p in problem
                            for _ in range(rollout_run)]
                with torch.no_grad():
                    meta_rollout_data = agent.rollout_batch_episode(envs = env_list,
                                                                    seeds = seed_list,
                                                                    para_mode = 'dummy',
                                                                    asynchronous = None,
                                                                    num_cpus = 1,
                                                                    num_gpus = 0,
                                                                    )
                pbar.set_postfix({'MetaBBO': agent.__str__()})
                pbar.update(1)
            
        
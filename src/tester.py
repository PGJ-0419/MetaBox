import copy
from utils import construct_problem_set
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
from tqdm import tqdm
import os
from environment.basic_environment import PBO_Env, BBO_Env, MetaBBO_Env
from logger import Logger
from VectorEnv.great_para_env import ParallelEnv
import json
import torch

from agent import (
    # DE_DDQN_Agent,
    # DEDQN_Agent,
    RL_HPSDE_Agent,
    LDE_Agent,
    # QLPSO_Agent,
    RLEPSO_Agent,
    RL_PSO_Agent,
    L2L_Agent,
    # RL_DAS_Agent,
    LES_Agent,
    # NRLPSO_Agent,
    # Symbol_Agent,
)
from optimizer import (
    DE_DDQN_Optimizer,
    DEDQN_Optimizer,
    RL_HPSDE_Optimizer,
    LDE_Optimizer,
    QLPSO_Optimizer,
    RLEPSO_Optimizer,
    RL_PSO_Optimizer,
    L2L_Optimizer,
    GLEET_Optimizer,
    RL_DAS_Optimizer,
    LES_Optimizer,
    NRLPSO_Optimizer,
    SYMBOL_Optimizer,
    RLDE_AFL_Optimizer,
    Surr_RLDE_Optimizer,
    RLEMMO_Optimizer,

    DEAP_DE,
    JDE21,
    MadDE,
    NL_SHADE_LBC,

    DEAP_PSO,
    GL_PSO,
    sDMS_PSO,
    SAHLPSO,

    DEAP_CMAES,
    Random_search,
    BayesianOptimizer
)

from agents import (
    GLEET_Agent,
    DE_DDQN_Agent,
    DEDQN_Agent,
    QLPSO_Agent,
    NRLPSO_Agent,
    RL_HPSDE_Agent,
    RLDE_AFL_Agent,
    SYMBOL_Agent,
    RL_DAS_Agent,
    Surr_RLDE_Agent,
    RLEMMO_Agent
)

from VectorEnv.great_para_env import ParallelEnv

from agents import (
    GLEET_Agent,
    DE_DDQN_Agent,
    DEDQN_Agent,
    QLPSO_Agent,
    NRLPSO_Agent,
    RL_HPSDE_Agent,
    RLDE_AFL_Agent,
    SYMBOL_Agent,
    RL_DAS_Agent,
    Surr_RLDE_Agent
)

def cal_t0(dim, fes):
    T0 = 0
    for i in range(10):
        start = time.perf_counter()
        for _ in range(fes):
            x = np.random.rand(dim)
            x + x
            x / (x+2)
            x * x
            np.sqrt(x)
            np.log(x)
            np.exp(x)
        end = time.perf_counter()
        T0 += (end - start) * 1000
    # ms
    return T0/10


def cal_t1(problem, dim, fes):
    T1 = 0
    for i in range(10):
        x = np.random.rand(fes, dim)
        start = time.perf_counter()
        # for i in range(fes):
        #     problem.eval(x[i])
        problem.eval(x)
        end = time.perf_counter()
        T1 += (end - start) * 1000
    # ms
    return T1/10


class Tester(object):
    def __init__(self, config):
        self.key_list = config.agent
        self.log_dir = config.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.config = config

        if self.config.problem[-6:]=='-torch':
            self.config.problem=self.config.problem[:-6]

        if config.problem =='bbob-surrogate':
            config.is_train = False

        _, self.test_set = construct_problem_set(self.config)
        # if 'L2L_Agent' in config.agent_for_cp or 'L2L_Agent' == config.agent:
        #     pre_problem=config.problem
        #     config.problem=pre_problem+'-torch'
        #     _,self.torch_test_set = construct_problem_set(config)
        #     config.problem=pre_problem
        
        self.seed = range(51)
        # initialize the dataframe for logging
        self.test_results = {'cost': {},
                             'fes': {},
                             'T0': 0.,
                             'T1': {},
                             'T2': {},
                             'pr': {},
                             'sr': {}}

        # prepare experimental optimizers and agents
        self.agent_for_cp = []
        self.agent_name_list = []
        self.l_optimizer_for_cp = []
        self.t_optimizer_for_cp = []

        with open('model.json', 'r', encoding = 'utf-8') as f:
            json_data = json.load(f)
        for key in self.key_list:
            if key not in json_data.keys():
                raise KeyError(f"Missing key '{key}' in model.json")

            # get key
            baseline = json_data[key]
            if "Agent" in baseline.keys():
                agent_name = baseline["Agent"]
                l_optimizer = baseline['Optimizer']
                dir = baseline['dir']
                # get agent
                self.agent_name_list.append(key)
                with open(dir, 'rb') as f:
                    self.agent_for_cp.append(pickle.load(f))
                self.l_optimizer_for_cp.append(eval(l_optimizer)(copy.deepcopy(config)))

            else:
                t_optimizer = baseline['Optimizer']
                self.t_optimizer_for_cp.append(eval(t_optimizer)(copy.deepcopy(config)))

        for optimizer in config.t_optimizer:
            self.t_optimizer_for_cp.append(eval(optimizer)(copy.deepcopy(config)))
        # logging
        if len(self.agent_for_cp) == 0:
            print('None of learnable agent')
        else:
            print(f'there are {len(self.agent_for_cp)} agent')
            for a, l_optimizer in zip(self.agent_name_list, self.l_optimizer_for_cp):
                print(f'learnable_agent:{a},l_optimizer:{type(l_optimizer).__name__}')

        if len(self.t_optimizer_for_cp) == 0:
            print('None of traditional optimizer')
        else:
            print(f'there are {len(self.t_optimizer_for_cp)} traditional optimizer')
            for t_optmizer in self.t_optimizer_for_cp:
                print(f't_optimizer:{type(t_optmizer).__name__}')

        for agent_name in self.agent_name_list:
            self.test_results['T1'][agent_name] = 0.
            self.test_results['T2'][agent_name] = 0.
        for optimizer in self.t_optimizer_for_cp:
            self.test_results['T1'][type(optimizer).__name__] = 0.
            self.test_results['T2'][type(optimizer).__name__] = 0.

        for problem in self.test_set.data:
            self.test_results['cost'][problem.__str__()] = {}
            self.test_results['fes'][problem.__str__()] = {}
            for agent_name in self.agent_name_list:
                self.test_results['cost'][problem.__str__()][agent_name] = []  # 51 np.arrays
                self.test_results['fes'][problem.__str__()][agent_name] = []  # 51 scalars
            for optimizer in self.t_optimizer_for_cp:
                self.test_results['cost'][problem.__str__()][type(optimizer).__name__] = []  # 51 np.arrays
                self.test_results['fes'][problem.__str__()][type(optimizer).__name__] = []  # 51 scalars

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def test(self):
        # todo 测试并行方式有多种 得考虑下

        print(f'start testing: {self.config.run_time}')
        # calculate T0
        T0 = cal_t0(self.config.dim, self.config.maxFEs)
        self.test_results['T0'] = T0
        # calculate T1
        # T1 = cal_t1(self.test_set[0], self.config.dim, self.config.maxFEs)
        # self.test_results['T1'] = T1
        pbar_len = (len(self.t_optimizer_for_cp) + len(self.agent_for_cp)) * self.test_set.N * 51
        with tqdm(range(pbar_len), desc='Testing') as pbar:
            for i,problem in enumerate(self.test_set):

                # run learnable optimizer
                for agent_id,(agent,optimizer) in enumerate(zip(self.agent_for_cp,self.l_optimizer_for_cp)):
                    T1 = 0
                    T2 = 0
                    for run in range(51):
                        env_list = [PBO_Env(p, copy.deepcopy(optimizer)) for p in problem]

                        start = time.perf_counter()
                        info = agent.rollout_batch_episode(envs = env_list,
                                                           seeds = self.seed[run],
                                                           para_mode = 'dummy',
                                                           asynchronous = None,
                                                           num_cpus = 1,
                                                           num_gpus = 0,
                        )
                        # np.random.seed(self.seed[run])
                        # problem.reset() 这里env_list reset有了
                        # construct an ENV for (problem,optimizer)
                        # env = PBO_Env(problem,optimizer)

                        # info = agent.rollout_episode(env)
                        # cost = info['cost']
                        # while len(cost) < 51:
                        #     cost.append(cost[-1])
                        # fes = info['fes']
                        # end = time.perf_counter()
                        # if i == 0:
                        #     T2 += (end - start) * 1000  # ms
                        #     T1 += env.problem.T1
                        # self.test_results['cost'][problem.__str__()][self.agent_name_list[agent_id]].append(cost)
                        # self.test_results['fes'][problem.__str__()][self.agent_name_list[agent_id]].append(fes)
                        pbar_info = {'agent': agent.__str__(),
                                     'run': run,
                                     }

                        # pbar_info = {'problem': problem.__str__(),
                        #              'optimizer': self.agent_name_list[agent_id],
                        #              'run': run,
                        #              'cost': cost[-1],
                        #              'fes': fes}
                        pbar.set_postfix(pbar_info)
                        pbar.update(len(env_list))
                    # if i == 0:
                    #     self.test_results['T1'][self.agent_name_list[agent_id]] = T1/51
                    #     self.test_results['T2'][self.agent_name_list[agent_id]] = T2/51
                    #     if type(agent).__name__ == 'L2L_Agent':
                    #         self.test_results['T1'][self.agent_name_list[agent_id]] *= self.config.maxFEs/100
                    #         self.test_results['T2'][self.agent_name_list[agent_id]] *= self.config.maxFEs/100
                # run traditional optimizer
        #         for optimizer in self.t_optimizer_for_cp:
        #             T1 = 0
        #             T2 = 0
        #             for run in range(51):
        #                 start = time.perf_counter()
        #                 np.random.seed(self.seed[run])
        #
        #                 problem.reset()
        #                 info = optimizer.run_episode(problem)
        #                 cost = info['cost']
        #                 while len(cost) < 51:
        #                     cost.append(cost[-1])
        #                 fes = info['fes']
        #                 end = time.perf_counter()
        #                 if i == 0:
        #                     T1 += problem.T1
        #                     T2 += (end - start) * 1000  # ms
        #                 self.test_results['cost'][problem.__str__()][type(optimizer).__name__].append(cost)
        #                 self.test_results['fes'][problem.__str__()][type(optimizer).__name__].append(fes)
        #                 pbar_info = {'problem': problem.__str__(),
        #                              'optimizer': type(optimizer).__name__,
        #                              'run': run,
        #                              'cost': cost[-1],
        #                              'fes': fes, }
        #                 pbar.set_postfix(pbar_info)
        #                 pbar.update(1)
        #             if i == 0:
        #                 self.test_results['T1'][type(optimizer).__name__] = T1/51
        #                 self.test_results['T2'][type(optimizer).__name__] = T2/51
        #                 if type(optimizer).__name__ == 'BayesianOptimizer':
        #                     self.test_results['T1'][type(optimizer).__name__] *= (self.config.maxFEs/self.config.bo_maxFEs)
        #                     self.test_results['T2'][type(optimizer).__name__] *= (self.config.maxFEs/self.config.bo_maxFEs)
        # with open(self.log_dir + 'test.pkl', 'wb') as f:
        #     pickle.dump(self.test_results, f, -1)
        # random_search_results = test_for_random_search(self.config)
        # with open(self.log_dir + 'random_search_baseline.pkl', 'wb') as f:
        #     pickle.dump(random_search_results, f, -1)
    def test_1(self):
        # todo 第一种 并行是 agent for 循环
        # todo 每个 agent 做一个问题 x test_run 的列表环境
        print(f'start testing: {self.config.run_time}')

        test_run = self.config.test_run
        seed_list = list(range(1, test_run + 1))
        pbar_len = (len(self.agent_for_cp) + len(self.t_optimizer_for_cp)) * self.test_set.N
        with tqdm(range(pbar_len), desc = "Testing") as pbar:
            for i, problem in enumerate(self.test_set.data):
                for agent_id, (agent, optimizer) in enumerate(zip(self.agent_for_cp, self.l_optimizer_for_cp)):
                    # for agent an env_list [1 * len(test_run)]
                    '''
                        example: bs = 3 test_run = 2 agent A1 A2
                        [        A1        |        A2        |        A1        |        A2        |        A1        |        A2        ]
                        [O1_F1_r1--O1_F1_r2|O2_F1_r1--O2_F1_r2|O1_F2_r1--O1_F2_r2|O2_F2_r1--O2_F2_r2|O2_F3_r1--O2_F3_r2|O2_F3_r1--O2_F3_r2]
                        PS：Test results are not affected by bs 
                    '''
                    env_list = [PBO_Env(copy.deepcopy(problem), copy.deepcopy(optimizer)) for _ in range(test_run)]
                    meta_test_data = agent.rollout_batch_episode(envs = env_list,
                                                                 seeds = seed_list,
                                                                 para_mode = 'dummy',
                                                                 asynchronous = None,
                                                                 num_cpus = 1,
                                                                 num_gpus = 0,
                                                                 )
                    # meta_test_data : {cost, fes, return}
                    pbar_info = {'MetaBBO': agent.__str__(),
                                 'problem': problem.__str__(),
                                 }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
                # run traditional optimizer
                for optimizer in self.t_optimizer_for_cp:
                    # env_list = [BBO_Env(copy.deepcopy(optimizer)) for _ in range(test_run)]
                    # action_list = [copy.deepcopy(problem) for _ in range(test_run)]
                    # env = ParallelEnv(env_list, para_mode = 'dummy', asynchronous = None, num_cpus = 1, num_gpus = 0)

                    env_list = [BBO_Env(copy.deepcopy(optimizer)) for _ in range(test_run)]
                    problem_list = [{'problem': copy.deepcopy(problem)} for _ in range(test_run)]
                    # problem_list = [copy.deepcopy(problem) for _ in range(test_run)]
                    env = ParallelEnv(env_list, para_mode = 'dummy', asynchronous = None, num_cpus = 1, num_gpus = 0)
                    if seed_list is not None:
                        env.seed(seed_list)
                    test_data = env.customized_method('run_batch_episode', problem_list) # List:[dict{cost, fes}] (test_run)
                    pbar_info = {'BBO': type(optimizer).__name__,
                                 'problem': problem.__str__(),
                                 }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)

    def test_2(self):
        # todo 第二种 并行是 agent for 循环
        # todo 每个 agent 做 bs 个问题 x test_run 的列表环境
        print(f'start testing: {self.config.run_time}')

        test_run = self.config.test_run
        bs = self.config.test_batch_size

        seed_list = list(range(1, test_run + 1)) * bs # test_run * bs
        pbar_len = (len(self.agent_for_cp) + len(self.t_optimizer_for_cp)) * (self.test_set.N // bs + self.test_set.N % bs)
        with tqdm(range(pbar_len), desc = "Testing") as pbar:
            for i, problem in enumerate(self.test_set):
                for agent_id, (agent, optimizer) in enumerate(zip(self.agent_for_cp, self.l_optimizer_for_cp)):
                    # for agent an env_list [bs * len(test_run)]
                    # [F1 F1 F1 F2 F2 F2 F3 F3 F3...]
                    '''
                        example: bs = 3 test_run = 2 agent A1 A2
                        [                            A1                            |                            A2                            ]    
                        [O1_F1_r1--O1_F1_r2--O1_F2_r1--O1_F2_r2--O1_F3_r1--O1_F3_r2|O2_F1_r1--O2_F1_r2--O2_F2_r1--O2_F2_r2--O2_F3_r1--O2_F3_r2]
                        PS：Test results can be affected by bs
                    '''
                    env_list = [
                        PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer))
                        for p in problem  # bs
                        for _ in range(test_run) # test_run
                    ]

                    meta_test_data = agent.rollout_batch_episode(envs = env_list,
                                                                 seeds = seed_list,
                                                                 para_mode = 'dummy',
                                                                 asynchronous = None,
                                                                 num_cpus = 1,
                                                                 num_gpus = 0,
                                                                 )
                    # meta_test_data : {cost, fes, return}
                    pbar_info = {'MetaBBO': agent.__str__(),
                                 }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
                # run traditional optimizer
                for optimizer in self.t_optimizer_for_cp:
                    env_list = [BBO_Env(copy.deepcopy(optimizer)) for _ in range(test_run) for _ in range(bs)]
                    problem_list = [{'problem': copy.deepcopy(p)} for p in problem for _ in range(test_run)]
                    # problem_list = [copy.deepcopy(problem) for _ in range(test_run)]
                    env = ParallelEnv(env_list, para_mode = 'dummy', asynchronous = None, num_cpus = 1, num_gpus = 0)
                    env.seed(seed_list)
                    test_data = env.customized_method('run_batch_episode', problem_list)  # List:[dict{cost, fes}] (test_run * bs)
                    pbar_info = {'BBO': type(optimizer).__name__,
                                 }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)




    def test_3(self):
        # todo 第三种 并行是 agent * bs 个问题 * run
        print(f'start testing: {self.config.run_time}')

        test_run = self.config.test_run
        bs = self.config.test_batch_size

        seed_list = list(range(1, test_run + 1)) # test_run
        pbar_len = 2 * (self.test_set.N // bs + self.test_set.N % bs)
        with tqdm(range(pbar_len), desc = "Testing") as pbar:
            for i, problem in enumerate(self.test_set):
                # env_list [bs * len(agent) * len(test_run)]
                # agent_list [bs * len(agent) * len(test_run)]


                '''
                    example: bs = 3 test_run = 2 agent A1 A2
                    [   A1   |   A1   |   A1   |   A1   |   A1   |   A1   |   A2   |   A2   |   A2   |   A2   |   A2   |   A2   ]
                    [O1_F1_r1|O1_F1_r2|O1_F2_r1|O1_F2_r2|O1_F3_r1|O1_F3_r2|O2_F1_r1|O2_F1_r2|O2_F2_r1|O2_F2_r2|O2_F3_r1|O2_F3_r2]
                    PS：Test results are not affected by bs          
                '''
                agent_list = [MetaBBO_Env(copy.deepcopy(agent)) for agent in self.agent_for_cp
                                                                for _ in range(test_run * bs)]
                env_list = [{'env': PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), 'seed': seed} for optimizer in self.l_optimizer_for_cp
                                                                                                       for p in problem
                                                                                                       for seed in seed_list]

                # agent parallel

                MetaBBO = ParallelEnv(agent_list, para_mode = 'ray', asynchronous = None, num_cpus = 1, num_gpus = 0)

                meta_test_data = MetaBBO.customized_method('run_batch_episode', env_list)
                pbar_info = {'Testing': "MetaBBO",
                             }
                pbar.set_postfix(pbar_info)
                pbar.update(1)

                # tradition
                optimizer_list = [BBO_Env(copy.deepcopy(optimizer)) for optimizer in self.t_optimizer_for_cp for _ in range(test_run * bs)]
                problem_list = [{'problem': copy.deepcopy(p)} for _ in range(len(self.t_optimizer_for_cp)) for p in problem for _ in range(test_run)]


                # optimizer_list = []
                # problem_list = []
                # for optimizer in self.t_optimizer_for_cp:
                #     for _ in range(test_run * bs):
                #         optimizer_list.append(BBO_Env(copy.deepcopy(optimizer)))
                #     problem_list.append({'problem': copy.deepcopy(p)} for p in problem for _ in range(test_run))
                BBO = ParallelEnv(optimizer_list, para_mode = 'ray', asynchronous = None, num_cpus = 1, num_gpus = 0)
                BBO.seed(seed_list * bs * len(self.agent_for_cp))
                test_data = BBO.customized_method('run_batch_episode', problem_list)
                pbar_info = {'Testing': "BBO",}
                pbar.set_postfix(pbar_info)
                pbar.update(1)




































def rollout_batch(config):
    print(f'start rollout: {config.run_time}')

    if config.problem[-6:]=='-torch':
        config.problem=config.problem[:-6]

    if config.problem == 'bbob-surrogate':
        config.is_train = False
    config.train_batch_size = 1
    train_set,_=construct_problem_set(config)
    # if 'L2L_Agent' in config.agent_for_rollout:
    #     pre_problem=config.problem
    #     config.problem=pre_problem+'-torch'
    #     torch_train_set,_ = construct_problem_set(config)
    #     config.problem=pre_problem

    agent_load_dir=config.agent_load_dir
    n_checkpoint=config.n_checkpoint

    train_rollout_results = {'cost': {},
                             'pr': {},
                             'sr': {},
                             'return':{}}

    agent_for_rollout=config.agent_for_rollout

    load_agents={}
    for agent_name in agent_for_rollout:
        load_agents[agent_name]=[]
        for checkpoint in range(0,n_checkpoint+1):
            file_path = agent_load_dir+ agent_name + '/' + 'checkpoint'+str(checkpoint) + '.pkl'
            with open(file_path, 'rb') as f:
                load_agents[agent_name].append(pickle.load(f))

    optimizer_for_rollout=[]
    for optimizer_name in config.optimizer_for_rollout:
        optimizer_for_rollout.append(eval(optimizer_name)(copy.deepcopy(config)))
    for problem in train_set:
        train_rollout_results['cost'][problem.__str__()] = {}
        train_rollout_results['pr'][problem.__str__()] = {}
        train_rollout_results['sr'][problem.__str__()] = {}
        train_rollout_results['return'][problem.__str__()] = {}
        for agent_name in agent_for_rollout:
            train_rollout_results['cost'][problem.__str__()][agent_name] = []
            train_rollout_results['pr'][problem.__str__()][agent_name] = []
            train_rollout_results['sr'][problem.__str__()][agent_name] = []
            train_rollout_results['return'][problem.__str__()][agent_name] = []
            for checkpoint in range(0,n_checkpoint+1):
                train_rollout_results['cost'][problem.__str__()][agent_name].append([])
                train_rollout_results['pr'][problem.__str__()][agent_name].append([])
                train_rollout_results['sr'][problem.__str__()][agent_name].append([])
                train_rollout_results['return'][problem.__str__()][agent_name].append([])

    pbar_len = (len(agent_for_rollout)) * train_set.N * (n_checkpoint+1)
    seed_list = list(range(1, 5 + 1))
    with tqdm(range(pbar_len), desc='Rollouting') as pbar:
        for agent_name,optimizer in zip(agent_for_rollout,optimizer_for_rollout):
            return_list=[]  # n_checkpoint + 1
            agent=None
            for checkpoint in range(0,n_checkpoint+1):
                agent=load_agents[agent_name][checkpoint]
                # return_sum=0
                for i,problem in enumerate(train_set):
                    env_list = [PBO_Env(copy.deepcopy(problem), copy.deepcopy(optimizer)) for _ in range(5)]
                    meta_rollout_data = agent.rollout_batch_episode(envs = env_list,
                                                                seeds = seed_list,
                                                                para_mode = 'dummy',
                                                                asynchronous = None,
                                                                num_cpus = 1,
                                                                num_gpus = 0,
                                                                )
                    cost=meta_rollout_data['cost']
                    pr=meta_rollout_data['pr']
                    sr=meta_rollout_data['sr']
                    R=meta_rollout_data['return']

                    train_rollout_results['cost'][problem.__str__()][agent_name][checkpoint] = np.array(cost)
                    train_rollout_results['pr'][problem.__str__()][agent_name][checkpoint] = np.array(pr)
                    train_rollout_results['sr'][problem.__str__()][agent_name][checkpoint] = np.array(sr)
                    train_rollout_results['return'][problem.__str__()][agent_name][checkpoint] = np.array(R)

                    pbar_info = {'problem': problem.__str__(),
                                'agent': type(agent).__name__,
                                'checkpoint': checkpoint,}
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
            
    log_dir=config.rollout_log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + 'rollout.pkl', 'wb') as f:
        pickle.dump(train_rollout_results, f, -1)


def test_for_random_search(config):
    # get entire problem set
    if config.problem == 'bbob-surrogate':
        config.is_train = False

    train_set, test_set = construct_problem_set(config)
    entire_set = train_set + test_set
    # get optimizer
    optimizer = Random_search(copy.deepcopy(config))
    # initialize the dataframe for logging
    test_results = {'cost': {},
                    'fes': {},
                    'T0': 0.,
                    'T1': {},
                    'T2': {}}
    test_results['T1'][type(optimizer).__name__] = 0.
    test_results['T2'][type(optimizer).__name__] = 0.
    for problem in entire_set:
        test_results['cost'][problem.__str__()] = {}
        test_results['fes'][problem.__str__()] = {}
        test_results['cost'][problem.__str__()][type(optimizer).__name__] = []  # 51 np.arrays
        test_results['fes'][problem.__str__()][type(optimizer).__name__] = []  # 51 scalars
    # calculate T0
    test_results['T0'] = cal_t0(config.dim, config.maxFEs)
    # begin testing
    seed = range(51)
    pbar_len = len(entire_set) * 51
    with tqdm(range(pbar_len), desc='test for random search') as pbar:
        for i, problem in enumerate(entire_set):
            T1 = 0
            T2 = 0
            for run in range(51):
                start = time.perf_counter()
                np.random.seed(seed[run])
                info = optimizer.run_episode(problem)
                cost = info['cost']
                while len(cost) < 51:
                    cost.append(cost[-1])
                fes = info['fes']
                end = time.perf_counter()
                if i == 0:
                    T1 += problem.T1
                    T2 += (end - start) * 1000  # ms
                test_results['cost'][problem.__str__()][type(optimizer).__name__].append(cost)
                test_results['fes'][problem.__str__()][type(optimizer).__name__].append(fes)
                pbar_info = {'problem': problem.__str__(),
                             'optimizer': type(optimizer).__name__,
                             'run': run,
                             'cost': cost[-1],
                             'fes': fes, }
                pbar.set_postfix(pbar_info)
                pbar.update(1)
            if i == 0:
                test_results['T1'][type(optimizer).__name__] = T1 / 51
                test_results['T2'][type(optimizer).__name__] = T2 / 51
    return test_results


def name_translate(problem):
    if problem in ['bbob', 'bbob-torch']:
        return 'Synthetic'
    elif problem in ['bbob-noisy', 'bbob-noisy-torch']:
        return 'Noisy-Synthetic'
    elif problem in ['protein', 'protein-torch']:
        return 'Protein-Docking'
    else:
        raise ValueError(problem + ' is not defined!')


def mgd_test(config):
    print(f'start MGD_test: {config.run_time}')
    # get test set

    if config.problem == 'bbob-surrogate':
        config.is_train = False

    _, test_set = construct_problem_set(config)
    # get agents
    with open(config.model_from, 'rb') as f:
        agent_from = pickle.load(f)
    with open(config.model_to, 'rb') as f:
        agent_to = pickle.load(f)
    # get optimizer
    l_optimizer = eval(config.optimizer)(copy.deepcopy(config))
    # initialize the dataframe for logging
    test_results = {'cost': {},
                    'fes': {},
                    'T0': 0.,
                    'T1': {},
                    'T2': {}}
    agent_name_list = [f'{config.agent}_from', f'{config.agent}_to']
    for agent_name in agent_name_list:
        test_results['T1'][agent_name] = 0.
        test_results['T2'][agent_name] = 0.
    for problem in test_set:
        test_results['cost'][problem.__str__()] = {}
        test_results['fes'][problem.__str__()] = {}
        for agent_name in agent_name_list:
            test_results['cost'][problem.__str__()][agent_name] = []  # 51 np.arrays
            test_results['fes'][problem.__str__()][agent_name] = []  # 51 scalars
    # calculate T0
    test_results['T0'] = cal_t0(config.dim, config.maxFEs)
    # begin mgd_test
    seed = range(51)
    pbar_len = len(agent_name_list) * len(test_set) * 51
    with tqdm(range(pbar_len), desc='MGD_Test') as pbar:
        for i, problem in enumerate(test_set):
            # run model_from and model_to
            for agent_id, agent in enumerate([agent_from, agent_to]):
                T1 = 0
                T2 = 0
                for run in range(51):
                    start = time.perf_counter()
                    np.random.seed(seed[run])
                    # construct an ENV for (problem,optimizer)
                    env = PBO_Env(problem, l_optimizer)
                    info = agent.rollout_episode(env)
                    cost = info['cost']
                    while len(cost) < 51:
                        cost.append(cost[-1])
                    fes = info['fes']
                    end = time.perf_counter()
                    if i == 0:
                        T1 += env.problem.T1
                        T2 += (end - start) * 1000  # ms
                    test_results['cost'][problem.__str__()][agent_name_list[agent_id]].append(cost)
                    test_results['fes'][problem.__str__()][agent_name_list[agent_id]].append(fes)
                    pbar_info = {'problem': problem.__str__(),
                                 'optimizer': agent_name_list[agent_id],
                                 'run': run,
                                 'cost': cost[-1],
                                 'fes': fes}
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
                if i == 0:
                    test_results['T1'][agent_name_list[agent_id]] = T1 / 51
                    test_results['T2'][agent_name_list[agent_id]] = T2 / 51
    if not os.path.exists(config.mgd_test_log_dir):
        os.makedirs(config.mgd_test_log_dir)
    with open(config.mgd_test_log_dir + 'test.pkl', 'wb') as f:
        pickle.dump(test_results, f, -1)
    random_search_results = test_for_random_search(config)
    with open(config.mgd_test_log_dir + 'random_search_baseline.pkl', 'wb') as f:
        pickle.dump(random_search_results, f, -1)
    logger = Logger(config)
    aei, aei_std = logger.aei_metric(test_results, random_search_results, config.maxFEs)
    print(f'AEI: {aei}')
    print(f'AEI STD: {aei_std}')
    print(f'MGD({name_translate(config.problem_from)}_{config.difficulty_from}, {name_translate(config.problem_to)}_{config.difficulty_to}) of {config.agent}: '
          f'{100 * (1 - aei[config.agent+"_from"] / aei[config.agent+"_to"])}%')


def mte_test(config):
    print(f'start MTE_test: {config.run_time}')
    pre_train_file = config.pre_train_rollout
    scratch_file = config.scratch_rollout
    agent = config.agent
    min_max = False

    # preprocess data for agent
    def preprocess(file, agent):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        # aggregate all problem's data together
        returns = data['return']
        results = None
        i = 0
        for problem in returns.keys():
            if i == 0:
                results = np.array(returns[problem][agent])
            else:
                results = np.concatenate([results, np.array(returns[problem][agent])], axis=1)
            i += 1
        return np.array(results)

    bbob_data = preprocess(pre_train_file, agent)
    noisy_data = preprocess(scratch_file, agent)
    # calculate min_max avg
    temp = np.concatenate([bbob_data, noisy_data], axis=1)
    if min_max:
        temp_ = (temp - temp.min(-1)[:, None]) / (temp.max(-1)[:, None] - temp.min(-1)[:, None])
    else:
        temp_ = temp
    bd, nd = temp_[:, :90], temp_[:, 90:]
    checkpoints = np.hsplit(bd, 18)
    g = []
    for i in range(18):
        g.append(checkpoints[i].tolist())
    checkpoints = np.array(g)
    avg = bd.mean(-1)
    avg = savgol_filter(avg, 13, 5)
    std = np.mean(np.std(checkpoints, -1), 0) / np.sqrt(5)
    checkpoints = np.hsplit(nd, 18)
    g = []
    for i in range(18):
        g.append(checkpoints[i].tolist())
    checkpoints = np.array(g)
    std_ = np.mean(np.std(checkpoints, -1), 0) / np.sqrt(5)
    avg_ = nd.mean(-1)
    avg_ = savgol_filter(avg_, 13, 5)
    plt.figure(figsize=(40, 15))
    plt.subplot(1, 3, (2, 3))
    x = np.arange(21)
    x = (1.5e6 / x[-1]) * x
    idx = 21
    smooth = 1
    s = np.zeros(21)
    a = s[0] = avg[0]
    norm = smooth + 1
    for i in range(1, 21):
        a = a * smooth + avg[i]
        s[i] = a / norm if norm > 0 else a
        norm *= smooth
        norm += 1

    s_ = np.zeros(21)
    a = s_[0] = avg_[0]
    norm = smooth + 1
    for i in range(1, 21):
        a = a * smooth + avg_[i]
        s_[i] = a / norm if norm > 0 else a
        norm *= smooth
        norm += 1
    plt.plot(x[:idx], s[:idx], label='pre-train', marker='*', markersize=30, markevery=1, c='blue', linewidth=5)
    plt.fill_between(x[:idx], s[:idx] - std[:idx], s[:idx] + std[:idx], alpha=0.2, facecolor='blue')
    plt.plot(x[:idx], s_[:idx], label='scratch', marker='*', markersize=30, markevery=1, c='red', linewidth=5)
    plt.fill_between(x[:idx], s_[:idx] - std_[:idx], s_[:idx] + std_[:idx], alpha=0.2, facecolor='red')
    # Search MTE
    scratch = s_[:idx]
    pretrain = s[:idx]
    topx = np.argmax(scratch)
    topy = scratch[topx]
    T = topx / 21
    t = 0
    if pretrain[0] < topy:
        for i in range(1, 21):
            if pretrain[i - 1] < topy <= pretrain[i]:
                t = ((topy - pretrain[i - 1]) / (pretrain[i] - pretrain[i - 1]) + i - 1) / 21
                break
    if np.max(pretrain[-1]) < topy:
        t = 1
    MTE = 1 - t / T

    print(f'MTE({name_translate(config.problem_from)}_{config.difficulty_from}, {name_translate(config.problem_to)}_{config.difficulty_to}) of {config.agent}: '
          f'{MTE}')

    ax = plt.gca()
    ax.xaxis.get_offset_text().set_fontsize(45)
    plt.xticks(fontsize=45, )
    plt.yticks(fontsize=45)
    plt.legend(loc=0, fontsize=60)
    plt.xlabel('Learning Steps', fontsize=55)
    plt.ylabel('Avg Return', fontsize=55)
    plt.title(f'Fine-tuning ({name_translate(config.problem_from)} $\\rightarrow$ {name_translate(config.problem_to)})',
              fontsize=60)
    plt.tight_layout()
    plt.grid()
    plt.subplots_adjust(wspace=0.2)
    if not os.path.exists(config.mte_test_log_dir):
        os.makedirs(config.mte_test_log_dir)
    plt.savefig(f'{config.mte_test_log_dir}/MTE_{agent}.png', bbox_inches='tight')

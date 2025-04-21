import copy
from environment.problem.utils import construct_problem_set
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
from tqdm import tqdm
import os, psutil
from environment.basic_environment import PBO_Env
from logger import *
from environment.parallelenv.parallelenv import ParallelEnv
import json
import torch
import gym
from typing import Optional, Union, Literal, List
from environment.optimizer.basic_optimizer import Basic_Optimizer
from rl import Basic_Agent
from environment.problem.basic_problem import Basic_Problem
from dill import dumps, loads
from environment.optimizer import (
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

    GLHF_Optimizer,
    B2OPT_Optimizer,
)

from baseline.bbo import (
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
    PYPOP7
)

from baseline.metabbo import (
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
    GLHF,
    B2OPT
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


def record_data(data, test_set, agent_for_rollout, checkpoints, results, meta_results, config):
    for item in data:
        for key in item.keys():
            if key == 'metadata' and config.full_meta_data:
                meta_results[item['problem_name']][item['agent_name']].append(item[key])
                continue
            if key not in ['agent_name', 'problem_name']:
                if key not in results.keys():
                    results[key] = {}
                    for problem in test_set.data:
                        results[key][problem.__str__()] = {}
                        for agent_id in checkpoints:
                            results[key][problem.__str__()][agent_for_rollout+f'-{agent_id}'] = []  # 51 np.arrays
                results[key][item['problem_name']][item['agent_name']].append(item[key])
    return results, meta_results


def store_meta_data(log_dir, meta_data_results):
    if not os.path.exists(log_dir+'/metadata/'):
        os.makedirs(log_dir+'/metadata/')
    for pname in meta_data_results.keys():
        if not os.path.exists(log_dir+f'/metadata/{pname}.pkl'):
            with open(log_dir + f'/metadata/{pname}.pkl', 'wb') as f:
                pickle.dump(meta_data_results[pname], f, -1)
            for agent in meta_data_results[pname].keys():  # clear memory storage
                meta_data_results[pname][agent] = []
        else:
            with open(log_dir + f'/metadata/{pname}.pkl', 'rb') as f:
                data_results = pickle.load(f)
            for key in meta_data_results[pname].keys():
                if key in data_results.keys():
                    data_results[key] += meta_data_results[pname][key]  # list + list
                else:
                    data_results[key] = meta_data_results[pname][key]
                meta_data_results[pname][key] = []  # clear memory storage
            with open(log_dir + f'/metadata/{pname}.pkl', 'wb') as f:
                pickle.dump(data_results, f, -1)
    return meta_data_results
                    
                    
class BBO_TestUnit():
    """
        A test unit for RAY parallel with a problem and a basic optimizer.
        """

    def __init__(self,
                 optimizer: Basic_Optimizer,
                 problem: Basic_Problem,
                 seed: int,
                 ):
        self.optimizer = optimizer
        self.problem = problem
        self.seed = seed

    def run_batch_episode(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.set_default_dtype(torch.float64)
        self.optimizer.seed(self.seed)
        self.problem.reset()
        start_time = time.perf_counter()
        res = self.optimizer.run_episode(self.problem)
        end_time = time.perf_counter()
        res['T1'] = self.problem.T1
        res['T2'] = (end_time - start_time) * 1000
        res['agent_name'] = self.optimizer.__str__()
        res['problem_name'] = self.problem.__str__()
        return res


class MetaBBO_TestUnit():
    """
        A test unit for RAY parallel with an agent, an env and a seed.
        """

    def __init__(self,
                 agent: Basic_Agent,
                 env: PBO_Env,
                 seed: int,
                 checkpoint: int=None,
                 ):
        self.agent = agent
        self.env = env
        self.seed = seed
        self.checkpoint = checkpoint

    def run_batch_episode(self, required_info = {}):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.set_default_dtype(torch.float64)

        start_time = time.perf_counter()
        res = self.agent.rollout_episode(self.env, self.seed, required_info)
        end_time = time.perf_counter()
        res['T1'] = self.env.problem.T1
        res['T2'] = (end_time - start_time) * 1000
        agent_name = self.agent.__str__()
        if self.checkpoint is not None:
            agent_name += f'-{self.checkpoint}'
        res['agent_name'] = agent_name
        res['problem_name'] = self.env.problem.__str__()
        return res


class Tester(object):
    def __init__(self, config):
        self.key_list = config.agent
        self.log_dir = config.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.config = config

        # if self.config.test_problem[-6:]=='-torch':
        #     self.config.test_problem=self.config.test_problem[:-6]

        if config.test_problem in ['bbob-surrogate-10D','bbob-surrogate-5D','bbob-surrogate-2D']:
            config.is_train = False

        self.train_set, self.test_set = construct_problem_set(self.config)
        self.config.dim = max(self.train_set.maxdim, self.test_set.maxdim)
        # if 'L2L_Agent' in config.agent_for_cp or 'L2L_Agent' == config.agent:
        #     pre_problem=config.problem
        #     config.problem=pre_problem+'-torch'
        #     _,self.torch_test_set = construct_problem_set(config)
        #     config.problem=pre_problem
        
        self.seed = range(51)
        # initialize the dataframe for logging
        self.test_results = {'cost': {},
                             'fes': {},
                             'T1': {},
                             'T2': {},
                             }
        self.meta_data_results = {}
        if not os.path.exists(self.log_dir+'/metadata/'):
            os.makedirs(self.log_dir+'/metadata/')
        with open(self.log_dir + f'/metadata/config.pkl', 'wb') as f:
            pickle.dump(self.config, f, -1)
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

        for key in self.test_results.keys():
            self.initialize_record(key)
        self.test_results['config'] = copy.deepcopy(self.config)
        self.test_results['T0'] = np.mean([cal_t0(p.dim, config.maxFEs) for p in self.test_set.data])
        if config.full_meta_data:
            for problem in self.test_set.data:
                self.meta_data_results[problem.__str__()] = {}
                for agent_name in self.agent_name_list:
                    self.meta_data_results[problem.__str__()][agent_name] = []  # test_run x fes
                for optimizer in self.t_optimizer_for_cp:
                    self.meta_data_results[problem.__str__()][type(optimizer).__name__] = []
            
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def initialize_record(self, key):
        if key not in self.test_results.keys():
            self.test_results[key] = {}
        for problem in self.test_set.data:
            self.test_results[key][problem.__str__()] = {}
            for agent_name in self.agent_name_list:
                self.test_results[key][problem.__str__()][agent_name] = []  # 51 np.arrays
            for optimizer in self.t_optimizer_for_cp:
                self.test_results[key][problem.__str__()][type(optimizer).__name__] = []  # 51 np.arrays
        
    def record_test_data(self, data: list):
        for item in data:
            for key in item.keys():
                if key == 'metadata' and self.config.full_meta_data:
                    self.meta_data_results[item['problem_name']][item['agent_name']].append(item[key])
                    continue
                if key not in ['agent_name', 'problem_name']:
                    if key not in self.test_results.keys():
                        self.initialize_record(key)
                    self.test_results[key][item['problem_name']][item['agent_name']].append(item[key])            

    def test(self, ):
        # todo 第三种 并行是 agent * bs 个问题 * run
        print(f'start testing: {self.config.run_time}')
        parallel_batch = self.config.parallel_batch  # 'Full', 'Baseline_Problem', 'Problem_Testrun', 'Batch'
        test_run = self.config.test_run
        seed_list = list(range(1, test_run + 1)) # test_run
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if parallel_batch == 'Full':
            testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for (agent, optimizer) in zip(self.agent_for_cp, self.l_optimizer_for_cp)
                                                                                                                               for p in self.test_set.data
                                                                                                                               for seed in seed_list]
            testunit_list += [BBO_TestUnit(copy.deepcopy(optimizer), copy.deepcopy(p), seed) for optimizer in self.t_optimizer_for_cp
                                                                                        for p in self.test_set.data
                                                                                        for seed in seed_list]
            MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
            meta_test_data = MetaBBO_test.rollout()
            self.record_test_data(meta_test_data)
            self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
                
        elif parallel_batch == 'Baseline_Problem':
            pbar = tqdm(total=len(seed_list), desc="Baseline_Problem Testing")
            for seed in seed_list:
                testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for (agent, optimizer) in zip(self.agent_for_cp, self.l_optimizer_for_cp)
                                                                                                                                for p in self.test_set.data
                                                                                                                                ]
                testunit_list += [BBO_TestUnit(copy.deepcopy(optimizer), copy.deepcopy(p), seed, self.config) for optimizer in self.t_optimizer_for_cp
                                                                                                for p in self.test_set.data
                                                                                                ]
                MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                meta_test_data = MetaBBO_test.rollout()
                self.record_test_data(meta_test_data)
                self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
                pbar.update()
            pbar.close()
                
        elif parallel_batch == 'Problem_Testrun':
            pbar = tqdm(total=len(self.agent_for_cp) + len(self.t_optimizer_for_cp), desc="Problem_Testrun Testing")
            for (agent, optimizer) in zip(self.agent_for_cp, self.l_optimizer_for_cp):
                pbar.set_description(f"Problem_Testrun Testing {agent.__str__()}")
                testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) 
                                                                                                                                for p in self.test_set.data
                                                                                                                                for seed in seed_list]
                MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                meta_test_data = MetaBBO_test.rollout()
                self.record_test_data(meta_test_data)
                self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
                pbar.update()
            for optimizer in self.t_optimizer_for_cp:
                pbar.set_description(f"Problem_Testrun Testing {optimizer.__str__()}")
                testunit_list += [BBO_TestUnit(copy.deepcopy(optimizer), copy.deepcopy(p), seed) for p in self.test_set.data
                                                                                                 for seed in seed_list]
                MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                meta_test_data = MetaBBO_test.rollout()
                self.record_test_data(meta_test_data)
                self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
                pbar.update()
            pbar.close()
                
        elif parallel_batch == 'Batch':
            pbar_len = (len(self.agent_for_cp) + len(self.t_optimizer_for_cp)) * np.ceil(self.test_set.N / self.config.test_batch_size) * self.config.test_run
            pbar = tqdm(total=pbar_len, desc="Batch Testing")
            for (agent, optimizer) in zip(self.agent_for_cp, self.l_optimizer_for_cp):
                for ip, problem in enumerate(self.test_set):
                    for i, seed in enumerate(seed_list):
                        pbar.set_description_str(f"Batch Testing Agent {agent.__str__()} with Problem Batch {ip}, Run {i}")
                        testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for p in problem]
                        MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                        meta_test_data = MetaBBO_test.rollout()
                        self.record_test_data(meta_test_data)
                        pbar.update()
                    self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
            for optimizer in self.t_optimizer_for_cp:
                for ip, problem in enumerate(self.test_set):
                    for i, seed in enumerate(seed_list):
                        pbar.set_description_str(f"Batch Testing Optimizer {optimizer.__str__()} with Problem Batch {ip}, Run {i}")
                        testunit_list = [BBO_TestUnit(loads(dumps(optimizer)), copy.deepcopy(p), seed) for p in problem]
                        MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                        meta_test_data = MetaBBO_test.rollout()
                        self.record_test_data(meta_test_data)
                        pbar.update()
                    self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
            pbar.close()
                        
        else:
            raise NotImplementedError
        
        with open(self.log_dir + f'/test_results.pkl', 'wb') as f:
            pickle.dump(self.test_results, f, -1)


    def test_for_random_search(self):
        config = self.config
        # get entire problem set
        if config.problem in ['bbob-surrogate-10D','bbob-surrogate-5D','bbob-surrogate-2D']:
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


    def name_translate(self, problem):
        if problem in ['bbob', 'bbob-torch']:
            return 'Synthetic'
        elif problem in ['bbob-noisy', 'bbob-noisy-torch']:
            return 'Noisy-Synthetic'
        elif problem in ['protein', 'protein-torch']:
            return 'Protein-Docking'
        else:
            raise ValueError(problem + ' is not defined!')

    def mgd_test(self, ):
        config = self.config
        print(f'start MGD_test: {config.run_time}')
        # get test set
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if config.problem in ['bbob-surrogate-10D','bbob-surrogate-5D','bbob-surrogate-2D']:
            config.is_train = False

        _, test_set = construct_problem_set(config)
        # get agents
        with open('model.json', 'r', encoding = 'utf-8') as f:
            json_data = json.load(f)
        baseline = json_data[config.model_from]
        agent_name = baseline["Agent"]
        l_optimizer = baseline['Optimizer']
        dir_from = baseline['dir']
        dir_to = json_data[config.model_to]['dir']
        # get agent
        with open(dir_from, 'rb') as f:
            agent_from = pickle.load(f)
        with open(dir_to, 'rb') as f:
            agent_to = pickle.load(f)
        
        # get optimizer
        l_optimizer = eval(l_optimizer)(copy.deepcopy(config))
        # initialize the dataframe for logging
        self.test_results = {'cost': {},
                             'fes': {},
                             'T1': {},
                             'T2': {},
                             }
        self.meta_data_results = {}
        agent_name_list = [f'{config.agent}_from', f'{config.agent}_to']
        for key in self.test_results.keys():
            self.initialize_record(key)
        
        if config.full_meta_data:
            for problem in self.test_set.data:
                self.meta_data_results[problem.__str__()] = {}
                for agent_name in self.agent_name_list:
                    self.meta_data_results[problem.__str__()][agent_name] = []  # test_run x fes
                for optimizer in self.t_optimizer_for_cp:
                    self.meta_data_results[problem.__str__()][type(optimizer).__name__] = []

        # calculate T0
        self.test_results['T0'] = np.mean([cal_t0(p.dim, config.maxFEs) for p in self.test_set.data])
        # begin mgd_test

        test_run = self.config.test_run
        parallel_batch = self.config.parallel_batch
        seed_list = list(range(1, test_run + 1))

        if parallel_batch == 'Full':
            testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_from), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer)), seed) for p in test_set.data for seed in seed_list]
            testunit_list += [MetaBBO_TestUnit(copy.deepcopy(agent_to), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer)), seed) for p in test_set.data for seed in seed_list]

            MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
            meta_test_data = MetaBBO_test.rollout()
            self.record_test_data(meta_test_data)
            self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)

        elif parallel_batch == 'Baseline_Problem':
            pbar = tqdm(total = len(seed_list), desc = "Baseline_Problem Testing")
            for seed in seed_list:
                testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_from), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer)), seed) for p in test_set.data]
                testunit_list += [MetaBBO_TestUnit(copy.deepcopy(agent_to), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer)), seed) for p in test_set.data]
                MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                meta_test_data = MetaBBO_test.rollout()
                self.record_test_data(meta_test_data)
                self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
                pbar.update()
            pbar.close()

        elif parallel_batch == 'Problem_Testrun':
            pbar_len = 2
            pbar = tqdm(total = pbar_len, desc = "Problem_Testrun Testing")
            pbar.set_description(f"Problem_Testrun Testing from {agent_from.__str__()} to {agent_to.__str__()}")
            testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_from), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer)), seed)
                             for p in self.test_set.data
                             for seed in seed_list]
            MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
            meta_test_data = MetaBBO_test.rollout()
            self.record_test_data(meta_test_data)
            self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
            pbar.update()

            testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_to), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer)), seed)
                             for p in self.test_set.data
                             for seed in seed_list]
            MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
            meta_test_data = MetaBBO_test.rollout()
            self.record_test_data(meta_test_data)
            self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
            pbar.update()
            pbar.close()

        elif parallel_batch == 'Batch':
            pbar_len = 2 * np.ceil(test_set.N / config.test_batch_size) * test_run
            pbar = tqdm(total = pbar_len, desc = "Batch Testing")
            for ip, problem in enumerate(test_set):
                for i, seed in enumerate(seed_list):
                    pbar.set_description_str(f"Batch Testing From Agent {agent_from.__str__()} with Problem Batch {ip}, Run {i}")
                    testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_from), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer)), seed) for p in problem]
                    MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                    meta_test_data = MetaBBO_test.rollout()
                    self.record_test_data(meta_test_data)
                    pbar.update()
                self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
            for ip, problem in enumerate(test_set):
                for i, seed in enumerate(seed_list):
                    pbar.set_description_str(f"Batch Testing To Agent {agent_to.__str__()} with Problem Batch {ip}, Run {i}")
                    testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_to), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer)), seed) for p in problem]
                    MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                    meta_test_data = MetaBBO_test.rollout()
                    self.record_test_data(meta_test_data)
                    pbar.update()
                self.meta_data_results = store_meta_data(self.log_dir, self.meta_data_results)
            pbar.close()

        with open(config.mgd_test_log_dir + 'test_results.pkl', 'wb') as f:
            pickle.dump(self.test_results, f, -1)

        # pbar_len = len(agent_name_list) * len(test_set) * 51
        # with tqdm(range(pbar_len), desc='MGD_Test') as pbar:
        #     for i, problem in enumerate(test_set):
        #         # run model_from and model_to
        #         for agent_id, agent in enumerate([agent_from, agent_to]):
        #             T1 = 0
        #             T2 = 0
        #             for run in range(51):
        #                 start = time.perf_counter()
        #                 np.random.seed(seed[run])
        #                 # construct an ENV for (problem,optimizer)
        #                 env = PBO_Env(problem, l_optimizer)
        #                 info = agent.rollout_episode(env)
        #                 cost = info['cost']
        #                 while len(cost) < 51:
        #                     cost.append(cost[-1])
        #                 fes = info['fes']
        #                 end = time.perf_counter()
        #                 if i == 0:
        #                     T1 += env.problem.T1
        #                     T2 += (end - start) * 1000  # ms
        #                 test_results['cost'][problem.__str__()][agent_name_list[agent_id]].append(cost)
        #                 test_results['fes'][problem.__str__()][agent_name_list[agent_id]].append(fes)
        #                 pbar_info = {'problem': problem.__str__(),
        #                             'optimizer': agent_name_list[agent_id],
        #                             'run': run,
        #                             'cost': cost[-1],
        #                             'fes': fes}
        #                 pbar.set_postfix(pbar_info)
        #                 pbar.update(1)
        #             if i == 0:
        #                 test_results['T1'][agent_name_list[agent_id]] = T1 / 51
        #                 test_results['T2'][agent_name_list[agent_id]] = T2 / 51
        # if not os.path.exists(config.mgd_test_log_dir):
        #     os.makedirs(config.mgd_test_log_dir)
        # with open(config.mgd_test_log_dir + 'test.pkl', 'wb') as f:
        #     pickle.dump(test_results, f, -1)
        # random_search_results = test_for_random_search(config)
        # with open(config.mgd_test_log_dir + 'random_search_baseline.pkl', 'wb') as f:
        #     pickle.dump(random_search_results, f, -1)
        # logger = Logger(config)
        # aei, aei_std = logger.aei_metric(test_results, random_search_results, config.maxFEs)
        # print(f'AEI: {aei}')
        # print(f'AEI STD: {aei_std}')
        # print(f'MGD({name_translate(config.problem_from)}_{config.difficulty_from}, {name_translate(config.problem_to)}_{config.difficulty_to}) of {config.agent}: '
        #     f'{100 * (1 - aei[config.agent+"_from"] / aei[config.agent+"_to"])}%')


    def mte_test(self, ):
        config = self.config
        print(f'start MTE_test: {config.run_time}')
        with open('model.json', 'r', encoding = 'utf-8') as f:
            json_data = json.load(f)
        pre_train = json_data[config.pre_train_rollout]
        scratch_rollout = json_data[config.scratch_rollout]

        pre_train_file = pre_train['dir']
        scratch_file = scratch_rollout['dir']

        agent = pre_train['Agent']

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

        print(f'MTE({self.name_translate(config.problem_from)}_{config.difficulty_from}, {self.name_translate(config.problem_to)}_{config.difficulty_to}) of {agent}: '
            f'{MTE}')

        ax = plt.gca()
        ax.xaxis.get_offset_text().set_fontsize(45)
        plt.xticks(fontsize=45, )
        plt.yticks(fontsize=45)
        plt.legend(loc=0, fontsize=60)
        plt.xlabel('Learning Steps', fontsize=55)
        plt.ylabel('Avg Return', fontsize=55)
        plt.title(f'Fine-tuning ({self.name_translate(config.problem_from)} $\\rightarrow$ {self.name_translate(config.problem_to)})',
                fontsize=60)
        plt.tight_layout()
        plt.grid()
        plt.subplots_adjust(wspace=0.2)
        if not os.path.exists(config.mte_test_log_dir):
            os.makedirs(config.mte_test_log_dir)
        plt.savefig(f'{config.mte_test_log_dir}/MTE_{agent}.png', bbox_inches='tight')


def rollout_batch(config):
    print(f'start rollout: {config.run_time}')
    num_gpus = 0 if config.device == 'cpu' else 1
    if config.test_problem in ['bbob-surrogate-10D','bbob-surrogate-5D','bbob-surrogate-2D']:
        config.is_train = False
    train_set, test_set = construct_problem_set(config)

    config.dim = max(train_set.maxdim, test_set.maxdim)

    agent_for_rollout=config.agent_for_rollout
    parallel_batch = config.parallel_batch

    agents=[]
    optimizer_for_rollout = []
    with open('model.json', 'r', encoding = 'utf-8') as f:
        json_data = json.load(f)
    if agent_for_rollout not in json_data.keys():
        raise KeyError(f"Missing key '{agent_for_rollout}' in model.json")

    # get key
    baseline = json_data[agent_for_rollout]
    agent_name = baseline["Agent"]
    l_optimizer = baseline['Optimizer']
    upper_dir = baseline['dir']
    if not os.path.isdir(upper_dir):  # path to .pkl files
        upper_dir = os.path.join(*tuple(str.split(upper_dir, '/')[:-1]))
        
    checkpoints = config.checkpoints_for_rollout
    if checkpoints is None:
        epoch_list = [f for f in os.listdir(upper_dir) if f.endswith('.pkl')]
        checkpoints = np.arange(len(epoch_list))
    n_checkpoint=len(checkpoints)

    # get agent
    # learning_step
    steps = []
    for agent_id in checkpoints:
        with open(os.path.join(upper_dir, f'checkpoint-{agent_id}.pkl'), 'rb') as f:
            agent = pickle.load(f)
            steps.append(agent.get_step())
            if agent_id:
                steps[-1] += 400
            agents.append(agent)
            optimizer_for_rollout.append(eval(l_optimizer)(copy.deepcopy(config)))

    rollout_results = {'cost': {},
                        'return':{},
                       }
    meta_data_results = {}
    for key in rollout_results.keys():
        if key not in rollout_results.keys():
            rollout_results[key] = {}
        for problem in test_set.data:
            rollout_results[key][problem.__str__()] = {}
            meta_data_results[problem.__str__()] = {}
            for agent_id in checkpoints:
                rollout_results[key][problem.__str__()][agent_name+f'-{agent_id}'] = []  # 51 np.arrays
                meta_data_results[problem.__str__()][agent_name+f'-{agent_id}'] = []

    rollout_results['config'] = copy.deepcopy(config)

    pbar_len = int(np.ceil(test_set.N * n_checkpoint / test_set.batch_size))
    seed_list = list(range(1, config.rollout_run + 1))

    if parallel_batch == 'Full':
        testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed, ckp) for (ckp, agent, optimizer) in zip(checkpoints, agents, optimizer_for_rollout)
                                                                                                                            for p in test_set.data
                                                                                                                            for seed in seed_list]
        MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
        meta_test_data = MetaBBO_test.rollout()
        rollout_results, meta_data_results = record_data(meta_test_data, test_set, agent_for_rollout, checkpoints, rollout_results, meta_data_results, config)
        meta_data_results = store_meta_data(config.rollout_log_dir, meta_data_results)
    elif parallel_batch == 'Baseline_Problem':
        pbar = tqdm(total=len(seed_list), desc="Baseline_Problem Rollouting")
        for seed in seed_list:
            testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed, ckp) for (ckp, agent, optimizer) in zip(checkpoints, agents, optimizer_for_rollout)
                                                                                                                            for p in test_set.data
                                                                                                                            ]
            MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
            meta_test_data = MetaBBO_test.rollout()
            rollout_results, meta_data_results = record_data(meta_test_data, test_set, agent_for_rollout, checkpoints, rollout_results, meta_data_results, config)
            meta_data_results = store_meta_data(config.rollout_log_dir, meta_data_results)
            pbar.update()
        pbar.close()
            
    elif parallel_batch == 'Problem_Testrun':
        pbar = tqdm(total=len(agents), desc="Problem_Testrun Rollouting")
        for (ckp, agent, optimizer) in zip(checkpoints, agents, optimizer_for_rollout):
            pbar.set_description(f"Problem_Testrun Rollouting Checkpoint {ckp}")
            testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed, ckp) 
                                                                                                                            for p in test_set.data
                                                                                                                            for seed in seed_list]
            MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
            meta_test_data = MetaBBO_test.rollout()
            rollout_results, meta_data_results = record_data(meta_test_data, test_set, agent_for_rollout, checkpoints, rollout_results, meta_data_results, config)
            meta_data_results = store_meta_data(config.rollout_log_dir, meta_data_results)
            pbar.update()
        pbar.close()
            
    elif parallel_batch == 'Batch':
        pbar_len = len(agents)  * np.ceil(test_set.N / config.test_batch_size) * config.test_run
        pbar = tqdm(total=pbar_len, desc="Batch Rollouting")
        for (ckp, agent, optimizer) in zip(checkpoints, agents, optimizer_for_rollout):
            for ip, problem in enumerate(test_set):
                for i, seed in enumerate(seed_list):
                    pbar.set_description_str(f"Batch Rollouting Checkpoint {ckp} with Problem Batch {ip}, Run {i}")
                    testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed, ckp) for p in problem]
                    MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                    meta_test_data = MetaBBO_test.rollout()
                    rollout_results, meta_data_results = record_data(meta_test_data, test_set, agent_for_rollout, checkpoints, rollout_results, meta_data_results, config)
                    pbar.update()
            meta_data_results = store_meta_data(config.rollout_log_dir, meta_data_results)
        pbar.close()
                    
    else:
        raise NotImplementedError

    rollout_results['steps'] = steps

    log_dir=config.rollout_log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + 'rollout.pkl', 'wb') as f:
        pickle.dump(rollout_results, f, -1)


# from problem import bbob, bbob_torch, protein_docking
import pickle
from tqdm import tqdm
from environment.basic_environment import PBO_Env
from logger import Logger
from utils import *
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import copy
from agent.utils import save_class
from problem.SOO import bbob_numpy,bbob_surrogate,bbob_torch,protein_docking
from problem.SOO.self_generated.Symbolic_bench_numpy.Symbolic_bench_Dataset import Symbolic_bench_Dataset
from problem.SOO.self_generated.Symbolic_bench_torch.Symbolic_bench_Dataset import Symbolic_bench_Dataset_torch
from problem.SOO.LSGO.classical.cec2013lsgo_torch.cec2013lsgo_dataset import CEC2013LSGO_Dataset as LSGO_Dataset_torch
from problem.SOO.LSGO.classical.cec2013lsgo_numpy.cec2013lsgo_dataset import CEC2013LSGO_Dataset as LSGO_Dataset_numpy
from problem.SOO.uav_numpy.uav_dataset import UAV_Dataset_numpy
from problem.SOO.uav_torch.uav_dataset import UAV_Dataset_torch
from problem import bbob, bbob_torch, protein_docking, mmo_dataset


def construct_problem_set(config):
    problem = config.problem
    if problem in ['bbob', 'bbob-noisy']:
        return bbob_numpy.bbob_dataset.BBOB_Dataset.get_datasets(suit=config.problem,
                                              dim=config.dim,
                                              upperbound=config.upperbound,
                                              train_batch_size=config.train_batch_size,
                                              test_batch_size=config.test_batch_size,
                                              difficulty=config.difficulty)
    elif problem in ['bbob-torch', 'bbob-noisy-torch']:
        return bbob_torch.BBOB_Dataset_torch.get_datasets(suit=config.problem,
                                                          dim=config.dim,
                                                          upperbound=config.upperbound,
                                                          train_batch_size=config.train_batch_size,
                                                          test_batch_size=config.test_batch_size,
                                                          difficulty=config.difficulty)

    elif problem in ['protein', 'protein-torch']:
        return protein_docking.Protein_Docking_Dataset.get_datasets(version=problem,
                                                                    train_batch_size=config.train_batch_size,
                                                                    test_batch_size=config.test_batch_size,
                                                                    difficulty=config.difficulty)

    elif problem in ['bbob-surrogate']:
        return bbob_surrogate.bbob_surrogate_Dataset.get_datasets(config=config,
                                              dim=config.dim,
                                              upperbound=config.upperbound,
                                              train_batch_size=config.train_batch_size,
                                              test_batch_size=config.test_batch_size,
                                              difficulty=config.difficulty)

    elif problem in ['Symbolic_bench','Symbolic_bench-torch']:
        if problem == 'Symbolic_bench':
            return Symbolic_bench_Dataset.get_datasets(upperbound=config.upperbound,
                                                       train_batch_size=config.train_batch_size,
                                                          test_batch_size=config.test_batch_size)
        else:
            return Symbolic_bench_Dataset_torch.get_datasets(upperbound=config.upperbound,
                                                       train_batch_size=config.train_batch_size,
                                                          test_batch_size=config.test_batch_size)
    elif problem in ['lsgo-torch']:
        return LSGO_Dataset_torch.get_datasets(train_batch_size = config.train_batch_size,
                                                 test_batch_size = config.test_batch_size,
                                                 difficulty = config.difficulty)

    elif problem in ['lsgo']:
        return LSGO_Dataset_numpy.get_datasets(train_batch_size = config.train_batch_size,
                                                 test_batch_size = config.test_batch_size,
                                                 difficulty = config.difficulty)
    elif problem in ['uav']:
        return UAV_Dataset_numpy.get_datasets(train_batch_size = config.train_batch_size,
                                              test_batch_size = config.test_batch_size,
                                              dv = 10,
                                              j_pen = 1e4,
                                              mode = "standard",
                                              num = 56,
                                              difficulty = config.difficulty)
    elif problem in ['uav-torch']:
        return UAV_Dataset_torch.get_datasets(train_batch_size = config.train_batch_size,
                                              test_batch_size = config.test_batch_size,
                                              dv = 10,
                                              j_pen = 1e4,
                                              mode = "standard",
                                              num = 56,
                                              difficulty = config.difficulty)
    elif problem in ['MMO', 'MMO-torch']:
        return mmo_dataset.MMO_Dataset.get_datasets(version=problem,
                                            train_batch_size=config.train_batch_size,
                                            test_batch_size=config.test_batch_size,
                                            difficulty=config.difficulty,
                                            user_train_list = config.user_train_list)

    else:
        raise ValueError(problem + ' is not defined!')
    
    
class Ma_Moo_Trainer:
    def __init__(self, config):
        self.config = config
        if config.resume_dir is None:
            self.agent = eval(config.train_agent)(config)
        else:
            file_path = config.resume_dir + config.train_agent + '.pkl'
            with open(file_path, 'rb') as f:
                self.agent = pickle.load(f)
            self.agent.update_setting(config)
        self.optimizer = eval(config.train_optimizer)(config)
        self.train_set, self.test_set = construct_problem_set(config)
        self.indicators = config.indicators

    def save_log(self, epochs, steps, indicators_record, returns):
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/log/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return_save = np.stack((steps, returns),  0)
        np.save(log_dir+'return', return_save)
        for problem in self.train_set:
            name = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
            for indicator in self.indicators:
                if len(indicators_record[name][indicator]) == 0:
                    continue
                while len(indicators_record[name][indicator]) < len(epochs):
                    indicators_record[name][indicator].append(indicators_record[name][indicator][-1])
                indictors_record_save = np.stack((epochs, indicators_record[name][indicator]),  0)
                np.save(log_dir+name+'_'+indicator,indictors_record_save)
            
    def draw_indicators(self, Name=None, normalize=False):
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        for problem in self.train_set:
            if Name is None:
                name = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
            elif (isinstance(Name, str) and problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var) != Name) \
                or (isinstance(Name, list) and problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var) not in Name):
                continue
            else:
                name = Name
            for indicator in self.indictors:
                plt.figure()
                plt.title(name + '_' + indicator)
                values = np.load(log_dir + 'log/' + name+'_'+indicator+'.npy')
                x, y, n = values
                if normalize:
                    y /= n
                plt.plot(x, y)
                plt.savefig(log_dir+f'pic/{name}_'+indicator+'.png')
                plt.close()
    
    def draw_average_indictors(self, normalize=True):
        # 这个函数用于画每一个epoch所有问题指标的平均值
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        for indictor in self.indicators:
            X = []
            Y = []
            for problem in self.train_set:
                name = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
                values = np.load(log_dir + 'log/' + name+'_'+indictor+'.npy')
                x, y = values
                X.append(x)
                Y.append(y)
            X = np.mean(X, 0)
            Y = np.mean(Y, 0)
            plt.figure()
            plt.title('all problem '+ indictor)
            plt.plot(X, Y)
            plt.savefig(log_dir+f'pic/all_problem_'+indictor+'.png')
            plt.close()

    def draw_return(self):
        log_dir = self.config.log_dir + f'/train/{self.agent.__class__.__name__}/{self.config.run_time}/'
        if not os.path.exists(log_dir + 'pic/'):
            os.makedirs(log_dir + 'pic/')
        plt.figure()
        plt.title('return')
        values = np.load(log_dir + 'log/return.npy')
        plt.plot(values[0], values[1])
        plt.savefig(log_dir+f'pic/return.png')
        plt.close()

    def train(self):
        print(f'start training: {self.config.run_time}')
        if self.indictors is None:
            self.indicators = ['best_igd','best_hv']
        indictors_record = {}
        for problem in self.train_set:
            problem_name = problem.__class__.__name__+"_n"+str(problem.n_obj)+"_m"+str(problem.n_var)
            indictors_record[problem_name] = {}
            for indictor in self.indictors:
                indictors_record[problem_name][indictor] = []
        is_end = False
        epoch = 0
        return_record = []
        learn_steps = []
        epoch_steps = []
        q_values_record = []
        loss_record = []

        # # 这里先让train_set bs 一直为1先
        # for problem in self.train_set.data:
        #     cost_record[problem.__str__()] = []
        #     normalizer_record[problem.__str__] = []

        # 然后根据train_mode 决定 bs
        # single ---> 从train_set 里取出 bs 个问题训练
        # multi ---> 每次从train_set 中取出 1 个问题，copy bs 个 训练
        bs = self.config.train_batch_size
        if self.config.train_mode == "single":
            self.train_set.batch_size = 1
        elif self.config.train_mode == "multi":
            self.train_set.batch_size = bs

        epoch_seed = self.config.epoch_seed
        id_seed = self.config.id_seed
        seed = self.config.seed

        while not is_end:
            learn_step = 0
            self.train_set.shuffle()
            with tqdm(range(self.train_set.N), desc = f'Training {self.agent.__class__.__name__} Epoch {epoch}') as pbar:
                for problem_id, problem in enumerate(self.train_set):
                    # set seed
                    seed_list = (epoch * epoch_seed + id_seed * (np.arange(bs) + bs * problem_id) + seed).tolist()

                    # 这里前面已经判断好 train_mode，这里只需要根据 train_mode 构造env就行
                    if self.config.train_mode == "single":
                        env_list = [PBO_Env(copy.deepcopy(problem), copy.deepcopy(self.optimizer)) for _ in range(bs)] # bs
                    elif self.config.train_mode == "multi":
                        env_list = [PBO_Env(copy.deepcopy(p), copy.deepcopy(self.optimizer)) for p in problem] # bs

                    # todo config add para
                    exceed_max_ls, train_meta_data = self.agent.train_episode(envs = env_list,
                                                                              seeds = seed_list,
                                                                              para_mode = "dummy",
                                                                              asynchronous = None,
                                                                              num_cpus = 1,
                                                                              num_gpus = 0,
                                                                              )
                    # exceed_max_ls, pbar_info_train = self.agent.train_episode(env)  # pbar_info -> dict
                    postfix_str = (
                        f"loss={train_meta_data['loss']:.2e}, "
                        f"learn_steps={train_meta_data['learn_steps']}, "
                        f"return={[f'{x:.2e}' for x in train_meta_data['return']]}"
                    )

                    pbar.set_postfix_str(postfix_str)
                    pbar.update(self.train_set.batch_size)
                    learn_step = train_meta_data['learn_steps']
                    for indictor in self.indictors:
                        indictors_record[problem_name][indictor].append(train_meta_data[indictor])
                    return_record.append(train_meta_data['return'])
                    learn_steps.append(learn_step)
                    q_values_record.append(train_meta_data['q_values'])
                    loss_record.append(train_meta_data['loss'])
                    # for id, p in enumerate(problem):
                    #     name = p.__str__()
                    #     cost_record[name].append(train_meta_data['gbest'][id])
                    #     normalizer_record[name].append(train_meta_data['normalizer'][id])
                    #     return_record.append(np.mean(train_meta_data['return']))
                    # learn_steps.append(learn_step)

                    if self.config.end_mode == "step" and exceed_max_ls:
                        is_end = True
                        break
                self.agent.train_epoch()
            epoch_steps.append(learn_step)
            epoch += 1

            # todo save
            # save_interval = 5
            # checkpoint0 0
            # checkpoint1 5
            if epoch >= (self.config.save_interval * self.agent.cur_checkpoint) and self.config.end_mode == "epoch":
                save_class(self.config.agent_save_dir, 'checkpoint' + str(self.agent.cur_checkpoint), self.agent)
                # 记录 checkpoint 和 total_step
                with open(self.config.agent_save_dir + "/checkpoint_log.txt", "a") as f:
                    f.write(f"Checkpoint {self.agent.cur_checkpoint}: {learn_step}\n")

                self.agent.cur_checkpoint += 1
            if self.config.end_mode == "epoch" and epoch >= self.config.max_epoch:
                is_end = True
            # if not os.path.exists(agent_save_dir):
            #     os.makedirs(agent_save_dir)
            # with open(agent_save_dir+'agent_epoch'+str(epoch)+'.pkl', 'wb') as f:
            #     pickle.dump(self.agent, f, -1)
            self.save_log(epoch_steps, learn_steps, indictors_record, return_record)
            epoch += 1
            if epoch % self.config.draw_interval == 0:
                self.draw_indictors()
                self.draw_average_indictors()
                self.draw_return()
        self.draw_indictors()
        self.draw_average_indictors()
        self.draw_return()
        

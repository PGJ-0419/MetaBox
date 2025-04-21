import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, Union, Callable
import argparse
params = {
    'axes.labelsize': '25',
    'xtick.labelsize': '25',
    'ytick.labelsize': '25',
    'lines.linewidth': '3',
    'legend.fontsize': '24',
    'figure.figsize': '20,11',
}
plt.rcParams.update(params)

markers = ['o', '^', '*', 'O', 'v', 'x', 'X', 'd', 'D', '.', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H']
colors = ['b', 'g', 'orange', 'r', 'purple', 'brown', 'grey', 'limegreen', 'turquoise', 'olivedrab', 'royalblue', 'darkviolet', 
          'chocolate', 'crimson', 'teal','seagreen', 'navy', 'deeppink', 'maroon', 'goldnrod', 
          ]


# def data_wrapper_prsr(data, ):
#     res = []
#     for key in data.keys():
#         res.append(np.array(data[key][:, -1, 3]))
#     return np.array(res)


def data_wrapper_cost(data, ):
    return np.array(data)[:, :, -1]


# def data_wrapper_prsr_test(data, ):
#     return np.array(data)[:,-1, 3]


def to_label(agent_name: str) -> str:
    label = agent_name
    if label == 'L2L_Agent':
        return 'RNN-OI'
    if len(label) > 6 and (label[-6:] == '_Agent' or label[-6:] == '_agent'):
        label = label[:-6]
    return label


class Basic_Logger:
    def __init__(self, config: argparse.Namespace) -> None:
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0

    def get_average_data(self, results: dict, norm: bool=False, data_wrapper: Callable = None):
        """
        Get the average and standard deviation of each agent from the results
        :param results  dict: The data to be process
        :param norm     bool: Whether to min-max normalize data
        :param data_wrapper callable: A data pre-processing function wrapper applied to each data item of each agent under each problem
        """
        problems=[]
        agents=[]

        for problem in results.keys():
            problems.append(problem)
        for agent in results[problems[0]].keys():
            agents.append(agent)
        avg_data={}
        std_data={}
        for agent in agents:
            avg_data[agent]=[]
            std_data[agent]=[]
            for problem in problems:
                values = results[problem][agent]
                if data_wrapper is not None:
                    values = data_wrapper(values)
                if norm:
                    values = (values - np.min((values))) / (np.max(values) - np.min(values))
                std_data[agent].append(np.std(values, -1))
                avg_data[agent].append(np.mean(values, -1))
            avg_data[agent] = np.mean(avg_data[agent], 0)
            std_data[agent] = np.mean(std_data[agent], 0)
        return avg_data, std_data

    def data_wrapper_cost_rollout(self, data):
        res = np.array(data)
        return res[:, -1]

    def cal_scores1(self, D: dict, maxf: float):
        """
        Tool function for CEC metric
        """
        SNE = []
        for agent in D.keys():
            values = D[agent]
            sne = 0.5 * np.sum(np.min(values, -1) / maxf)
            SNE.append(sne)
        SNE = np.array(SNE)
        score1 = (1 - (SNE - np.min(SNE)) / SNE) * 50
        return score1

    def get_random_baseline(self, results: dict, fes: Optional[Union[int, float]]):
        """
        Get the results of Random Search for further usage, i.e., for normalization
        """
        baseline = {}
        T1 = []
        T2 = []
        for pname in results['T1'].keys():
            T1.append(results['T1'][pname]['Random_search'])
            T2.append(results['T2'][pname]['Random_search'])
        baseline['complexity_avg'] = np.mean(np.log10(1. / (np.array(T2) - np.array(T1)) / results['T0']))
        baseline['complexity_std'] = np.std(np.log10(1. / (np.array(T2) - np.array(T1)) / results['T0']))
        avg = []
        std = []
        for problem in results['fes'].keys():
            g = np.log10(fes/np.array(results['fes'][problem]['Random_search']))
            avg.append(g.mean())
            std.append(g.std())
        baseline['fes_avg'] = np.mean(avg)
        baseline['fes_std'] = np.mean(std)
        avg = []
        std = []
        for problem in results['cost'].keys():
            g = np.log10(1/(np.array(results['cost'][problem]['Random_search'])[:, -1]+1))
            avg.append(g.mean())
            std.append(g.std()) 
        baseline['cost_avg'] = np.mean(avg)
        baseline['cost_std'] = np.mean(std)
        return baseline

    def gen_algorithm_complexity_table(self, results: dict, out_dir: str) -> None:
        """
        Store algorithm complexity data as excel table 
        """
        save_list=[]
        t0=results['T0']
        ratios=[]
        t1_list = {}
        t2_list = {}
        indexs=list(results['T1'][list(results['T1'].keys())[0]].keys())
        columns=['T0','T1','T2','(T2-T1)/T0']
        for agent in indexs:
            t1_list[agent] = []
            t2_list[agent] = []
            for pname in results['T1'].keys():
                t1_list[agent].append(results['T1'][pname][agent])
                t2_list[agent].append(results['T2'][pname][agent])
            t1_list[agent] = np.mean(t1_list[agent])
            t2_list[agent] = np.mean(t2_list[agent])
            ratios.append((t2_list[agent] - t1_list[agent])/t0)

        n=len(indexs)
        data=np.zeros((n,4))
        data[:,0]=t0
        data[:,1]=list(t1_list.values())
        data[:,2]=list(t2_list.values())
        data[:,3]=ratios
        table=pd.DataFrame(data=np.round(data,2),index=indexs,columns=columns)
        table.to_excel(os.path.join(out_dir,'algorithm_complexity.xlsx'))

    def gen_agent_performance_table(self, results: dict, out_dir: str) -> None:
        """
        Store the `Worst`, `Best`, `Median`, `Mean` and `Std` of cost results of each agent as excel
        """
        total_cost=results['cost']
        table_data={}
        indexs=[]
        columns=['Worst','Best','Median','Mean','Std']
        for problem,value in total_cost.items():
            indexs.append(problem)
            problem_cost=value
            for alg,alg_cost in problem_cost.items():
                n_cost=[]
                for run in alg_cost:
                    n_cost.append(run[-1])
                # if alg == 'MadDE' and problem == 'F5':
                #     for run in alg_cost:
                #         print(len(run))
                #     print(len(n_cost))
                best=np.min(n_cost)
                best=np.format_float_scientific(best,precision=3,exp_digits=3)
                worst=np.max(n_cost)
                worst=np.format_float_scientific(worst,precision=3,exp_digits=3)
                median=np.median(n_cost)
                median=np.format_float_scientific(median,precision=3,exp_digits=3)
                mean=np.mean(n_cost)
                mean=np.format_float_scientific(mean,precision=3,exp_digits=3)
                std=np.std(n_cost)
                std=np.format_float_scientific(std,precision=3,exp_digits=3)

                if not alg in table_data:
                    table_data[alg]=[]
                table_data[alg].append([worst,best,median,mean,std])
        for alg,data in table_data.items():
            dataframe=pd.DataFrame(data=data,index=indexs,columns=columns)
            #print(dataframe)
            dataframe.to_excel(os.path.join(out_dir,f'{alg}_concrete_performance_table.xlsx'))

    def gen_overall_tab(self, results: dict, out_dir: str) -> None:
        """
        Store the overall results inculding `objective values` (costs), `gap` with CMAES and the consumed `FEs` as excel
        """
        # get multi-indexes first
        problems = []
        statics = ['Obj','Gap','FEs']
        optimizers = []
        for problem in results['cost'].keys():
            problems.append(problem)
        for optimizer in results['cost'][problems[0]].keys():
            optimizers.append(optimizer)
        multi_columns = pd.MultiIndex.from_product(
            [problems,statics], names=('Problem', 'metric')
        )
        df_results = pd.DataFrame(np.ones(shape=(len(optimizers),len(problems)*len(statics))),
                                index=optimizers,
                                columns=multi_columns)

        # calculate baseline1 cmaes
        cmaes_obj = {}
        for problem in problems:
            blobj_problem = results['cost'][problem]['CMAES']  # 51 * record_length
            objs = []
            for run in range(self.config.test_run):
                objs.append(blobj_problem[run][-1])
            cmaes_obj[problem] = sum(objs) / self.config.test_run

        # calculate baseline2 random_search
        rs_obj = {}
        for problem in problems:
            blobj_problem = results['cost'][problem]['Random_search']  # 51 * record_length
            objs = []
            for run in range(self.config.test_run):
                objs.append(blobj_problem[run][-1])
            rs_obj[problem] = sum(objs) / self.config.test_run

        # calculate each Obj
        for problem in problems:
            for optimizer in optimizers:
                obj_problem_optimizer = results['cost'][problem][optimizer]
                objs_ = []
                for run in range(self.config.test_run):
                    objs_.append(obj_problem_optimizer[run][-1])
                avg_obj = sum(objs_)/self.config.test_run
                std_obj = np.std(objs_)
                df_results.loc[optimizer, (problem, 'Obj')] = np.format_float_scientific(avg_obj, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_obj, precision=3, exp_digits=1) + ")"
                # calculate each Gap
                df_results.loc[optimizer, (problem, 'Gap')] = "%.3f" % (1-(rs_obj[problem]-avg_obj) / (rs_obj[problem]-cmaes_obj[problem]+1e-10))
                fes_problem_optimizer = np.array(results['fes'][problem][optimizer])
                avg_fes = np.mean(fes_problem_optimizer)
                std_fes = np.std(fes_problem_optimizer)
                df_results.loc[optimizer, (problem, 'FEs')] = np.format_float_scientific(avg_fes, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_fes, precision=3, exp_digits=1) + ")"
        df_results.to_excel(out_dir+'overall_table.xlsx')

    def aei_cost(self, cost_data: dict, baseline: dict, ignore: Optional[list]=None):
        avg = baseline['cost_avg']
        problems = cost_data.keys()
        agents = cost_data[list(problems)[0]].keys()
        results_cost = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            costs_problem = []
            for problem in problems:
                cost_ = np.log10(1/(np.array(cost_data[problem][agent])[:, -1]+1))
                costs_problem.append(cost_.mean())
            results_cost[agent] = np.exp((costs_problem - avg) * 1)
        aei_mean, aei_std = self.cal_aei(results_cost, agents, ignore)
        return results_cost, aei_mean, aei_std
    
    def aei_fes(self, fes_data: dict, baseline: dict, maxFEs: Optional[Union[int, float]]=20000, ignore: Optional[list]=None):
        avg = baseline['fes_avg']
        problems = fes_data.keys()
        agents = fes_data[list(problems)[0]].keys()
        results_fes = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            fes_problem = []
            for problem in problems:
                if agent == 'L2L':
                    fes_ = np.log10(100/np.array(fes_data[problem][agent]))
                else:
                    fes_ = np.log10(maxFEs/np.array(fes_data[problem][agent]))
                fes_problem.append(fes_.mean())
            results_fes[agent] = np.exp((fes_problem - avg) * 1)
        aei_mean, aei_std = self.cal_aei(results_fes, agents, ignore)
        return results_fes, aei_mean, aei_std
    
    def aei_complexity(self, complexity_data: dict, baseline: dict, ignore: Optional[list]=None):
        avg = baseline['complexity_avg']
        std = baseline['complexity_std']
        problems = complexity_data['T1'].keys()
        agents = complexity_data['T1'][list(problems)[0]].keys()
        results_complex = {}
        complexity_data['complexity'] = {}
        for key in agents:
            if (ignore is not None) and (key in ignore):
                continue
            if key not in complexity_data['complexity'].keys():
                t0 = complexity_data['T0']
                t1 = np.array([complexity_data['T1'][pname][key] for pname in problems])
                t2 = np.array([complexity_data['T2'][pname][key] for pname in problems])
                complexity_data['complexity'][key] = np.mean((t2 - t1) / t0)
            results_complex[key] = np.exp((np.log10(1/complexity_data['complexity'][key]) - avg)/std/1000 * 1)
        aei_mean, aei_std = self.cal_aei(results_complex, agents, ignore)
        return results_complex, aei_mean, aei_std

    def cal_aei(self, results: dict, agents: dict, ignore: Optional[list]=None):
        mean = {}
        std = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            if agent == 'Random_search':
                continue
            aei_k = results[agent]
            mean[agent] = np.mean(aei_k)
            if self.config.test_problem in ['protein', 'protein-torch']:
                std[agent] = np.std(aei_k) * 5.
            else:
                std[agent] = np.std(aei_k) / 5.
        return mean, std

    def aei_metric(self, data: dict, maxFEs: Optional[Union[int, float]]=20000, ignore: Optional[list]=None):
        """
        Calculate the AEI metric
        """
        baseline = self.get_random_baseline(data, maxFEs)
        problems = data['cost'].keys()
        agents = data['cost'][list(problems)[0]].keys()
        
        results_cost, aei_cost_mean, aei_cost_std = self.aei_cost(data['cost'], baseline, ignore)
        results_fes, aei_fes_mean, aei_fes_std = self.aei_fes(data['fes'], baseline, maxFEs, ignore)
        results_complex, aei_clx_mean, aei_clx_std = self.aei_complexity(data, baseline, ignore)
        
        mean = {}
        std = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            if agent == 'Random_search':
                continue
            aei_k = results_complex[agent] * results_cost[agent] * results_fes[agent]
            mean[agent] = np.mean(aei_k)
            if self.config.test_problem in ['protein', 'protein-torch']:
                std[agent] = np.std(aei_k) * 5.
            else:
                std[agent] = np.std(aei_k) / 5.
        return {'mean': mean, 'std': std}

    def cec_metric(self, data: dict, ignore: Optional[list]=None):
        """
        Calculate the metric adopted in CEC
        """
        score = {}
        M = []
        X = []
        Y = []
        R = []
        data, fes = data['cost'], data['fes']
        for problem in list(data.keys()):
            maxf = 0
            avg_cost = []
            avg_fes = []
            for agent in list(data[problem].keys()):
                if ignore is not None and agent in ignore:
                    continue
                key = to_label(agent)
                if key not in score.keys():
                    score[key] = []
                values = np.array(data[problem][agent])[:, -1]
                score[key].append(values)
                maxf = max(maxf, np.max(values))
                avg_cost.append(np.mean(values))
                avg_fes.append(np.mean(fes[problem][agent]))

            M.append(maxf)
            order = np.lexsort((avg_fes, avg_cost))
            rank = np.zeros(len(avg_cost))
            rank[order] = np.arange(len(avg_cost)) + 1
            R.append(rank)
        sr = 0.5 * np.sum(R, 0)
        score2 = (1 - (sr - np.min(sr)) / sr) * 50
        score1 = self.cal_scores1(score, M)
        for i, key in enumerate(score.keys()):
            score[key] = score1[i] + score2[i]
        return score

    def draw_ECDF(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, pdf_fig: bool = True):
        data = data['cost']
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            plt.figure()
            for agent in list(data[name].keys()):
                if agent not in self.color_arrangement.keys():
                    self.color_arrangement[agent] = colors[self.arrange_index]
                    self.arrange_index += 1
                values = np.array(data[name][agent])[:, -1]
                plt.ecdf(values, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
            plt.grid()
            plt.xlabel('costs')
            plt.legend()
            fig_type = 'pdf' if pdf_fig else 'png'
            plt.savefig(output_dir + f'ECDF_{problem}.{fig_type}', bbox_inches='tight')

    def draw_covergence_curve(self, agent: str, problem: str, metadata_dir: str, output_dir: str, pdf_fig: bool = True):
        def cal_max_distance(X):
            X = np.array(X)
            return np.max(np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, -1)))
        with open(metadata_dir + f'/{problem}.pkl', 'rb') as f:
            metadata = pickle.load(f)[agent]
        plt.figure()
        Xs = []
        n_generations = int(1e9)
        for item in metadata:
            Xs.append(item['X'])
            n_generations = min(n_generations, len(item['X']))
        diameter = np.zeros(n_generations)
        std = np.zeros(n_generations)
        x_axis = np.arange(n_generations)
        for i in range(n_generations):  # episode length
            d = []
            for j in range(len(Xs)):  # test_run
                d.append(cal_max_distance(Xs[j][i]))
            diameter[i] = np.mean(d)
            std[i] = np.std(d)
        plt.plot(x_axis, diameter, marker='*', markersize=12, markevery=2, c=self.color_arrangement[agent])
        plt.fill_between(x_axis, (diameter - std), (diameter + std), alpha=0.2, facecolor=self.color_arrangement[agent])
        plt.grid()
        plt.xlabel('Optimization Generations')    
        plt.ylabel('Population Diameter')
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'convergence_curve_{agent}_{problem}.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_test_data(self, data: dict, data_type: str, output_dir: str, Name: Optional[Union[str, list]]=None, logged: bool=False, categorized: bool=False, pdf_fig: bool = True, data_wrapper: Callable = None) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            # if logged:
            #     plt.title('log cost curve ' + name)
            # else:
            #     plt.title('cost curve ' + name)
            if not categorized:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = data[name][agent]
                    if data_wrapper is not None:
                        values = data_wrapper(values)
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                if logged:
                    plt.ylabel(f'log {data_type}')
                    plt.savefig(output_dir + f'{name}_log_{data_type}_curve.{fig_type}', bbox_inches='tight')
                else:
                    plt.ylabel(data_type)
                    plt.savefig(output_dir + f'{name}_{data_type}_curve.{fig_type}', bbox_inches='tight')
                plt.close()
            else:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.agent:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = data[name][agent]
                    if data_wrapper is not None:
                        values = data_wrapper(values)
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                if logged:
                    plt.ylabel(f'log {data_type}')
                    plt.savefig(output_dir + f'learnable_{name}_log_{data_type}_curve.{fig_type}', bbox_inches='tight')
                else:
                    plt.ylabel(data_type)
                    plt.savefig(output_dir + f'learnable_{name}_{data_type}_curve.{fig_type}', bbox_inches='tight')
                plt.close()

                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.t_optimizer:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = data[name][agent]
                    if data_wrapper is not None:
                        values = data_wrapper(values)
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                
                plt.legend()
                if logged:
                    plt.ylabel(f'log {data_type}')
                    plt.savefig(output_dir + f'classic_{name}_log_{data_type}_curve.{fig_type}', bbox_inches='tight')
                else:
                    plt.ylabel(data_type)
                    plt.savefig(output_dir + f'classic_{name}_{data_type}_curve.{fig_type}', bbox_inches='tight')
                plt.close()
    
    def draw_named_average_test_costs(self, data: dict, output_dir: str, named_agents: dict, logged: bool=False, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        fig = plt.figure(figsize=(50, 10))
        # plt.title('all problem cost curve')
        plots = len(named_agents.keys())
        for id, title in enumerate(named_agents.keys()):
            ax = plt.subplot(1, plots+1, id+1)
            ax.set_title(title, fontsize=25)
            
            Y = {}
            for problem in list(data.keys()):
                for agent in list(data[problem].keys()):
                    if agent not in named_agents[title]:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if agent not in Y.keys():
                        Y[agent] = {'mean': [], 'std': []}
                    values = np.array(data[problem][agent])
                    values /= values[:, 0].repeat(values.shape[-1]).reshape(values.shape)
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    Y[agent]['mean'].append(mean)
                    Y[agent]['std'].append(std)

            for id, agent in enumerate(list(Y.keys())):
                mean = np.mean(Y[agent]['mean'], 0)
                std = np.mean(Y[agent]['std'], 0)

                X = np.arange(mean.shape[-1])
                X = np.array(X, dtype=np.float64)
                X *= (self.config.maxFEs / X[-1])
                # X = np.log10(X)
                # X[0] = 0

                ax.plot(X, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                ax.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=self.color_arrangement[agent])
            plt.grid()
            # plt.xlabel('log10 FEs')
            plt.xlabel('FEs')
            plt.ylabel('Normalized Costs')
            plt.legend()
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels, bbox_to_anchor=(plots/(plots+1)-0.02, 0.5), borderaxespad=0., loc=6, facecolor='whitesmoke')
        
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
        plt.savefig(output_dir + f'all_problem_cost_curve_logX.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_concrete_performance_hist(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        D = {}
        X = []
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D.keys():
                    D[agent] = []
                values = np.array(data[name][agent])
                D[agent].append(values[:, -1] / values[:, 0])

        for agent in D.keys():
            plt.figure()
            # plt.title(f'{agent} performance histgram')
            X = list(data.keys())
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])
            for a,b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)
            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            plt.ylabel('Normalized Costs')
            plt.savefig(output_dir + f'{agent}_concrete_performance_hist.{fig_type}', bbox_inches='tight')

    def draw_boxplot(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        data = data['cost']
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            Y = []
            X = []
            plt.figure(figsize=(30, 15))
            for agent in list(data[name].keys()):
                if ignore is not None and agent in ignore:
                    continue
                X.append(agent)
                values = np.array(data[name][agent])
                Y.append(values[:, -1])
            Y = np.transpose(Y)
            plt.boxplot(Y, labels=X, showmeans=True, patch_artist=True, showfliers=False,
                        medianprops={'color': 'green', 'linewidth': 3}, 
                        meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                        boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        )
            plt.xticks(rotation=30, fontsize=18)
            plt.xlabel('Agents')
            plt.ylabel(f'{name} Cost Boxplots')
            plt.savefig(output_dir + f'{name}_boxplot.{fig_type}', bbox_inches='tight')
            plt.close()

    def draw_overall_boxplot(self, data: dict, output_dir: str, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        problems=[]
        agents=[]
        for problem in data.keys():
            problems.append(problem)
        for agent in data[problems[0]].keys():
            if ignore is not None and agent in ignore:
                continue
            agents.append(agent)
        run = len(data[problems[0]][agents[0]])
        values = np.zeros((len(agents), len(problems), run))
        plt.figure(figsize=(30, 15))
        for ip, problem in enumerate(problems):
            for ia, agent in enumerate(agents):
                values[ia][ip] = np.array(data[problem][agent])[:, -1]
            values[:, ip, :] = (values[:, ip, :] - np.min(values[:, ip, :])) / (np.max(values[:, ip, :]) - np.min(values[:, ip, :]))
        values = values.reshape(len(agents), -1).transpose()
        
        plt.boxplot(values, labels=agents, showmeans=True, patch_artist=True, showfliers=False,
                    medianprops={'color': 'green', 'linewidth': 3}, 
                    meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                    boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                    capprops={'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    )
        plt.xticks(rotation=30, fontsize=18)
        plt.xlabel('Agents')
        plt.ylabel('Cost Boxplots')
        plt.savefig(output_dir + f'overall_boxplot.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_rank_hist(self, data: dict, random: dict, output_dir: str, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        metric, metric_std = self.aei_metric(data, random, maxFEs=self.config.maxFEs, ignore=ignore)
        X, Y = list(metric.keys()), list(metric.values())
        _, S = list(metric_std.keys()), list(metric_std.values())
        n_agents = len(X)
        for i in range(n_agents):
            X[i] = to_label(X[i])

        plt.figure(figsize=(4*n_agents,15))
        plt.bar(X, Y)
        plt.errorbar(X, Y, S, fmt='s', ecolor='dimgray', ms=1, color='dimgray', elinewidth=5, capsize=30, capthick=5)
        for a,b in zip(X, Y):
            plt.text(a, b+0.05, '%.2f' % b, ha='center', fontsize=55)
        plt.xticks(rotation=45, fontsize=60)
        plt.yticks(fontsize=60)
        plt.ylim(0, np.max(np.array(Y) + np.array(S)) * 1.1)
        plt.title(f'The AEI for {self.config.dim}D {self.config.problem}-{self.config.difficulty}', fontsize=70)
        plt.ylabel('AEI', fontsize=60)
        plt.savefig(output_dir + f'rank_hist.{fig_type}', bbox_inches='tight')
        
    def draw_train_logger(self, data_type: str, steps: list, data: dict, output_dir: str, ylabel: str = None, norm: bool = False, pdf_fig: bool = True, data_wrapper: Callable = None) -> None:
        means, stds = self.get_average_data(data, norm=norm, data_wrapper=data_wrapper)
        plt.figure()

        y = np.array([means[k] for k in means])
        y_std = np.array([stds[k] for k in stds])
        x = np.array(steps, dtype = np.float64)

        s = np.zeros(y.shape[0])
        a = s[0] = y[0]

        agent_for_rollout = self.config.agent_for_rollout

        norm = self.config.plot_smooth + 1
        for i in range(1, y.shape[0]):
            a = a * self.config.plot_smooth + y[i]
            s[i] = a / norm if norm > 0 else a
            norm *= self.config.plot_smooth
            norm += 1
        if agent_for_rollout not in self.color_arrangement.keys():
            self.color_arrangement[agent_for_rollout] = colors[self.arrange_index]
            self.arrange_index += 1

        plt.plot(x, s, label = to_label(agent_for_rollout), marker = '*', markersize = 12, markevery = 2, c = self.color_arrangement[agent_for_rollout])
        plt.fill_between(x, (s - y_std), (s + y_std), alpha = 0.2, facecolor = self.color_arrangement[agent_for_rollout])

        plt.legend()
        plt.grid()
        plt.xlabel('Learning Steps')
        if ylabel is None:
            ylabel = data_type
        plt.ylabel(ylabel)
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'avg_{data_type}_curve.{fig_type}', bbox_inches='tight')
        plt.close()


        # if agent not in self.color_arrangement.keys():
        #     self.color_arrangement[agent] = colors[self.arrange_index]
        #     self.arrange_index += 1
        # plt.plot(x, s, label = to_label(agent), marker = '*', markersize = 12, markevery = 2, c = self.color_arrangement[agent])
        # plt.fill_between(x, (s - stds[agent]), (s + stds[agent]), alpha = 0.2, facecolor = self.color_arrangement[agent])


        # for agent in means.keys():
        #     x = np.arange(len(means[agent]), dtype=np.float64)
        #     x = (self.config.maxFEs / x[-1]) * x
        #     y = means[agent]
        #     s = np.zeros(y.shape[0])
        #     a = s[0] = y[0]
        #     norm = self.config.plot_smooth + 1
        #     for i in range(1, y.shape[0]):
        #         a = a * self.config.plot_smooth + y[i]
        #         s[i] = a / norm if norm > 0 else a
        #         norm *= self.config.plot_smooth
        #         norm += 1
        #     if agent not in self.color_arrangement.keys():
        #         self.color_arrangement[agent] = colors[self.arrange_index]
        #         self.arrange_index += 1
        #     plt.plot(x, s, label=to_label(agent), marker='*', markersize=12, markevery=2, c=self.color_arrangement[agent])
        #     plt.fill_between(x, (s - stds[agent]), (s + stds[agent]), alpha=0.2, facecolor=self.color_arrangement[agent])
        #     # plt.plot(x, returns[agent], label=to_label(agent))
    def post_processing_test_statics(self, log_dir: str, include_random_baseline: bool = True, pdf_fig: bool = True) -> None:
        print('Post processing & drawing')
        with open(log_dir + 'test_results.pkl', 'rb') as f:
            results = pickle.load(f)
            
        metabbo = self.config.agent
        bbo = self.config.t_optimizer
        
        if not os.path.exists(log_dir + 'tables/'):
            os.makedirs(log_dir + 'tables/')

        self.gen_overall_tab(results, log_dir + 'tables/')
        self.gen_algorithm_complexity_table(results, log_dir + 'tables/')

        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')

        # 如果需要，可以为不同的算法绘制图形（例如 cost 图）
        if 'cost' in results:
            self.draw_test_data(results['cost'], 'cost', log_dir + 'pics/', logged=True, categorized=True, pdf_fig=pdf_fig, data_wrapper=np.array)
            self.draw_named_average_test_costs(results['cost'], log_dir + 'pics/',
                                                {'MetaBBO-RL': metabbo,
                                                'Classic Optimizer': bbo},
                                                logged=False, pdf_fig=pdf_fig)
            self.draw_ECDF(results, log_dir + 'pics/', pdf_fig=pdf_fig)
            self.draw_boxplot(results, log_dir + 'pics/', pdf_fig=pdf_fig)
            with open(log_dir + 'aei.pkl', 'wb') as f:
                pickle.dump(self.aei_metric(results, self.config.maxFEs), f)

    def post_processing_rollout_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        print('Post processing & drawing')
        with open(log_dir+'rollout.pkl', 'rb') as f:
            results = pickle.load(f)
        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')
        self.draw_train_logger('return', results['steps'], results['return'], log_dir + 'pics/', pdf_fig=pdf_fig)
        self.draw_train_logger('cost', results['steps'], results['cost'], log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper = Basic_Logger.data_wrapper_cost_rollout)

    
class MOO_Logger(Basic_Logger):
    def __init__(self, config: argparse.Namespace) -> None:
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0
        self.indicators = config.indicators
    
    def is_pareto_efficient(self,points):
        """计算帕累托前沿"""
        points = np.array(points)
        pareto_mask = np.ones(points.shape[0], dtype=bool)
        for i, p in enumerate(points):
            if pareto_mask[i]:
                pareto_mask[pareto_mask] = np.any(points[pareto_mask] < p, axis=1)
                pareto_mask[i] = True
        return points[pareto_mask]
    
    def draw_pareto_fronts(self,data: dict, output_dir: str, Name: Optional[Union[str, list]] = None):
        # 输入的数据格式为：dict[problem][algo][run][generation][objective]
        
        for problem in list(data.keys()):
            if Name is not None and ((isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name)):
                continue
            else:
                name = problem

            fig = plt.figure(figsize=(8, 6))  # 更小的画布尺寸
            is_3d = False
            algo_obj_dict = {}

            # 收集每个算法所有回合的最后一代目标值
            for algo, runs in data[problem].items():
                all_obj_values = []
                for generations in runs:
                    last_gen = np.array(generations[-1])
                    obj_values = last_gen.reshape(-1, last_gen.shape[-1])
                    if obj_values.shape[1] == 3:
                        is_3d = True
                    all_obj_values.append(obj_values)
                algo_obj_dict[algo] = np.vstack(all_obj_values)

            # 初始化画布
            if is_3d:
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=40, azim=135)  # 更改视角
                ax.set_proj_type('persp')
            else:
                ax = fig.add_subplot(111)

            colors = ['r', 'g', 'b', 'c', 'm', 'y']

            for algo_idx, (algo, obj_values) in enumerate(algo_obj_dict.items()):
                pareto_front = self.is_pareto_efficient(obj_values)
                color = colors[algo_idx % len(colors)]
                label = f"{algo}"

                if obj_values.shape[1] == 2:
                    ax.scatter(pareto_front[:, 0], pareto_front[:, 1],
                                label=label, color=color, edgecolors='k')
                elif obj_values.shape[1] == 3:
                    ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                                label=label, color=color, edgecolors='k')

            if is_3d:
                # 更改坐标轴标签为简写，并减小Z轴标签字体
                ax.set_xlabel('X', fontsize=12, labelpad=10)
                ax.set_ylabel('Y', fontsize=12, labelpad=10)
                ax.set_zlabel('Z', fontsize=12, labelpad=-0.5,color='black')  # 减小labelpad
                

                # 微调Z轴标签的位置，使其靠近坐标轴
                ax.zaxis.set_label_coords(1.05, 0.5)  # 调整位置使标签更靠近右侧

                # 设置3D比例并调整图形位置
                ax.set_box_aspect([1.2, 1.1, 0.9])  # 将Z轴比例稍微缩小，增加Z轴的空间

            else:
                ax.set_xlabel('X', fontsize=14, labelpad=20)
                ax.set_ylabel('Y', fontsize=14, labelpad=20)

            # 调整图形与边缘的距离，特别是右边的边距
            plt.subplots_adjust(right=0.85)

            plt.legend()
            plt.grid(True)
            plt.title(f'Pareto Fronts of Algorithms on {problem}', fontsize=14)

            # 增加边距来确保Z轴标签能显示
            plt.savefig(output_dir + f'{name}_pareto_fronts.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.show()
    
    def draw_test_indicator(self, data: dict, output_dir: str, indicator:str,Name: Optional[Union[str, list]]=None, categorized: bool=False, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            if not categorized:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])

                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                plt.ylabel(str(indicator))
                plt.savefig(output_dir + f'{name}_{indicator}_curve.png', bbox_inches='tight')
                plt.close()
            else:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.agent:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                plt.ylabel(str(indicator))
                plt.savefig(output_dir + f'learnable_{name}_{indicator}_curve.png', bbox_inches='tight')
                plt.close()
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.t_optimizer:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                plt.ylabel(str(indicator))
                plt.savefig(output_dir + f'classic_{name}_{indicator}_curve.{fig_type}', bbox_inches='tight')
                plt.close()
    
    def draw_named_average_test_indicator(self, data: dict, output_dir: str, named_agents: dict, indicator:str,pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        fig = plt.figure(figsize=(50, 10))
        # plt.title('all problem cost curve')
        plots = len(named_agents.keys())
        for id, title in enumerate(named_agents.keys()):
            ax = plt.subplot(1, plots+1, id+1)
            ax.set_title(title, fontsize=25)
            Y = {}
            for problem in list(data.keys()):
                # 计算全局最大值和最小值
                all_values = []
                for agent in data[problem].keys():
                    all_values.append(np.array(data[problem][agent]))
                all_values = np.concatenate(all_values, axis=0)  # 拼接所有数据
                global_min = np.min(all_values)  # 计算全局最小值
                global_max = np.max(all_values)  # 计算全局最大值
                
                for agent in list(data[problem].keys()):
                    if agent not in named_agents[title]:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if agent not in Y.keys():
                        Y[agent] = {'mean': [], 'std': []}
                    values = np.array(data[problem][agent][indicator])
                    values = (values - global_min) / (global_max - global_min + 1e-8)  # 避免除零
                
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    Y[agent]['mean'].append(mean)
                    Y[agent]['std'].append(std)

            for id, agent in enumerate(list(Y.keys())):
                mean = np.mean(Y[agent]['mean'], 0)
                std = np.mean(Y[agent]['std'], 0)

                X = np.arange(mean.shape[-1])
                X = np.array(X, dtype=np.float64)
                X *= (self.config.maxFEs / X[-1])

                ax.plot(X, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                ax.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=self.color_arrangement[agent])
            plt.grid()
            plt.xlabel('FEs')
            plt.ylabel('Normalized {indicator}')
            plt.legend()
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels, bbox_to_anchor=(plots/(plots+1)-0.02, 0.5), borderaxespad=0., loc=6, facecolor='whitesmoke')
        
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
        plt.savefig(output_dir + f'all_problem_{indicator}_curve.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_concrete_performance_hist(self, data: dict, output_dir: str, indicator: Optional[str] = None, Name: Optional[Union[str, list]] = None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        D = {}
        X = []
        
        # 遍历所有问题
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D:
                    D[agent] = []
                values = np.array(data[name][agent])
                D[agent].append(values[:, -1])

        # 绘制图表
        for agent in D.keys():
            plt.figure()
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])

            for a, b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)

            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            
            ylabel = indicator
            plt.ylabel(ylabel)

            plt.savefig(output_dir + f'{agent}_concrete_{indicator}_performance_hist.{fig_type}', bbox_inches='tight')
    
    def draw_boxplot(self, data: dict, output_dir: str, indicator:str,Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            Y = []
            X = []
            plt.figure(figsize=(30, 15))
            for agent in list(data[name].keys()):
                if ignore is not None and agent in ignore:
                    continue
                X.append(agent)
                values = np.array(data[name][agent])
                Y.append(values[:, -1])
            Y = np.transpose(Y)
            plt.boxplot(Y, labels=X, showmeans=True, patch_artist=True, showfliers=False,
                        medianprops={'color': 'green', 'linewidth': 3}, 
                        meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                        boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        )
            plt.xticks(rotation=30, fontsize=18)
            plt.xlabel('Agents')
            plt.ylabel(f'{name} {indicator} Boxplots')
            plt.savefig(output_dir + f'{name}_{indicator}_boxplot.{fig_type}', bbox_inches='tight')
            plt.close()
    
    def draw_overall_boxplot(self, data: dict, output_dir: str, indicator:str,ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        problems=[]
        agents=[]
        for problem in data.keys():
            problems.append(problem)
        for agent in data[problems[0]].keys():
            if ignore is not None and agent in ignore:
                continue
            agents.append(agent)
        run = len(data[problems[0]][agents[0]])
        values = np.zeros((len(agents), len(problems), run))
        plt.figure(figsize=(30, 15))
        for ip, problem in enumerate(problems):
            for ia, agent in enumerate(agents):
                values[ia][ip] = np.array(data[problem][agent])[:, -1]
            values[:, ip, :] = (values[:, ip, :] - np.min(values[:, ip, :])) / (np.max(values[:, ip, :]) - np.min(values[:, ip, :]))
        values = values.reshape(len(agents), -1).transpose()
        
        plt.boxplot(values, labels=agents, showmeans=True, patch_artist=True, showfliers=False,
                    medianprops={'color': 'green', 'linewidth': 3}, 
                    meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                    boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                    capprops={'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    )
        plt.xticks(rotation=30, fontsize=18)
        plt.xlabel('Agents')
        plt.ylabel(f'{indicator} Boxplots')
        plt.savefig(output_dir + f'overall_{indicator}_boxplot.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_train_logger(self, data_type: str, data: dict, output_dir: str, ylabel: str = None, norm: bool = False, pdf_fig: bool = True, data_wrapper: Callable = None) -> None:
        means, stds = self.get_average_data(data_type, data, norm=norm, data_wrapper=data_wrapper)
        plt.figure()
        for agent in means.keys():
            x = np.arange(len(means[agent]), dtype=np.float64)
            x = (self.config.max_learning_step / x[-1]) * x
            y = means[agent]
            s = np.zeros(y.shape[0])
            a = s[0] = y[0]
            norm = self.config.plot_smooth + 1
            for i in range(1, y.shape[0]):
                a = a * self.config.plot_smooth + y[i]
                s[i] = a / norm if norm > 0 else a
                norm *= self.config.plot_smooth
                norm += 1
            if agent not in self.color_arrangement.keys():
                self.color_arrangement[agent] = colors[self.arrange_index]
                self.arrange_index += 1
            plt.plot(x, s, label=to_label(agent), marker='*', markersize=12, markevery=2, c=self.color_arrangement[agent])
            plt.fill_between(x, (s - stds[agent]), (s + stds[agent]), alpha=0.2, facecolor=self.color_arrangement[agent])
            # plt.plot(x, returns[agent], label=to_label(agent))
        plt.legend()
        plt.grid()
        plt.xlabel('Learning Steps')    
        if ylabel is None:
            ylabel = data_type
        plt.ylabel(ylabel)
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'avg_{data_type}_curve.{fig_type}', bbox_inches='tight')
        plt.close()
    
    def post_processing_test_statics(self, log_dir: str, include_random_baseline: bool = False, pdf_fig: bool = True) -> None:
        with open(log_dir + 'test.pkl', 'rb') as f:
            results = pickle.load(f)
            
        metabbo = self.config.agent
        bbo = self.config.t_optimizer
        
        # 可选地读取 random_search_baseline.pkl
        if include_random_baseline:
            with open(log_dir + 'random_search_baseline.pkl', 'rb') as f:
                random = pickle.load(f)

        if not os.path.exists(log_dir + 'tables/'):
            os.makedirs(log_dir + 'tables/')

        self.gen_algorithm_complexity_table(results, log_dir + 'tables/')

        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')

        for indicator in self.indicators:
            self.draw_test_indicator(results[indicator], log_dir + 'pics/', indicator, pdf_fig=pdf_fig)
            self.draw_named_average_test_indicator(results[indicator], log_dir + 'pics/', \
                {'MetaBBO-RL': metabbo, 'Classic Optimizer': bbo}, indicator, pdf_fig=pdf_fig)
    
    def post_processing_rollout_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        with open(log_dir+'rollout.pkl', 'rb') as f:
            results = pickle.load(f)
        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')
        self.draw_train_logger('return', results['return'], log_dir + 'pics/', pdf_fig=pdf_fig)
        for indicator in self.indicators:
            self.draw_train_logger(indicator, results[indicator], log_dir + 'pics/', pdf_fig=pdf_fig)



class MMO_Logger(Basic_Logger):
    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__(config)

    def data_wrapper_prsr_rollout(self, data, ):
        res = []
        for key in data.keys():
            res.append(np.array(data[key])[:, -1, 3])
        return np.array(res)

    def data_wrapper_prsr_hist(self,data,):
        return np.array(data)[:, :, 3]

    def data_wrapper_cost_rollout(self,data, ):
        res = []
        for key in data.keys():
            res.append(np.array(data[key])[:, -1])
        return np.array(res)

    def data_wrapper_return_rollout(self,data, ):
        res = []
        for key in data.keys():
            res.append(np.array(data[key]))
        return np.array(res)

    def gen_agent_performance_prsr_table(self, results: dict, data_type: str, out_dir: str) -> None:
        """
        Store the `Worst`, `Best`, `Median`, `Mean` and `Std` of cost results of each agent as excel
        """
        total_data=results
        table_data={}
        indexs=[]
        columns=['Worst','Best','Median','Mean','Std']
        for problem,value in total_data.items():
            indexs.append(problem)
            problem_cost=value
            for alg,alg_data in problem_cost.items():
                n_data=np.array(alg_data)[:, -1, 3]
                # if alg == 'MadDE' and problem == 'F5':
                #     for run in alg_data:
                #         print(len(run))
                #     print(len(n_data))
                best=np.min(n_data)
                best=np.format_float_scientific(best,precision=3,exp_digits=3)
                worst=np.max(n_data)
                worst=np.format_float_scientific(worst,precision=3,exp_digits=3)
                median=np.median(n_data)
                median=np.format_float_scientific(median,precision=3,exp_digits=3)
                mean=np.mean(n_data)
                mean=np.format_float_scientific(mean,precision=3,exp_digits=3)
                std=np.std(n_data)
                std=np.format_float_scientific(std,precision=3,exp_digits=3)

                if not alg in table_data:
                    table_data[alg]=[]
                table_data[alg].append([worst,best,median,mean,std])
        for alg,data in table_data.items():
            dataframe=pd.DataFrame(data=data,index=indexs,columns=columns)
            #print(dataframe)
            dataframe.to_excel(os.path.join(out_dir,f'{alg}_concrete_performance_{data_type}_table.xlsx'))

    def gen_overall_tab(self, results: dict, out_dir: str) -> None:
        """
        Store the overall results inculding `objective values` (costs), `pr` and `sr` as excel
        """
        # get multi-indexes first
        problems = []
        statics = ['Obj','Pr', 'Sr']
        optimizers = []
        for problem in results['cost'].keys():
            problems.append(problem)
        for optimizer in results['cost'][problems[0]].keys():
            optimizers.append(optimizer)
        multi_columns = pd.MultiIndex.from_product(
            [problems,statics], names=('Problem', 'metric')
        )
        df_results = pd.DataFrame(np.ones(shape=(len(optimizers),len(problems)*len(statics))),
                                index=optimizers,
                                columns=multi_columns)

        # # calculate baseline1 cmaes
        # cmaes_obj = {}
        # for problem in problems:
        #     blobj_problem = results['cost'][problem]['CMAES']  # 51 * record_length
        #     objs = []
        #     for run in range(self.config.test_run):
        #         objs.append(blobj_problem[run][-1])
        #     cmaes_obj[problem] = sum(objs) / self.config.test_run

        # # calculate baseline2 random_search
        # rs_obj = {}
        # for problem in problems:
        #     blobj_problem = results['cost'][problem]['Random_search']  # 51 * record_length
        #     objs = []
        #     for run in range(self.config.test_run):
        #         objs.append(blobj_problem[run][-1])
        #     rs_obj[problem] = sum(objs) / self.config.test_run

        # calculate each Obj
        for problem in problems:
            for optimizer in optimizers:
                obj_problem_optimizer = results['cost'][problem][optimizer]
                objs_ = np.array(obj_problem_optimizer)[:, -1]
                avg_obj = sum(objs_)/self.config.test_run
                std_obj = np.std(objs_)
                df_results.loc[optimizer, (problem, 'Obj')] = np.format_float_scientific(avg_obj, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_obj, precision=3, exp_digits=1) + ")"

                pr_problem_optimizer = results['pr'][problem][optimizer]
                prs_ = np.array(pr_problem_optimizer)[:, -1, 3]
                avg_pr = sum(prs_)/self.config.test_run
                std_pr = np.std(prs_)
                df_results.loc[optimizer, (problem, 'Pr')] = np.format_float_scientific(avg_pr, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_pr, precision=3, exp_digits=1) + ")"

                sr_problem_optimizer = results['sr'][problem][optimizer]
                srs_ = np.array(sr_problem_optimizer)[:, -1, 3]
                avg_sr = sum(srs_)/self.config.test_run
                std_sr = np.std(srs_)
                df_results.loc[optimizer, (problem, 'Sr')] = np.format_float_scientific(avg_sr, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_sr, precision=3, exp_digits=1) + ")"

        df_results.to_excel(out_dir+'overall_table.xlsx')

    def draw_concrete_performance_prsr_hist(self, data: dict, data_type: str,output_dir: str, Name: Optional[Union[str, list]]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        D = {}
        X = []
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D.keys():
                    D[agent] = []
                values = np.array(data[name][agent])[:, :, 3]
                D[agent].append(values[:, -1])

        for agent in D.keys():
            plt.figure()
            # plt.title(f'{agent} performance histgram')
            X = list(data.keys())
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])
            for a,b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)
            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            plt.ylabel(f'Normalized {data_type}')
            plt.savefig(output_dir + f'{agent}_concrete_performance_{data_type}_hist.{fig_type}', bbox_inches='tight')

    def draw_boxplot_prsr(self, data: dict, data_type: str,output_dir: str, Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            Y = []
            X = []
            plt.figure(figsize=(30, 15))
            for agent in list(data[name].keys()):
                if ignore is not None and agent in ignore:
                    continue
                X.append(agent)
                # values = np.array(data[name][agent])
                # Y.append(values[:, -1])
                Y.append(np.array(data[name][agent])[:,-1,3])
            Y = np.transpose(Y)
            plt.boxplot(Y, labels=X, showmeans=True, patch_artist=True, showfliers=False,
                        medianprops={'color': 'green', 'linewidth': 3}, 
                        meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                        boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        )
            plt.xticks(rotation=30, fontsize=18)
            plt.xlabel('Agents')
            plt.ylabel(f'{name} {data_type} Boxplots')
            plt.savefig(output_dir + f'{name}_{data_type}_boxplot.{fig_type}', bbox_inches='tight')
            plt.close()

    def draw_overall_boxplot_prsr(self, data: dict, data_type: str,output_dir: str, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        problems=[]
        agents=[]
        for problem in data.keys():
            problems.append(problem)
        for agent in data[problems[0]].keys():
            if ignore is not None and agent in ignore:
                continue
            agents.append(agent)
        run = len(data[problems[0]][agents[0]])
        values = np.zeros((len(agents), len(problems), run))
        plt.figure(figsize=(30, 15))
        for ip, problem in enumerate(problems):
            for ia, agent in enumerate(agents):
                values[ia][ip] = np.array(data[problem][agent])[:, -1, 3]
            values[:, ip, :] = (values[:, ip, :] - np.min(values[:, ip, :])) / (np.max(values[:, ip, :]) - np.min(values[:, ip, :]))
        values = values.reshape(len(agents), -1).transpose()
        
        plt.boxplot(values, labels=agents, showmeans=True, patch_artist=True, showfliers=False,
                    medianprops={'color': 'green', 'linewidth': 3}, 
                    meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                    boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                    capprops={'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    )
        plt.xticks(rotation=30, fontsize=18)
        plt.xlabel('Agents')
        plt.ylabel(f'{data_type} Boxplots')
        plt.savefig(output_dir + f'overall_{data_type}_boxplot.{fig_type}', bbox_inches='tight')
        plt.close()

    def get_average_prsr_rank(self, results: dict):
        problems=[]
        agents=[]
        for problem in results.keys():
            problems.append(problem)
        for agent in results[problems[0]].keys():
            agents.append(agent)
        avg_data={}
        std_data={}
        for agent in agents:
            avg_data[agent]=[]
            std_data[agent]=[]
            for problem in problems:
                values = results[problem][agent]
                values = np.array(values)[:, -1, 3]
                std_data[agent].append(np.std(values, -1))
                avg_data[agent].append(np.mean(values, -1))
            avg_data[agent] = np.mean(avg_data[agent], 0)
            std_data[agent] = np.mean(std_data[agent], 0)
        return avg_data, std_data

    def draw_rank_hist_prsr(self, data: dict, data_type: str,output_dir: str, pdf_fig: bool = True) -> None:
        fig_type = 'pdf' if pdf_fig else 'png'
        metric, metric_std = self.get_average_prsr_rank(data)
        X, Y = list(metric.keys()), list(metric.values())
        _, S = list(metric_std.keys()), list(metric_std.values())
        n_agents = len(X)
        for i in range(n_agents):
            X[i] = to_label(X[i])

        plt.figure(figsize=(4*n_agents,15))
        plt.bar(X, Y)
        plt.errorbar(X, Y, S, fmt='s', ecolor='dimgray', ms=1, color='dimgray', elinewidth=5, capsize=30, capthick=5)
        for a,b in zip(X, Y):
            plt.text(a, b+0.05, '%.2f' % b, ha='center', fontsize=55)
        plt.xticks(rotation=45, fontsize=60)
        plt.yticks(fontsize=60)
        plt.ylim(0, np.max(np.array(Y) + np.array(S)) * 1.1)
        plt.title(f'The {data_type} for {self.config.test_problem}-{self.config.test_difficulty}', fontsize=70)
        plt.ylabel(f'{data_type}', fontsize=60)
        plt.savefig(output_dir + f'{data_type}_rank_hist.{fig_type}', bbox_inches='tight')

    def post_processing_test_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        with open(log_dir + 'test.pkl', 'rb') as f:
            results = pickle.load(f)
            
        metabbo = self.config.agent
        bbo = self.config.t_optimizer
        

        if not os.path.exists(log_dir + 'tables/'):
            os.makedirs(log_dir + 'tables/')

        self.gen_overall_tab(results, log_dir + 'tables/')
        self.gen_algorithm_complexity_table(results, log_dir + 'tables/')
        self.gen_agent_performance_table(results, log_dir + 'tables/')
        self.gen_agent_performance_prsr_table(results['pr'],'pr', log_dir+'tables/') 
        self.gen_agent_performance_prsr_table(results['sr'], 'sr',log_dir + 'tables/')
        

        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')

        self.draw_concrete_performance_hist(results['cost'], log_dir+'pics/',pdf_fig=pdf_fig)
        self.draw_concrete_performance_prsr_hist(results['pr'], 'pr', log_dir+'pics/', pdf_fig = pdf_fig)
        self.draw_concrete_performance_prsr_hist(results['sr'], 'sr', log_dir+'pics/', pdf_fig = pdf_fig)
        self.draw_boxplot(results, log_dir+'pics/', pdf_fig=pdf_fig)
        self.draw_boxplot_prsr(results['pr'], 'pr', log_dir+'pics/', pdf_fig=pdf_fig)
        self.draw_boxplot_prsr(results['sr'], 'sr', log_dir+'pics/', pdf_fig=pdf_fig)
        self.draw_overall_boxplot(results['cost'], log_dir+'pics/', pdf_fig=pdf_fig)
        self.draw_overall_boxplot_prsr(results['pr'], 'pr', log_dir+'pics/',pdf_fig=pdf_fig)
        self.draw_overall_boxplot_prsr(results['sr'], 'sr', log_dir+'pics/',pdf_fig=pdf_fig)

        self.draw_test_data(results['cost'], 'cost', log_dir + 'pics/', logged=True, categorized=False, pdf_fig=pdf_fig, data_wrapper=np.array)
        self.draw_test_data(results['pr'],'pr', log_dir + 'pics/', logged=False, categorized=False, pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_prsr_hist)
        self.draw_test_data(results['sr'],'sr', log_dir + 'pics/', logged=False, categorized=False, pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_prsr_hist)
        self.draw_rank_hist_prsr(results['pr'], 'pr',log_dir + 'pics/', pdf_fig=pdf_fig) 
        self.draw_rank_hist_prsr(results['sr'], 'sr', log_dir + 'pics/',pdf_fig=pdf_fig)


    def post_processing_rollout_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        with open(log_dir+'rollout.pkl', 'rb') as f:
            results = pickle.load(f)
        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')
        self.draw_train_logger('return', results['return'], log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_return_rollout)
        self.draw_train_logger('cost', results['cost'], log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_cost_rollout)
        self.draw_train_logger('pr', results['pr'], log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_prsr_rollout)
        self.draw_train_logger('sr', results['sr'], log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_prsr_rollout)

#logger
# class basic_Logger:
class MTO_Logger(Basic_Logger):
    def __init__(self, config):
        super().__init__(config)

    def draw_avg_train_return(self, data: list, output_dir: str) -> None: 
        plt.figure()
        return_data = np.array(data,dtype=np.float32) #[epochs, env_cnt]
        x = np.arange(return_data.shape[0])
        y = np.mean(return_data, axis=-1)
        plt.plot(x, y, 
         color='blue',       
         marker='o',         
         linestyle='-',     
         linewidth=2,        
         markersize=8)       
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Return')
        plt.grid()
        plt.savefig(output_dir + f'avg_mto_return.png', bbox_inches='tight')
        plt.close()

    def draw_avg_train_cost(self, data:list, output_dir: str) -> None:
        plt.figure()
        cost_data = np.array(data,dtype=np.float32) #[epochs, env_cnt, task_cnt]
        x = np.arange(cost_data.shape[0])
        y = np.mean(np.mean(cost_data, axis=-1), axis=-1)
        plt.plot(x, y, 
         color='blue',       
         marker='o',         
         linestyle='-',     
         linewidth=2,        
         markersize=8)       
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Cost')
        plt.grid()
        plt.savefig(output_dir + f'avg_mto_cost.png', bbox_inches='tight')
        plt.close()

    def draw_per_task_cost(self, data:list, output_dir: str) -> None:
        data = np.array(data, dtype=np.float32)
        if data.ndim == 3:  
            data = np.mean(data, axis=1)

        epochs, task_cnt = data.shape
        fig, axes = plt.subplots(task_cnt, 1, figsize=(10, 2 * task_cnt))  
        if task_cnt == 1:
            axes = [axes]

        for task_idx in range(task_cnt):
            ax = axes[task_idx]
            ax.plot(range(epochs), data[:, task_idx], color='blue', label=f'Task {task_idx+1}')
            ax.set_title(f'Task {task_idx+1}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_dir + f'mto_each_task_cost.png', bbox_inches='tight')
        plt.close()

    def save_mto_cost_to_csv(self, data:list, output_dir: str) -> None:
        data = np.array(data, dtype=np.float32)
        if data.ndim == 3:  
            data = np.mean(data, axis=1)

        epochs, task_cnt = data.shape
        df = pd.DataFrame(data, columns=[f'Task_{i+1}' for i in range(task_cnt)])
        df.insert(0, 'Epoch', np.arange(epochs))
        output_path = output_dir + f'mto_each_task_cost.csv'
        df.to_csv(output_path, index=False)

    def save_mto_reward_to_csv(self, data:list, output_dir: str) -> None:
        data = np.array(data, dtype=np.float32)
        if data.ndim == 2:  
            data = np.mean(data, axis=-1)
        epochs = data.shape[0]
        df = pd.DataFrame({
            "Epoch": np.arange(epochs),  
            "Value": data              
        })
        output_path = output_dir + f'mto_return.csv'
        df.to_csv(output_path, index=False)

    def draw_env_task_cost(self, data:list, output_dir:str) -> None:
        data = np.array(data, dtype=np.float32)
        if data.ndim < 3:
            return 
        epochs, env_cnt, task_cnt = data.shape

        for task in range(task_cnt):
            plt.figure(figsize=(10, 5))
            for env in range(env_cnt):
                plt.plot(data[:, env, task], label=f'Env {env+1}')
            plt.title(f'Task {task+1} Performance Across Environments')
            plt.xlabel('Epochs')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid()
        plt.savefig(output_dir + f'mto_env_task_{task+1}_cost.png', bbox_inches='tight')
        plt.close()
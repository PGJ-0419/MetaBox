import argparse
import time
import os
import subprocess, sys
# class Config:
#     def set(user_config):
#         default_config = get_config()
#         for key, value in user_config.items():
#             if hasattr(default_config, key):
#                 setattr(default_config, key, value)
#         return default_config
def Config(user_config, datasets_dir = None):
    default_config = get_config()
    for key, value in user_config.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)

    # # 判断是不是 HPO-B 任务
    # is_hpo_b = (
    #         'hpo-b' in getattr(default_config, 'train_probelm', '') or
    #         'hpo-b' in getattr(default_config, 'test_probelm', '')
    # )
    #
    # if is_hpo_b:
    #     print("Detected HPO-B Problem.")
    #
    #     # 数据集目录默认使用当前工作目录
    #     if datasets_dir is None:
    #         datasets_dir = os.path.join(os.getcwd(), "metabox_data", "hpo_b")
    #     os.makedirs(datasets_dir, exist_ok = True)
    #
    #     # 检查该目录下是否已有对应数据文件
    #     expected_files = ['hpo_b_train.json', 'hpo_b_test.json']  # 你可以换成真实文件名
    #     missing_files = [f for f in expected_files if not os.path.exists(os.path.join(datasets_dir, f))]
    #
    #     if missing_files:
    #         print(f"[Warning] HPO-B dataset files not found: {missing_files}")
    #         print(f"Expected in directory: {datasets_dir}")
    #         # 可以在这里加入自动下载逻辑 if you want
    #     else:
    #         print(f"HPO-B dataset is ready in: {datasets_dir}")

    # 判断是不是 uav 任务
    is_uav = 'uav' in default_config.train_problem or 'uav' in default_config.test_problem
    if is_uav:
        print("Detected UAV Problem.")

        # 数据集目录默认使用当前工作目录
        if datasets_dir is None:
            datasets_dir = os.path.join(os.getcwd(), "metabox_data", "uav")
        os.makedirs(datasets_dir, exist_ok = True)

        # 检查该目录下是否已有对应数据文件
        expected_files = ['Model56.pkl']  # 你可以换成真实文件名
        missing_files = [f for f in expected_files if not os.path.exists(os.path.join(datasets_dir, f))]

        if missing_files:
            print(f"[Warning] UAV dataset files not found: {missing_files}")
            print(f"Expected in directory: {datasets_dir}")
            # 可以在这里加入自动下载逻辑 if you want
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                # check the required package, if not exists, pip install it
                try:
                    subprocess.check_call([sys.executable, '-m', "pip", "install", 'huggingface_hub'])
                    # print("huggingface_hub has been installed successfully!")
                    from huggingface_hub import snapshot_download
                except subprocess.CalledProcessError as e:
                    print(f"Install huggingface_hub leads to errors: {e}")
            snapshot_download(repo_id = 'GMC-DRL/MetaBox-uav', repo_type = "dataset", local_dir = datasets_dir)
        else:
            print(f"UAV dataset is ready in: {datasets_dir}")
        default_config.uav_path = datasets_dir + '/Model56.pkl'

    return default_config

def get_config(args=None):
    parser = argparse.ArgumentParser()
    # Common config
    parser.add_argument('--train_problem', default = 'bbob-10D', choices = ['bbob-10D', 'bbob-30D', 'bbob-torch-10D', 'bbob-torch-30D', 'bbob-noisy-10D',
                                                                        'bbob-noisy-30D', 'bbob-noisy-torch-10D', 'bbob-noisy-torch-30D', 'bbob-surrogate-2D','bbob-surrogate-5D','bbob-surrogate-10D',
                                                                         'hpo-b', 'lsgo', 'lsgo-torch', 'protein', 'protein-torch', 'uav',
                                                                                'mmo', 'mmo-torch', 'wcci2020', 'cec2017mto', 'moo-synthetic'],
                        help='specify the problem suite for training')
    parser.add_argument('--test_problem', default = None, choices = [None, 'bbob-10D', 'bbob-30D', 'bbob-torch-10D', 'bbob-torch-30D', 'bbob-noisy-10D', 
                                                                        'bbob-noisy-30D', 'bbob-noisy-torch-10D', 'bbob-noisy-torch-30D', 'bbob-surrogate-2D','bbob-surrogate-5D','bbob-surrogate-10D', 'hpo-b',
                                                                                'lsgo', 'lsgo-torch', 'protein', 'protein-torch', 'uav', 'uav-torch', 'ne', 
                                                                                'mmo', 'mmo-torch', 'wcci2020', 'cec2017mto', 'moo-synthetic'],
                        help='specify the problem suite for testing, default to be consistent with training')
    parser.add_argument('--train_difficulty', default='easy', choices=['all', 'easy', 'difficult', 'user-define'], help='difficulty level for training problems')
    parser.add_argument('--test_difficulty', default=None, choices=['all', 'easy', 'difficult', 'user-define'], help='difficulty level for testing problems, default to be consistent with training')
    # parser.add_argument('--dim', type=int, default=10, help='dimension of search space')
    parser.add_argument('--upperbound', type=float, default=5, help='upperbound of search space')
    parser.add_argument('--user_train_problem_list', nargs='+', default=None, help = 'user define training problem list')
    parser.add_argument('--user_test_problem_list', nargs='+', default=None, help = 'user define testing problem list')
    parser.add_argument('--device', default='cpu', help='device to use')
    parser.add_argument('--train', default=None, action='store_true', help='switch to train mode')
    parser.add_argument('--test', default=None, action='store_true', help='switch to inference mode')
    parser.add_argument('--rollout', default=None, action='store_true', help='switch to rollout mode')
    parser.add_argument('--run_experiment', default=None, action='store_true', help='switch to run_experiment mode')
    parser.add_argument('--mgd_test', default=None, action='store_true', help='switch to mgd_test mode')
    parser.add_argument('--mte_test', default=None, action='store_true', help='switch to mte_test mode')

    parser.add_argument('--task_cnt', type=int, default=10, help='number of tasks in multitask') #for multitask
    parser.add_argument('--generation', type=int, default=250, help='total generations for L2O') #for multitask

    parser.add_argument('--full_meta_data', type=bool, default=True, help='store the metadata')
    # Training parameters
    parser.add_argument('--max_learning_step', type=int, default=1500000, help='the maximum learning step for training')
    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size of train set')
    parser.add_argument('--train_agent', default=None, help='agent for training')
    parser.add_argument('--train_optimizer', default=None, help='optimizer for training')
    parser.add_argument('--agent_save_dir', type=str, default='agent_model/train/',
                        help='save your own trained agent model')
    parser.add_argument('--log_dir', type=str, default='output/',
                        help='logging testing output')
    parser.add_argument('--draw_interval', type=int, default=3, help='interval epochs in drawing figures')
    parser.add_argument('--agent_for_plot_training', type=str, nargs='+', default=['RL_HPSDE_Agent'],
                        help='learnable optimizer to compare')
    parser.add_argument('--n_checkpoint', type=int, default=20, help='number of training checkpoints')
    parser.add_argument('--resume_dir', type=str, help='directory to load previous checkpoint model')
    parser.add_argument('--train_parallel_mode', type=str, default='dummy', choices=['dummy', 'subproc', 'ray'], help='the parellel processing method for batch env step in training')

    # Testing parameters
    parser.add_argument('--agent', type = str, nargs = '+', default = [], help = 'Key written in key.json')
    parser.add_argument('--t_optimizer', type = str, nargs = '+', default = [], help = 'traditional optimizer')

    # parser.add_argument('--agent', default=None, help='None: traditional optimizer, else Learnable optimizer')
    # parser.add_argument('--agent_load_dir', type=str,
    #                     help='load your own agent model')
    # parser.add_argument('--optimizer', default=None, help='your own learnable or traditional optimizer')
    # parser.add_argument('--agent_for_cp', type=str, nargs='+', default=[],
    #                     help='learnable optimizer to compare')
    # parser.add_argument('--l_optimizer_for_cp', type=str, nargs='+', default=[],
    #                     help='learnable optimizer to compare')  # same length with "agent_for_cp"
    # parser.add_argument('--t_optimizer_for_cp', type=str, nargs='+', default=[],
    #                     help='traditional optimizer to compare')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test set')
    parser.add_argument('--parallel_batch', type=str, default='Batch', choices=['Full', 'Baseline_Problem', 'Problem_Testrun', 'Batch'], help='the parellel processing mode for testing')
    

    # Rollout parameters
    parser.add_argument('--agent_for_rollout', type=str, help='learnable agent for rollout')
    parser.add_argument('--checkpoints_for_rollout', default=None, type=int, nargs='+', help='the index of checkpoints for rollout')
    parser.add_argument('--plot_smooth', type=float, default=0.8,
                        help='a float between 0 and 1 to control the smoothness of figure curves')

    # parameters common to mgd_test(zero-shot) & mte_test(transfer_learning)
    parser.add_argument('--problem_from', choices=['bbob', 'bbob-noisy', 'bbob-torch', 'bbob-noisy-torch', 'protein', 'protein-torch'],
                        help='source problem set in zero-shot and transfer learning')
    parser.add_argument('--problem_to', choices=['bbob', 'bbob-noisy', 'bbob-torch', 'bbob-noisy-torch', 'protein', 'protein-torch'],
                        help='target problem set in zero-shot and transfer learning')
    parser.add_argument('--difficulty_from', default='easy', choices=['easy', 'difficult'],
                        help='difficulty of source problem set in zero-shot and transfer learning')
    parser.add_argument('--difficulty_to', default='easy', choices=['easy', 'difficult'],
                        help='difficulty of target problem set in zero-shot and transfer learning')

    # mgd_test(zero-shot) parameters
    parser.add_argument('--model_from', type=str, help='the model trained on source problem set')
    parser.add_argument('--model_to', type=str, help='the model trained on target problem set')

    # mte_test(transfer_learning) parameters
    parser.add_argument('--pre_train_rollout', type=str, help='key of pre-train models rollout in model.json')
    parser.add_argument('--scratch_rollout', type=str, help='key of scratch models rollout result in model.json')

    # todo add new config

    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--seed', type = int, default = 3849)
    parser.add_argument('--epoch_seed', type = int, default = 100)
    parser.add_argument('--id_seed', type = int, default = 5)
    parser.add_argument('--train_mode', default='single', type = str, choices = ['single', 'multi'])
    parser.add_argument('--end_mode', type = str, default = 'epoch', choices = ['step', 'epoch'])

    parser.add_argument('--test_run', type = int, default = 51)
    parser.add_argument('--rollout_run', type = int, default = 10)

    parser.add_argument('--no_tb', action='store_true', default = False, help = 'disable tensorboard logging')
    parser.add_argument('--log_step', type = int, default = 50, help = 'log every log_step steps')


    config = parser.parse_args(args)

    config.maxFEs = 2000
    # for bo, maxFEs is relatively smaller due to time limit
    config.n_logpoint = 50
    
    if config.test_problem is None:
        config.test_problem = config.train_problem
    if config.test_difficulty is None:
        config.test_difficulty = config.train_difficulty
    if config.end_mode == 'epoch':
        config.max_learning_step = 1e9
    # if config.run_experiment and len(config.agent_for_cp) >= 1:
    #     assert config.agent_load_dir is not None, "Option --agent_load_dir must be given since you specified option --agent_for_cp."

    if config.mgd_test or config.mte_test:
        config.train_problem = config.problem_to
        config.train_difficulty = config.difficulty_to

    if config.train_problem in ['protein', 'protein-torch']:
        config.dim = 12
        config.maxFEs = 1000
        config.n_logpoint = 5

    config.run_time = f'{time.strftime("%Y%m%dT%H%M%S")}_{config.train_problem}_{config.train_difficulty}'
    config.test_log_dir = config.log_dir + 'test/' + config.run_time + '/'
    config.rollout_log_dir = config.log_dir + 'rollout/' + config.run_time + '/'
    config.mgd_test_log_dir = config.log_dir + 'mgd_test/' + config.run_time + '/'
    config.mte_test_log_dir = config.log_dir + 'mte_test/' + config.run_time + '/'

    if config.train or config.run_experiment:
        config.agent_save_dir = config.agent_save_dir + config.train_agent + '/' + config.run_time + '/'

    if config.end_mode == "step":
        config.save_interval = config.max_learning_step // config.n_checkpoint
    elif config.end_mode == "epoch":
        config.save_interval = config.max_epoch // config.n_checkpoint
    config.log_interval = config.maxFEs // config.n_logpoint

    if 'CMAES' not in config.t_optimizer:
        config.t_optimizer.append('CMAES')
    if 'Random_search' not in config.t_optimizer:
        config.t_optimizer.append('Random_search') # todo

    return config

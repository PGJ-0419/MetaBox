import numpy as np
from torch.utils.data import Dataset
import subprocess, sys, os
from .hpo_b import HPOB_Problem
from tqdm import tqdm
import json
import xgboost as xgb


class HPOB_Dataset(Dataset):
    def __init__(self,
                 data,
                 batch_size=1):
        super().__init__()
        self.data = data
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(datapath=None,
                     train_batch_size=1,
                     test_batch_size=1,
                     upperbound = None,
                     difficulty = None,
                     user_train_list=None,
                     user_test_list=None,
                     cost_normalize=False,):
        # get functions ID of indicated suit
        if datapath is None:
            datapath = os.path.join(os.getcwd(), "metabox_data")
        root_dir = datapath+"HPO-B-main/hpob-data/"
        surrogates_dir = datapath+"HPO-B-main/saved-surrogates/"
        
        # if not os.path.exists(root_dir) or len(os.listdir(root_dir)) < 7 or not os.path.exists(surrogates_dir) or len(os.listdir(surrogates_dir)) < 1909:
        #     try:
        #         from huggingface_hub import snapshot_download
        #     except ImportError:
        #         # check the required package, if not exists, pip install it
        #         try:
        #             subprocess.check_call([sys.executable,'-m', "pip", "install", 'huggingface_hub'])
        #             # print("huggingface_hub has been installed successfully!")
        #             from huggingface_hub import snapshot_download
        #         except subprocess.CalledProcessError as e:
        #             print(f"Install huggingface_hub leads to errors: {e}")
                    
        #     snapshot_download(repo_id='GMC-DRL/MetaBox-HPO-B', repo_type="dataset", local_dir=datapath)
        #     print("Extract data...")
        #     os.system(f'tar -xf {datapath}HPO-B-main.tar.gz -C {datapath}')
        #     os.system(f'rm {datapath}HPO-B-main.tar.gz')
        #     os.system(f'rm {datapath}.gitattributes')
            
        meta_train_data,meta_vali_data,meta_test_data,bo_initializations,surrogates_stats=get_data(root_dir=root_dir, mode="v3", surrogates_dir=surrogates_dir)
        
        if (user_train_list is None and user_test_list is None) or difficulty == 'all':
            
            def process_data(data, name, n):
                problems = []
                pbar = tqdm(desc=f'Loading {name}', total=n, leave=False)
                for search_space_id in data.keys():
                    for dataset_id in data[search_space_id].keys():
                        bst_model,y_min,y_max=get_bst(surrogates_dir=datapath+'HPO-B-main/saved-surrogates/',search_space_id=search_space_id,dataset_id=dataset_id,surrogates_stats=surrogates_stats)
                        X = np.array(data[search_space_id][dataset_id]["X"])
                        dim = X.shape[1]
                        p=HPOB_Problem(bst_surrogate=bst_model,dim=dim,y_min=y_min,y_max=y_max,lb = -upperbound,ub = upperbound,normalized=cost_normalize)
                        problems.append(p)
                        pbar.update()
                pbar.close()
                return problems

            train_set = process_data(meta_train_data, 'meta_train_data', 758)
            test_set = process_data(meta_vali_data, 'meta_vali_data', 91) + process_data(meta_test_data, 'meta_test_data', 86)
            if difficulty == 'all':
                train_set = test_set = train_set + test_set
            
        else:
            train_set = []
            test_set = []
            
            def process_data(data, name, n):
                pbar = tqdm(desc=f'Loading {name}', total=n, leave=False)
                for search_space_id in data.keys():
                    for dataset_id in data[search_space_id].keys():
                        bst_model,y_min,y_max=get_bst(surrogates_dir=datapath+'HPO-B-main/saved-surrogates/',search_space_id=search_space_id,dataset_id=dataset_id,surrogates_stats=surrogates_stats)
                        X = np.array(data[search_space_id][dataset_id]["X"])
                        dim = X.shape[1]
                        p=HPOB_Problem(bst_surrogate=bst_model,dim=dim,y_min=y_min,y_max=y_max,normalized=cost_normalize)
                        if user_train_list is not None and user_test_list is not None:
                            if search_space_id+'-'+dataset_id in user_train_list:
                                train_set.append(p)
                            if search_space_id+'-'+dataset_id in user_test_list:
                                test_set.append(p)
                        elif user_train_list is not None:
                            if search_space_id+'-'+dataset_id in user_train_list:
                                train_set.append(p)
                            else:
                                test_set.append(p)
                        elif user_test_list is not None:
                            if search_space_id+'-'+dataset_id in user_test_list:
                                test_set.append(p)
                            else:
                                train_set.append(p)
                        else:
                            raise NotImplementedError
                        pbar.update()
                pbar.close()
                
            process_data(meta_train_data, 'meta_train_data', 758)
            process_data(meta_vali_data, 'meta_vali_data', 91)
            process_data(meta_test_data, 'meta_test_data', 86)
        
            
        return HPOB_Dataset(train_set, train_batch_size), HPOB_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'HPOB_Dataset'):
        return HPOB_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
        
        
def get_data(mode,surrogates_dir,root_dir):
    train_set,vali_set,test_set=None,None,None
    if mode == "v3-test":
        train_set,vali_set,test_set,bo_initializations=load_data(root_dir, only_test=True)
    elif mode == "v3-train-augmented":
        train_set,vali_set,test_set,bo_initializations=load_data(root_dir, only_test=False, augmented_train=True)
    elif mode in ["v1", "v2", "v3"]:
        train_set,vali_set,test_set,bo_initializations=load_data(root_dir, version = mode, only_test=False)
    else:
        raise ValueError("Provide a valid mode")

    surrogates_file = surrogates_dir+"summary-stats.json"
    if os.path.isfile(surrogates_file):
        with open(surrogates_file) as f:
            surrogates_stats = json.load(f)

    return train_set,vali_set,test_set,bo_initializations,surrogates_stats

def load_data( rootdir="", version = "v3", only_test = True, augmented_train = False):
    
        """
        Loads data with some specifications.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * version: name indicating what HPOB version to use. Options: v1, v2, v3).
            * Only test: Whether to load only testing data (valid only for version v3).  Options: True/False
            * augmented_train: Whether to load the augmented train data (valid only for version v3). Options: True/False

        """

        print("Reading data...")
        meta_train_augmented_path = os.path.join(rootdir, "meta-train-dataset-augmented.json")
        meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
        meta_test_path = os.path.join(rootdir,"meta-test-dataset.json")
        meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")
        bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

        with open(meta_test_path, "rb") as f:
            meta_test_data = json.load(f)

        with open(bo_initializations_path, "rb") as f:
            bo_initializations = json.load(f)

        meta_train_data = None
        meta_validation_data = None
        
        if not only_test:
            if augmented_train or version=="v1":
                with open(meta_train_augmented_path, "rb") as f:
                    meta_train_data = json.load(f)
            else:
                with open(meta_train_path, "rb") as f:
                    meta_train_data = json.load(f)
            with open(meta_validation_path, "rb") as f:
                meta_validation_data = json.load(f)

        if version != "v3":
            temp_data = {}
            for search_space in meta_train_data.keys():
                temp_data[search_space] = {}

                for dataset in meta_train_data[search_space].keys():
                    temp_data[search_space][dataset] =  meta_train_data[search_space][dataset]

                if search_space in meta_test_data.keys():
                    for dataset in meta_test_data[search_space].keys():
                        temp_data[search_space][dataset] = meta_test_data[search_space][dataset]

                    for dataset in meta_validation_data[search_space].keys():
                        temp_data[search_space][dataset] = meta_validation_data[search_space][dataset]

            meta_train_data = None
            meta_validation_data = None
            meta_test_data = temp_data

        search_space_dims = {}

        for search_space in meta_test_data.keys():
            dataset = list(meta_test_data[search_space].keys())[0]
            X = meta_test_data[search_space][dataset]["X"][0]
            search_space_dims[search_space] = len(X)

        return meta_train_data,meta_validation_data,meta_test_data,bo_initializations

def get_bst(surrogates_dir,search_space_id,dataset_id,surrogates_stats):
    surrogate_name='surrogate-'+search_space_id+'-'+dataset_id
    bst_surrogate = xgb.Booster()
    bst_surrogate.load_model(surrogates_dir+surrogate_name+'.json')

    y_min = surrogates_stats[surrogate_name]["y_min"]
    y_max = surrogates_stats[surrogate_name]["y_max"]
    assert y_min is not None, 'y_min is None!!'

    return bst_surrogate,y_min,y_max


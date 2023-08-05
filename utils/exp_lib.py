import networkx as nx
import numpy as np
import copy
# import knapsack
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pyomo.environ as pyo
import sys
# sys.path.append('/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI')
# sys.path.append('/home/yanghong/YH/Laurence/GS')
import argparse
import tqdm
import json
import tensorflow as tf
from networkx.readwrite import json_graph
from eval_scripts import post_graphsage_trained
from graphsage.minibatch import NodeMinibatchIterator
from graphsage import supervised_train
from collections import defaultdict
import random
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, Accuracy, BinaryF1Score, F1Score,\
      BinaryPrecision, BinaryRecall 
import functools
from IPython.display import display, HTML, clear_output
# import pyinputplus as pyip
import glob
from pyomo.opt import SolverFactory
import Graph_Sampling as gs
from importlib import reload
import click
import functools
from utils.common import *
from torch_geometric.utils import to_undirected

device = "cpu"
if torch.cuda.is_available(): 
    device = "cuda:0"  
     
# def class_register(cls):
#     cls._alias_dict = {}
#     cls._alias2method_dict = {}
#     for methodname in dir(cls):
#         method = getattr(cls, methodname)
#         if hasattr(method, '_alias'):
#             cls._alias_dict.update(
#                 {cls.__name__ + '.' + methodname: method._alias})
#             cls._alias2method_dict.update(
#                 {method._alias: method})
#     return cls

# @class_register
# class DependencyController:
#     def __init__(self):
#         self._called_dict = {}
    
#     def require(name, *args0, **kwargs0):
#         def decorator(func):
#             @functools.wraps(func)
#             def wrap(self, *args, **kwargs):
#                 # print(f"inside wrap: {name}")
#                 if type(name) == str:
#                    if name not in self._called_dict.keys():
#                     self._alias2method_dict[name](self, *args0, **kwargs0)

#                 elif type(name) == list:
#                     for n in name:
#                         assert type(n) == str
#                         if n not in self._called_dict.keys():
#                             self._alias2method_dict[n](self, *args0, **kwargs0)

#                 else:
#                     raise ValueError(f"Invalid type {type(name)}")

#                 return func(self, *args, **kwargs)
#             return wrap
#         return decorator   

#     def track(func):
#         @functools.wraps(func)
#         def wrap(self, *args, **kwargs):
#             # print(self)
#             # print(self._called_dict)
#             # print(func.__name__)
#             self._called_dict[func._alias] = True
#             return func(self, *args, **kwargs)
#         return wrap 
    
#     def alias(name):
#         def new_func(func):
#             func._alias = name
#             return func
#         return new_func
    
    # @track
    # @alias("1.1")
    # def method1(self):
    #     print("METHOD1")

    # @require("1.1")
    # @track
    # @alias("1.2")
    # def method2(self):
    #     print("METHOD2")

    # @require("1.2")
    # @track
    # @alias("1.3")
    # def method3(self):
    #     print("METHOD3")


# def pyip_prompt_confirm(confirm_msg, confirm_default, **kwargs0):
#     def decorator(function):
#         def wrapper(*args, **kwargs):
#             if "_disable_prompt" in kwargs.keys() and kwargs["_disable_prompt"]:
#                 if '_is_last_prompt' not in kwargs0 or kwargs0['_is_last_prompt'] == True:
#                     del kwargs["_disable_prompt"]
#                 return function(*args, **kwargs)
#             else:
#                 if click.confirm(confirm_msg, default=confirm_default) == True:
#                     return function(*args, **kwargs)
#         return wrapper
#     return decorator

# def pyip_prompt_input(prompt_msg, target_param, **kwargs0):
#     def decorator(function):
#         def wrapper(*args, **kwargs):
#             if "_disable_prompt" in kwargs.keys() and kwargs["_disable_prompt"]:
#                 if '_is_last_prompt' not in kwargs0 or kwargs0['_is_last_prompt'] == True:
#                     del kwargs["_disable_prompt"]
#                 return function(*args, **kwargs)
#             else:
#                 _input = pyip.inputStr(prompt_msg + '\n')
#                 kwargs[target_param] = _input
#                 return function(*args, **kwargs)
#         return wrapper
#     return decorator

# def pyip_prompt_input_num(prompt_msg, target_param):
#     def decorator(function):
#         def wrapper(*args, **kwargs):
#             if "_disable_prompt" in kwargs.keys() and kwargs["_disable_prompt"]:
#                 del kwargs["_disable_prompt"]
#                 return function(*args, **kwargs)
#             else:
#                 _input = pyip.inputNum(prompt_msg + '\n')
#                 kwargs[target_param] = _input
#                 return function(*args, **kwargs)
#         return wrapper
#     return decorator




# def pyip_prompt_menu(prompt_msg, target_param, option_name):
#     def decorator(function):
#         def wrapper(*args, **kwargs):
#             if "_disable_prompt" in kwargs.keys() and kwargs["_disable_prompt"]:
#                 del kwargs["_disable_prompt"]
#                 return function(*args, **kwargs)
#             else:
#                 ctx = args[0]
#                 _options = getattr(ctx, option_name)
#                 _input = pyip.inputMenu(_options, prompt=prompt_msg+'\n', lettered=False, numbered=True)
#                 kwargs[target_param] = _input
#                 return function(*args, **kwargs)
#         return wrapper
#     return decorator

# import remove_edge_test_L
# reload(remove_edge_test_L)
# from remove_edge_test_L import generate_graph, get_args




def get_args(args=[]):
    parser = argparse.ArgumentParser(description = 'Experiment Node Selection Based on Attention')
    parser.add_argument('--N', type=int, help='Number of nodes in a generated graph.', default=500)
    parser.add_argument('--S', type=int, help='Mean cluster size.', default=100)
    parser.add_argument('--V', type=int, help='Shape parameter. The variance of cluster size distribution is s/v.', default=20)
    parser.add_argument('--mean_nb_partition', type=int, help='Mean of partition node number.', default=100)
    parser.add_argument('--std_nb_partition', type=int, help='Std of partition node number', default=5)
    parser.add_argument('--p_in', type=float, help='Intra partition edge probability', default=0.05)
    parser.add_argument('--p_out', type=float, help='Inter partition edge probability', default=0.005)
    parser.add_argument('--B', type=int, help='Budget value (maximum node number in output selection)', default=[200] )
    parser.add_argument('--B_positive_max', type=int, help='(Maximum) Budget value of the number positive samples (nodes)', default=120 )
    parser.add_argument('--B_positive_min', type=int, help='(Minimum) Budget value of the number positive samples (nodes)', default=20 )
    parser.add_argument('--B_negative_max', type=int, help='(Maximum) Budget value of the number negative samples (nodes)', default=120 )
    parser.add_argument('--B_negative_min', type=int, help='(Minimum) Budget value of the number positive samples (nodes)', default=20 )
    parser.add_argument('--label_rate_per_part', type=float, help='ratio of label data to all data for each partition', default=0.2 )
    parser.add_argument('--is_only_one_part_has_real_label', type=float, help='indicate whether we only make positive labels in one partition.', default=0 )
    parser.add_argument('--unlabel_ratio_max', type=float, help='Maximum ratio for unlabeled data in a cc', default=0.5 )
    parser.add_argument('--ignore_partition_ls', type=int, nargs='+', help='Make the partitions as noise so that no labels exist', default=[])
    parser.add_argument('--is_randomness_used', type=int, help='Indicate whether to use randomness in graph reduction step', default=0)
    parser.add_argument('--early_stop_epochs', type=int, help='Number of consecutive epochs to stop if no improvement found.', default=10)
    parser.add_argument('--lbn_0_constraint', type=int, help='Budget value (maximum node number of label 0 in output selection)', default=1_000_000 )
    parser.add_argument('--lbn_1_constraint', type=int, help='Budget value (maximum node number of label 1 in output selection)', default=1_000_000 )
    parser.add_argument('--lbn_2_constraint', type=int, help='Budget value (maximum node number of label 2 in output selection)', default=1_000_000 )
    parser.add_argument('--lbn_3_constraint', type=int, help='Budget value (maximum node number of label 3 in output selection)', default=1_000_000 )
    parser.add_argument('--lbn_4_constraint', type=int, help='Budget value (maximum node number of label 4 in output selection)', default=1_000_000 )
    parser.add_argument('--lbn_5_constraint', type=int, help='Budget value (maximum node number of label 5 in output selection)', default=1_000_000 )
    parser.add_argument('--lbn_6_constraint', type=int, help='Budget value (maximum node number of label 6 in output selection)', default=1_000_000 )
    parser.add_argument('--lbn_n1_constraint', type=int, help='Budget value (maximum node number of label -1 in output selection)', default=1_000_000 )
    parser.add_argument('--lbn_0_constraint_gt', type=int, help='Budget value (minimum node number of label 0 in output selection)', default=-1_000_000 )
    parser.add_argument('--lbn_1_constraint_gt', type=int, help='Budget value (minimum node number of label 1 in output selection)', default=-1_000_000 )
    parser.add_argument('--lbn_2_constraint_gt', type=int, help='Budget value (minimum node number of label 1 in output selection)', default=-1_000_000 )
    parser.add_argument('--lbn_3_constraint_gt', type=int, help='Budget value (minimum node number of label 1 in output selection)', default=-1_000_000 )
    parser.add_argument('--lbn_4_constraint_gt', type=int, help='Budget value (minimum node number of label 1 in output selection)', default=-1_000_000 )
    parser.add_argument('--lbn_5_constraint_gt', type=int, help='Budget value (minimum node number of label 1 in output selection)', default=-1_000_000 )
    parser.add_argument('--lbn_6_constraint_gt', type=int, help='Budget value (minimum node number of label 1 in output selection)', default=-1_000_000 )
    parser.add_argument('--lbn_n1_constraint_lt', type=int, help='Budget value (minimum node number of label -1 in output selection)', default=-1_000_000 )
    
    parser.add_argument('--ignore_size_1_cc', type=int, help='Indicator: whether to remove cc with size 1', default=1 )
    parser.add_argument('--ignore_cc_only_has_one_label', type=int, help='Indicator: whether to remove cc with one type of labels', default=0 )
    parser.add_argument('--save_learned_feat', type=int, help='Whether to save the learned node features', default=0)


    
    parser.add_argument('--stop_cut_group_size', type=int, help='Size can stop cutting a graph.', default=30)
    parser.add_argument('--is_simulate_atten', type=int, help='Whether to simulate attention?', default=1)
    parser.add_argument('--decay_from', type=float, help='Decay from what value.', default=0.78)
    parser.add_argument('--decay_to', type=float, help='Decay to what value.', default=0.48) # 0.5
    parser.add_argument('--decay_step', type=int, help='Number of steps to decay.', default=16) # 20
    parser.add_argument('--exp_scale', type=float, help='Exponential scale parameter', default=1.0)
    parser.add_argument('--temperature', type=float, help='Exponential scale parameter', default=0.5)
    parser.add_argument('--feature_dim', type=int, help='node feature dimension', default=3)
    parser.add_argument('--feature_mean', type=float, help='node feature mean', default=0.0)
    parser.add_argument('--feature_std', type=float, help='node feature std', default=1.0)
    args = parser.parse_args(args)
    return args


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

@class_register
class GraphLoader(DependencyController):
    def __init__(self, graph_name, graph_parent_path, **kwargs):
        super().__init__(**kwargs)
        self.confirmed_graph_options = ['GRPG_007_tr_te', 'elliptic','ppi_tr_te', 'cora-nb2708-tr','cora-nb5416-stellargraph-half-tr']
        self.graph_path = ''
        self.G_data = None
        self.g_tr_te = None
        self.graph_id = graph_name
        self.unk_lable = -1
        self.graph_parent_path = graph_parent_path
    
    @DependencyController.track
    @DependencyController.alias("2.1")
    @pyip_prompt_menu("Which graph do you want to load?\n", 'graph_name', 'confirmed_graph_options')
    def load_graph(self, graph_name=None, set_tr_te=True, **kwargs):
        # self.graph_id = graph_name
        # self.graph_path = f'/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/notebook/node_selection_experiment/suvi_20230113/{self.graph_id}'
        self.graph_path = Path(self.graph_parent_path).joinpath(Path(self.graph_id))
        self.G_data = json.load(open(str(self.graph_path) + "-G.json"))
        self.g_tr_te = json_graph.node_link_graph(self.G_data)
        
        if '_label' not in self.g_tr_te.nodes[0].keys():
            if type(self.g_tr_te.nodes[0]['label']) == list:
                for n in self.g_tr_te.nodes:
                    self.g_tr_te.nodes[n]['_label'] = np.argmax(self.g_tr_te.nodes[n]['label'])
            else:
                for n in self.g_tr_te.nodes:
                    self.g_tr_te.nodes[n]['_label'] = self.g_tr_te.nodes[n]['label']
                    
        self.df_g_tr_te_data = pd.DataFrame.from_dict(self.g_tr_te.nodes, orient='index')
        self.label_set = set(sorted(self.df_g_tr_te_data.query("real in [True, 1, 1.0]")['_label'].unique()))
        self.label_set_one_hot = np.eye(len(self.label_set)).tolist()
        self.label_map = {k: v for k,v in zip(self.label_set, self.label_set_one_hot) }
        self.label_map[self.unk_lable] = (np.zeros(len(self.label_set)) -1).tolist()

        if type(self.g_tr_te.nodes[0]['label']) != list:
            for n in self.g_tr_te.nodes:
                self.g_tr_te.nodes[n]['label'] = self.label_map[self.g_tr_te.nodes[n]['label']]


        assert np.unique(self.g_tr_te.nodes).shape[0] == len(self.g_tr_te.nodes)
        if set_tr_te:
            self.set_tr_te_graph()

    def set_tr_te_graph(self):
        self.node_tr_ls = [n for n in self.g_tr_te if self.g_tr_te.nodes[n]['test'] == False ]
        self.node_te_ls = [n for n in self.g_tr_te if self.g_tr_te.nodes[n]['test'] == True ]
        self.g_te = self.g_tr_te.subgraph(self.node_te_ls)
        self.g_tr = self.g_tr_te.subgraph(self.node_tr_ls)
        self.df_g_te_data = pd.DataFrame.from_dict(self.g_te.nodes, orient='index')
        self.df_g_tr_data = pd.DataFrame.from_dict(self.g_tr.nodes, orient='index')

    def get_ppi_graph(self, mode='train', task=0):
        from torch_geometric.datasets  import PPI
        dataset = PPI(root='/tmp/PPI', split=mode )
        _g = nx.from_edgelist( dataset.data.edge_index.T.numpy() )
        _g_node_set = set(_g.nodes)
        for n in range(len(dataset.data.x)):
            if n not in _g_node_set:
                continue
            _g.nodes[n]['feat'] = dataset.data.x[n].numpy()
            _g.nodes[n]['label'] = dataset.data.y[n].numpy()[task].item()
            _g.nodes[n]['real'] = True
            _g.nodes[n]['test'] = False
            _g.nodes[n]['val'] = False
            _g.nodes[n]['train'] = False
        if mode == 'train':
            for n in _g.nodes:
                _g.nodes[n]['train'] = True

        if mode == 'val':
            for n in _g.nodes:
                _g.nodes[n]['val'] = True

        if mode == 'test':
            for n in _g.nodes:
                _g.nodes[n]['test'] = True

        return _g

    def create_g_ppi_G(self, task=0, save=False):
        # task = 0
        g_ppi_tr = self.get_ppi_graph('train', task=task)
        g_ppi_te = self.get_ppi_graph('test', task=task)
        g_ppi_tr = nx.relabel_nodes(g_ppi_tr, {j: i for i,j in enumerate(g_ppi_tr.nodes) })
        _node_max_id = max(g_ppi_tr.nodes)
        g_ppi_te = nx.relabel_nodes(g_ppi_te, {j: i + _node_max_id + 1 for i,j in enumerate(g_ppi_te.nodes) })
        g_ppi_tr_te = nx.compose(g_ppi_tr, g_ppi_te)
        if save:
            dump_data = json_graph.node_link_data(g_ppi_tr_te)
            with open('ppi_tr_te-G.json', 'w') as outfile1:
                json.dump(dump_data, outfile1, cls=NpEncoder)    

    def show_graph_data_stats_1(self):
        return self.df_g_tr_te_data.groupby(['test','val','_label','real']).agg({'_label':len})



@class_register
class GraphSAGE_Caller(DependencyController):
    def __init__(self, graph_loader, attn_parent_path, **kwargs):
        super().__init__(**kwargs)
        self.FLAGS = supervised_train.flags.FLAGS
        self.FLAGS.base_log_dir = None
        self.FLAGS.mark_as_parsed()
        self.FLAGS.max_degree = 100
        self.FLAGS.samples_1 = 25
        self.FLAGS.samples_2 = 10
        self.FLAGS.batch_size = 580
        self.FLAGS.print_every = 1
        self.FLAGS.remove_isolated_nodes = False
        self.FLAGS.train_return_model_once_ready = False
        self.FLAGS.epochs = 50
        self.normalized = False 
        self.set_train_removed_by_rules = True # 🚩
        self.graph_loader = graph_loader
        # self.FLAGS.train_prefix = f'{self.graph_loader.graph_path.split("-G.json")[0]}'
        self.FLAGS.validate_batch_size = -1
        self.FLAGS.k_of_sb = [0]        
        self.FLAGS.dim_1 = 32
        self.FLAGS.dim_2 = 32
        self.FLAGS.learning_rate = 0.01
        self.FLAGS.model = 'graphsage_attn'
        
        self.model_folder_options = ['GRPG_007_02','elliptic_001_01','cora-nb5416-140-n1-stellargraph','cora-nb5416-half-n1-stellargraph','PROTEINS_node_cls','citeseer_node_cls']
        # Notes:
        # 'GRPG_007_03' -> Turn label column to type list instead of int
        # if attn_parent_path == None:
        #     attn_parent_path = f"/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/notebook/node_selection_experiment/suvi_20230113/"
        self._attn_path = attn_parent_path
        self._attn_options = ["GRPG_007_tr_te_attn_ind002.npy",
                              "elliptic_attn_ind001.npy"]
        
    

    def update_flags_value_by_model_folder(self):
        if self.selected_model_folder_name in ['GRPG_007_03', 'GRPG_007_02', 'GRPG_007_01']:
            self.FLAGS.T = 0.1

        if self.selected_model_folder_name in ['elliptic_001_01']:
            self.FLAGS.print_every = 60
            self.FLAGS.max_degree = 128
            self.FLAGS.samples_1 = 25
            self.FLAGS.samples_2 = 10
            self.FLAGS.batch_size = 256
            self.FLAGS.dim_1 = 128
            self.FLAGS.dim_2 = 128
            self.FLAGS.epochs = 200
            self.FLAGS.learning_rate = 0.001

        if self.selected_model_folder_name in ['PROTEINS_node_cls']:
            self.FLAGS.print_every = 60
            self.FLAGS.max_degree = 128
            self.FLAGS.samples_1 = 25
            self.FLAGS.samples_2 = 10
            self.FLAGS.batch_size = 256
            self.FLAGS.dim_1 = 4
            self.FLAGS.dim_2 = 4
            self.FLAGS.epochs = 200
            self.FLAGS.learning_rate = 0.001

        if self.selected_model_folder_name in ['cora-nb5416-140-n1-stellargraph', 'cora-nb5416-half-n1-stellargraph']:
            self.FLAGS.learning_rate = 0.005
            self.FLAGS.dropout = 0.5
            self.FLAGS.epochs = 100

        if self.selected_model_folder_name in ['citeseer_node_cls']:
            self.FLAGS.learning_rate = 0.005
            self.FLAGS.dropout = 0.5
            self.FLAGS.epochs = 100
           
    @DependencyController.track
    @DependencyController.alias("3.2.9")
    @pyip_prompt_menu("Select one folder name to save/load the mdoel.\n", 'selected_model', 'model_folder_options')
    def propmt_set_base_log_dir(self, selected_model, base_log_parent_dir='/home/yanghong/YH/Laurence/GS/notebook/node_selection_experiment/graphsage_model/'):
        print("Select one folder name to save/load the mdoel.")
        # confirm_children_path = input(f"Stage 3.4: What is the folder name of the model weights? (E.g., GRPG_001)")
        self.selected_model_folder_name = selected_model
        self.FLAGS.base_log_dir = f'{base_log_parent_dir}{self.selected_model_folder_name}/'
        self.update_flags_value_by_model_folder()

    @DependencyController.track
    @DependencyController.alias("3.1")
    def set_train_data(self):
        assert self.graph_loader.graph_path
        self.FLAGS.train_prefix = f'{str(self.graph_loader.graph_path).split("-G.json")[0]}'
        self.train_data = supervised_train.load_data(self.FLAGS.train_prefix, normalize=self.normalized, remove_isolated_nodes=self.FLAGS.remove_isolated_nodes, remove_cc_by_rules=False, set_train_removed_by_rules=self.set_train_removed_by_rules, graph_loader=self.graph_loader)
        self.minibatch_it = post_graphsage_trained.get_tr_iter(self.train_data, self.FLAGS, NodeMinibatchIterator)


    @DependencyController.require(["3.2.9", "3.2"])
    @DependencyController.track
    @DependencyController.alias("3.4")
    def run_train(self):
        supervised_train.train(self.train_data)

    @DependencyController.require("3.1")
    @DependencyController.track
    @DependencyController.alias("3.2")
    def set_model(self):
        _old = self.FLAGS.train_return_model_once_ready
        self.FLAGS.train_return_model_once_ready = True
        self.model = supervised_train.train(self.train_data)
        self.FLAGS.train_return_model_once_ready = _old
        print('Model is set.')

    @DependencyController.require(["3.2.9", "3.2"])
    @DependencyController.track
    @DependencyController.alias("3.3")
    def model_load_weight(self):
        # tmp_path = self._attn_path.joinpath(self.selected_model_folder_name)
        # tmp_path = self.selected_model_folder_name
        tmp_path = self.FLAGS.base_log_dir
        tmp_options = sorted(glob.glob(f"{tmp_path}/*/*/*.index"))[-3:]
        print("Please select one of the following model weight")
        self.selected_model_path = pyip.inputMenu(tmp_options, numbered=True).split('.index')[0]
        # _model_path_default = '/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/notebook/node_selection_experiment/suvi_20230113/Cora_000_01/sup-suvi_20230113/graphsage_attn_small_0.0100_13-03-2023-10:17:52/weights.044val_precision-0.698-val_recall-0.744-val_f1-0.712'
        # self.selected_model_path = input(f"Stage 4.3: What is the model path (E.g., {_model_path_default})?") or _model_path_default
        self.model.load_weights(self.selected_model_path)


    @DependencyController.require("3.3")
    @DependencyController.track
    @DependencyController.alias("3.5")
    def set_df_gs(self, include_test=True):
        # Laurence 20230213 Very Important: Switch to testing context for data generator>>>
        for l in self.model.layer_infos:  # 
            l.neigh_sampler.adj_info = self.minibatch_it.test_adj
        # Laurence 20230213 Very Important: Switch to testing context for data generator<<<

        while True:
            val_cost, val_f1_mic, val_f1_mac, val_acc, val_precision, val_recall, val_f1, duration = supervised_train.incremental_evaluate(self.model, self.minibatch_it, self.FLAGS.batch_size, test=True)
            confirm_val_rs = input(f"Stage 4.4: Is the evaluation result ok? [{val_precision:.3f}, {val_recall:.3f}, {val_f1:.3f}]") or "Y"
            if confirm_val_rs == 'Y':
                break
        
        self.df_gs = post_graphsage_trained.get_df_gs_info(self.train_data, self.FLAGS, NodeMinibatchIterator, self.model, nb_iter=1, include_test=include_test )

    @DependencyController.require("3.5")
    @DependencyController.track
    @DependencyController.alias("3.6")
    def set_attn(self):
        self.attn = dict()
        for row in self.df_gs.to_dict("records"):
            attn_agg_0_hop_0 = np.array(row['attn_agg_0_hop_0']).mean(axis=0)[0,:]
            attn_agg_0_hop_1 = np.array(row['attn_agg_0_hop_1']).mean(axis=1)[:,0,:]
            n1_list = row['neigh_0']
            n1_to_n2_list = row['neigh_1']
            n = row['id']
            self.attn.setdefault(n, defaultdict(float))
            for i in range(attn_agg_0_hop_0.shape[0]):
                self.attn[n][tuple(sorted([n1_list[i], n ]))] += attn_agg_0_hop_0[i]
                for j in range(attn_agg_0_hop_1.shape[1]):
                    
                    self.attn[n][tuple(sorted([n1_list[i], n1_to_n2_list[i][j] ]))] += attn_agg_0_hop_0[i] * attn_agg_0_hop_1[i][j]

        print("Attention Created.")
        self.save_attn_suffix = input(f"Stage 4.5: What is the name suffix of the saved attention? E.g., ind001 --> {self.graph_loader.graph_id}_attn_ind001")
        _npy_path = f"{self.graph_loader.graph_id}_attn_{self.save_attn_suffix}.npy"
        if input(f"Do you confirm to save a npy as {_npy_path}") == 'Y':
            np.save(_npy_path, self.attn)

        _df_path = f"df_{self.graph_loader.graph_id}_attn_{self.save_attn_suffix}.pkl"
        if input(f"Do you confirm to save a dataframe as {_df_path}") == 'Y':
            self.df_gs.to_pickle(_df_path)
            print("Attention Saved.")

    @DependencyController.track
    @DependencyController.alias("3.7")
    @pyip_prompt_menu("Which atten file do you want to load?\n", 'selected_attn_path', '_attn_options')
    def load_attn(self, selected_attn_path, **kwargs):
        assert self.FLAGS.model == 'graphsage_attn'
        # _attn_path_default = f"/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/notebook/node_selection_experiment/suvi_20230113/{graph_attn_prefix_id}{graph_prefix_mode}_attn_ind001.npy"
        # self.selected_attn_path = pyip.inputMenu(self._attn_options, numbered=True)
        self.selected_attn_path = selected_attn_path
        if 'version' in kwargs.keys() and kwargs['version'] == 2:
            self.attn = np.load(self.selected_attn_path, allow_pickle=True).item()
            self.df_gs = pd.read_pickle(kwargs['df_gs_path'])
        else:
            self.attn = np.load(self._attn_path.joinpath(self.selected_attn_path), allow_pickle=True).item()
            self.df_gs = pd.read_pickle(self._attn_path.joinpath(f"df_{self.selected_attn_path.replace('.npy','.pkl')}"))





@class_register
class GraphSample_Caller(DependencyController):
    def __init__(self, graph_loader, graphsage_caller, df_reduce_graph_rs_path=None, reduce_id=None, **kwargs):
        super().__init__(**kwargs)
        self.graph_loader = graph_loader
        self.graphsage_caller = graphsage_caller

        if df_reduce_graph_rs_path:
            self.set_df_reduce_graph_rs_path(df_reduce_graph_rs_path)
        if reduce_id:
            self.set_reduce_id(reduce_id)
        # self.g_tr = nx.Graph(self.graph_loader.g_tr)
        # self.df_g_tr_data = self.graph_loader.df_g_tr_data
        self.atten_mode = 'attn_sum'

        self.arg_str_options = []
        self.arg_str_options += ["--B 250 " +\
            "--stop_cut_group_size 50"]
        self.arg_str_options += ["--B 150 " +\
            "--stop_cut_group_size 30"]
        self.arg_str_options += ["--B 1000 " +\
            "--stop_cut_group_size 200"]
        # self.set_edge_attn_attr()


    @DependencyController.require("3.7@graphsage_caller")
    @DependencyController.track
    @DependencyController.alias("4.2")
    def set_edge_attn_attr(self):
        edge_acc_attn_sum = defaultdict(float)
        for n, n_ego_dict in self.graphsage_caller.attn.items():    
            if n not in self.graph_loader.g_tr.nodes:
                continue
            
            for e, e_attn in n_ego_dict.items():
                edge_acc_attn_sum[e] += e_attn 
        nx.set_edge_attributes(self.graph_loader.g_tr,edge_acc_attn_sum,'attn_sum')

    @DependencyController.track
    @DependencyController.alias("4.1")
    @pyip_prompt_menu("What is the arg to reduce the graph?", 'arg_str', 'arg_str_options')
    @experiment_track(exp_param_ls=[])
    def set_args(self, arg_str, **kwargs):
        self.selected_arg_str = str.strip(arg_str)
        # self.selected_arg_str = pyip.inputMenu(self.arg_str_options, numbered=True)
        self.arg_ls = [str.strip(a) for a in self.selected_arg_str.split(' ')]
        self.args = get_args(self.arg_ls)

    def get_df_summary_for_method(self, method_name, retain, selected_nodes, df_g_data):
            # _df_one = {'name':[], 'selected_nodes':[], 'R_attn':[], 'pos_num':[], 'neg_num':[], }
            _df_one = {}
            _df_one['name'] = method_name
            _df_one['selected_nodes'] = [selected_nodes]
            _df_one['R_attn'] = retain

            if method_name in ['FF','Ours (sum)','Ours (max)','Ours (large)','SB','SRW','RWF', 'ISRW']:
                # Laurence 20230304 >>>
                _unique_labels = sorted(df_g_data['_label'].unique().tolist())
                for lbl in _unique_labels:
                    _df_one[f'label_{lbl}'] =  df_g_data.loc[selected_nodes].query(f'_label=={lbl}').shape[0]
                # Laurence 20230304 <<<

                _df_one['total_num'] =  df_g_data.loc[selected_nodes].shape[0] 
                _df_one = pd.DataFrame(_df_one)

            else:
                raise ValueError(f"Unknow method {method_name}")
            return _df_one
        
    def get_R_attn(self, selected_nodes, atten_mode):
        r_attn = sum(nx.get_edge_attributes(self.graph_loader.g_tr.subgraph(selected_nodes),atten_mode).values())/sum(nx.get_edge_attributes(self.graph_loader.g_tr,atten_mode).values())
        return r_attn

    def _get_df_on_different_method(self, method_name, func_select):
        if len(self.graph_loader.g_tr.nodes) < self.args.B:
            _sampled_graph = self.graph_loader.g_tr
        else:
            _sampled_graph = func_select(self.graph_loader.g_tr)
        
        _df_rs = None
        # for _attn_mode in atten_mode_options:
        # if _attn_mode == 'all':
        #     continue
        r_attn_ff = self.get_R_attn(_sampled_graph.nodes, self.atten_mode)
        _selected_nodes = list(_sampled_graph.nodes)
        if len(_selected_nodes) > self.args.B:
            _selected_nodes = _selected_nodes[:self.args.B]

        _df = self.get_df_summary_for_method(method_name, r_attn_ff, _selected_nodes, self.graph_loader.df_g_tr_data)
        _new_col_name = f'R_{self.atten_mode}'
        _df = _df.rename(columns={'R_attn':_new_col_name})
        if type(_df_rs).__name__ == 'NoneType':
            _df_rs = _df
        else:
            _df_rs = _df_rs.assign(**{_new_col_name: _df[_new_col_name] })
        # df_cmp_ls.append()
        return _df_rs
    
    
    @DependencyController.require(["4.2","4.1"])
    @DependencyController.track
    @DependencyController.alias("4.3")
    @pyip_prompt_input_num("How many times you want to repeat to reduce graph (using our method)? [0-999]", 'repeat_ours_exp_num')
    @experiment_track(exp_param_ls=["repeat_ours_exp_num"])
    def repeat_run_our_reduce_graph(self, repeat_ours_exp_num, **kwargs):
        # repeat_ours_exp_num = int(input("How many times you want to repeat to reduce graph (as there are randomness)? [0-999]") or 1 ) 
        df_ls = []        
        for i in tqdm.tqdm(range(repeat_ours_exp_num)):
            # Laurence 20230225 >>>
            # if 'attn_sum' in atten_mode_options:
            retain, meta = self.reduce_graph(return_meta=True)
            df_ls.append(self.get_df_summary_for_method('Ours (sum)', retain, meta['selected_nodes'], self.graph_loader.df_g_tr_data).rename(columns={'R_attn':'R_attn_sum'}))
            df_cc_map = meta['df_cc_map']
            # Laurence 20230225 <<<

        _df_rs = pd.concat(df_ls).reset_index(drop=True).assign(ours=1)
        _df_rs.attrs['args'] = self.selected_arg_str
        _df_rs.attrs['repeat_ours_exp_num'] = repeat_ours_exp_num
        # subgraph_ls = []
        # for selection in _df_rs['selected_nodes']:
        #     subgraph_ls.append(self.graph_loader.g_tr.subgraph(selection))
        # _df_rs.attrs['subgraph_ls'] = subgraph_ls
        self.set_df_our_reduce_graph_rs(_df_rs)

    @DependencyController.track
    @DependencyController.alias("4.3.1")    
    def set_df_our_reduce_graph_rs(self, df):
        self.df_our_reduce_graph_rs = df

    @DependencyController.require(["4.2","4.1"])
    @DependencyController.track
    @DependencyController.alias("4.4")
    @pyip_prompt_input_num("How many times you want to repeat to reduce graph (using other methods)? [0-999]", 'repeat_others_exp_num')
    @experiment_track(exp_param_ls=["repeat_others_exp_num"])
    def repeat_run_other_reduce_graph(self, repeat_others_exp_num, **kwargs):

        # repeat_others_exp_num = int(input("How many times you want to repeat to reduce graph (as there are randomness)? [0-999]") or 1 ) 

        df_ls = []        
        # tqdm.tqdm = functools.partial(tqdm.tqdm, disable=False)
        for i in tqdm.tqdm(range(repeat_others_exp_num)):
    
            # func_select = lambda x: reduce_graph_lib.Graph_Sampling.ForestFire().forestfire(x, args.B)
            _ff = gs.ForestFire()
            _func = lambda x: _ff.forestfire(x, self.args.B)
            df_ls.append(self._get_df_on_different_method('FF', _func))

            obj1 = gs.SRW_RWF_ISRW()
            _func = lambda x: obj1.random_walk_sampling_simple(x, self.args.B)
            df_ls.append(self._get_df_on_different_method('SRW', _func))

            _func = lambda x: obj1.random_walk_sampling_with_fly_back(x, self.args.B, 0.2)
            df_ls.append(self._get_df_on_different_method('RWF', _func))

            _func = lambda x: obj1.random_walk_induced_graph_sampling(x, self.args.B)
            df_ls.append(self._get_df_on_different_method('ISRW', _func))

            _func = lambda x: gs.Snowball().snowball(x, self.args.B, 15)
            df_ls.append(self._get_df_on_different_method('SB', _func))       
            
        # tqdm.tqdm = functools.partial(tqdm.tqdm, disable=False)
        # self.df_other_reduce_graph_rs = pd.concat(df_ls).reset_index(drop=True).assign(ours=0)
        _df_rs = pd.concat(df_ls).reset_index(drop=True).assign(ours=0)
        _df_rs.attrs['repeat_others_exp_num'] = repeat_others_exp_num
        self.set_df_other_reduce_graph_rs(_df_rs)

    
    @DependencyController.track
    @DependencyController.alias("4.4.2")    
    def set_df_other_reduce_graph_rs(self, df):
        self.df_other_reduce_graph_rs = df

    @DependencyController.require(["4.4.2"])
    @DependencyController.track
    @DependencyController.alias("4.4.4")    
    def show_df_other_reduce_graph_rs_static_1(self):
        display(self.df_other_reduce_graph_rs.groupby('name').agg({"R_attn_sum":np.mean, "label_0":np.mean,"label_1":np.mean}))
        
    @DependencyController.require(["4.3.1"])
    @DependencyController.track
    @DependencyController.alias("4.4.5")    
    def show_df_our_reduce_graph_rs_static_1(self):
        display(self.df_our_reduce_graph_rs.groupby('name').agg({"R_attn_sum":np.mean, "label_0":np.mean,"label_1":np.mean}))
        


    
    def knapsack_flex(self, feed_dict):
        """
        {
        'items':..., 'values': ..., 
        'weights':..., 'weight_constraint': ..., 
        'label_pos_num': ..., 'label_pos_num_constraint_max': ..., 
        'label_neg_num': ..., 'label_neg_num_constraint_max': ...,
        'constraint_ls': [
            'weight_constraint',
            'label_pos_num_constraint_max',
            'label_pos_num_constraint_min',
            'label_neg_num_constraint_max',
            'label_neg_num_constraint_min',
            ]
        }
        """
        model = pyo.ConcreteModel()
        items = feed_dict['items']
        model.I = pyo.Set(initialize=items)
        model.c = pyo.Param(model.I, initialize=feed_dict['values'])
        model.x = pyo.Var(model.I, within=pyo.Binary)

        if 'weight_constraint' in feed_dict['constraint_ls']:
            def weight_constraint_func(model):
                return  sum(model.x[i]*model.w[i] for i in model.I) <= model.w_constrain
            model.w = pyo.Param(model.I, initialize=feed_dict['weights'])
            model.w_constrain = pyo.Param(initialize=feed_dict['weight_constraint'])
            model.weight_constraint_func = pyo.Constraint(rule=weight_constraint_func)

        if 'lbn_0_constraint' in feed_dict['constraint_ls']:
            def lbn_0_constraint_func(model):
                return  sum(model.x[i]*model.lbn_0[i] for i in model.I) <= model.lbn_0_constrain
            model.lbn_0 = pyo.Param(model.I, initialize=feed_dict['lbn_0'])
            model.lbn_0_constrain = pyo.Param(initialize=feed_dict['lbn_0_constraint'])
            model.lbn_0_constraint_func = pyo.Constraint(rule=lbn_0_constraint_func)

        if 'lbn_1_constraint' in feed_dict['constraint_ls']:
            def lbn_1_constraint_func(model):
                return  sum(model.x[i]*model.lbn_1[i] for i in model.I) <= model.lbn_1_constrain
            model.lbn_1 = pyo.Param(model.I, initialize=feed_dict['lbn_1'])
            model.lbn_1_constrain = pyo.Param(initialize=feed_dict['lbn_1_constraint'])
            model.lbn_1_constraint_func = pyo.Constraint(rule=lbn_1_constraint_func)        
        
        if 'lbn_2_constraint' in feed_dict['constraint_ls']:
            def lbn_2_constraint_func(model):
                return  sum(model.x[i]*model.lbn_2[i] for i in model.I) <= model.lbn_2_constrain
            model.lbn_2 = pyo.Param(model.I, initialize=feed_dict['lbn_2'])
            model.lbn_2_constrain = pyo.Param(initialize=feed_dict['lbn_2_constraint'])
            model.lbn_2_constraint_func = pyo.Constraint(rule=lbn_2_constraint_func)          
        if 'lbn_3_constraint' in feed_dict['constraint_ls']:
            def lbn_3_constraint_func(model):
                return  sum(model.x[i]*model.lbn_3[i] for i in model.I) <= model.lbn_3_constrain
            model.lbn_3 = pyo.Param(model.I, initialize=feed_dict['lbn_3'])
            model.lbn_3_constrain = pyo.Param(initialize=feed_dict['lbn_3_constraint'])
            model.lbn_3_constraint_func = pyo.Constraint(rule=lbn_3_constraint_func)    
        if 'lbn_4_constraint' in feed_dict['constraint_ls']:
            def lbn_4_constraint_func(model):
                return  sum(model.x[i]*model.lbn_4[i] for i in model.I) <= model.lbn_4_constrain
            model.lbn_4 = pyo.Param(model.I, initialize=feed_dict['lbn_4'])
            model.lbn_4_constrain = pyo.Param(initialize=feed_dict['lbn_4_constraint'])
            model.lbn_4_constraint_func = pyo.Constraint(rule=lbn_4_constraint_func)  
        if 'lbn_5_constraint' in feed_dict['constraint_ls']:
            def lbn_5_constraint_func(model):
                return  sum(model.x[i]*model.lbn_5[i] for i in model.I) <= model.lbn_5_constrain
            model.lbn_5 = pyo.Param(model.I, initialize=feed_dict['lbn_5'])
            model.lbn_5_constrain = pyo.Param(initialize=feed_dict['lbn_5_constraint'])
            model.lbn_5_constraint_func = pyo.Constraint(rule=lbn_5_constraint_func)  
        if 'lbn_6_constraint' in feed_dict['constraint_ls']:
            def lbn_6_constraint_func(model):
                return  sum(model.x[i]*model.lbn_6[i] for i in model.I) <= model.lbn_6_constrain
            model.lbn_6 = pyo.Param(model.I, initialize=feed_dict['lbn_6'])
            model.lbn_6_constrain = pyo.Param(initialize=feed_dict['lbn_6_constraint'])
            model.lbn_6_constraint_func = pyo.Constraint(rule=lbn_6_constraint_func) 
                                                


        if 'lbn_n1_constraint' in feed_dict['constraint_ls']:
            def lbn_n1_constraint_func(model):
                return  sum(model.x[i]*model.lbn_n1[i] for i in model.I) <= model.lbn_n1_constrain
            model.lbn_n1 = pyo.Param(model.I, initialize=feed_dict['lbn_-1'])
            model.lbn_n1_constrain = pyo.Param(initialize=feed_dict['lbn_n1_constraint'])
            model.lbn_n1_constraint_func = pyo.Constraint(rule=lbn_n1_constraint_func)
            
        if 'lbn_0_constraint_gt' in feed_dict['constraint_ls']:
            def lbn_0_constraint_gt_func(model):
                return  sum(model.x[i]*model.lbn_0[i] for i in model.I) >= model.lbn_0_constrain_gt
            model.lbn_0 = pyo.Param(model.I, initialize=feed_dict['lbn_0'])
            model.lbn_0_constrain_gt = pyo.Param(initialize=feed_dict['lbn_0_constraint_gt'])
            model.lbn_0_constraint_gt_func = pyo.Constraint(rule=lbn_0_constraint_gt_func)

        if 'lbn_1_constraint_gt' in feed_dict['constraint_ls']:
            def lbn_1_constraint_gt_func(model):
                return  sum(model.x[i]*model.lbn_1[i] for i in model.I) >= model.lbn_1_constrain_gt
            model.lbn_1 = pyo.Param(model.I, initialize=feed_dict['lbn_1'])
            model.lbn_1_constrain_gt = pyo.Param(initialize=feed_dict['lbn_1_constraint_gt'])
            model.lbn_1_constraint_gt_func = pyo.Constraint(rule=lbn_1_constraint_gt_func)

        if 'lbn_2_constraint_gt' in feed_dict['constraint_ls']:
            def lbn_2_constraint_gt_func(model):
                return  sum(model.x[i]*model.lbn_2[i] for i in model.I) >= model.lbn_2_constrain_gt
            model.lbn_2 = pyo.Param(model.I, initialize=feed_dict['lbn_2'])
            model.lbn_2_constrain_gt = pyo.Param(initialize=feed_dict['lbn_2_constraint_gt'])
            model.lbn_2_constraint_gt_func = pyo.Constraint(rule=lbn_2_constraint_gt_func)

        if 'lbn_3_constraint_gt' in feed_dict['constraint_ls']:
            def lbn_3_constraint_gt_func(model):
                return  sum(model.x[i]*model.lbn_3[i] for i in model.I) >= model.lbn_3_constrain_gt
            model.lbn_3 = pyo.Param(model.I, initialize=feed_dict['lbn_3'])
            model.lbn_3_constrain_gt = pyo.Param(initialize=feed_dict['lbn_3_constraint_gt'])
            model.lbn_3_constraint_gt_func = pyo.Constraint(rule=lbn_3_constraint_gt_func)

        if 'lbn_4_constraint_gt' in feed_dict['constraint_ls']:
            def lbn_4_constraint_gt_func(model):
                return  sum(model.x[i]*model.lbn_4[i] for i in model.I) >= model.lbn_4_constrain_gt
            model.lbn_4 = pyo.Param(model.I, initialize=feed_dict['lbn_4'])
            model.lbn_4_constrain_gt = pyo.Param(initialize=feed_dict['lbn_4_constraint_gt'])
            model.lbn_4_constraint_gt_func = pyo.Constraint(rule=lbn_4_constraint_gt_func)

        if 'lbn_5_constraint_gt' in feed_dict['constraint_ls']:
            def lbn_5_constraint_gt_func(model):
                return  sum(model.x[i]*model.lbn_5[i] for i in model.I) >= model.lbn_5_constrain_gt
            model.lbn_5 = pyo.Param(model.I, initialize=feed_dict['lbn_5'])
            model.lbn_5_constrain_gt = pyo.Param(initialize=feed_dict['lbn_5_constraint_gt'])
            model.lbn_5_constraint_gt_func = pyo.Constraint(rule=lbn_5_constraint_gt_func)

        if 'lbn_6_constraint_gt' in feed_dict['constraint_ls']:
            def lbn_6_constraint_gt_func(model):
                return  sum(model.x[i]*model.lbn_6[i] for i in model.I) >= model.lbn_6_constrain_gt
            model.lbn_6 = pyo.Param(model.I, initialize=feed_dict['lbn_6'])
            model.lbn_6_constrain_gt = pyo.Param(initialize=feed_dict['lbn_6_constraint_gt'])
            model.lbn_6_constraint_gt_func = pyo.Constraint(rule=lbn_6_constraint_gt_func)
                                                                        

        # if 'lbn_n1_constraint_lt' in feed_dict['constraint_ls']:
        #     def lbn_n1_constraint_lt_func(model):
        #         return  sum(model.x[i]*model.lbn_n1[i] for i in model.I) <= model.lbn_n1_constrain_gt
        #     model.lbn_n1 = pyo.Param(model.I, initialize=feed_dict['lbn_-1'])
        #     model.lbn_n1_constrain_gt = pyo.Param(initialize=feed_dict['lbn_n1_constraint_lt'])
        #     model.lbn_n1_constraint_lt_func = pyo.Constraint(rule=lbn_n1_constraint_lt_func)            

            # if constrain == 'label_pos_num_constraint_max':
            #     def label_pos_num_constraint_max_func(model):
            #         return sum(model.x[i]*model.lpn[i] for i in model.I) <= model.lpn_constrain_max
            #     model.lpn_constrain_max = pyo.Param(initialize=feed_dict['label_pos_num_constraint_max'])
            #     model.label_pos_num_constraint_max_func = pyo.Constraint(rule=label_pos_num_constraint_max_func)

            # if constrain == 'label_pos_num_constraint_min':
            #     def label_pos_num_constraint_min_func(model):
            #         return sum(model.x[i]*model.lpn[i] for i in model.I) >= model.lpn_constrain_min
            #     model.lpn_constrain_min = pyo.Param(initialize=feed_dict['label_pos_num_constraint_min'])
            #     model.label_pos_num_constraint_min_func = pyo.Constraint(rule=label_pos_num_constraint_min_func)

            # if constrain == 'label_neg_num_constraint_max':
            #     def label_neg_num_constraint_max_func(model):
            #         return sum(model.x[i]*model.lnn[i] for i in model.I) <= model.lnn_constrain_max
            #     model.lnn_constrain_max = pyo.Param(initialize=feed_dict['label_neg_num_constraint_max'])
            #     model.label_neg_num_constraint_max_func = pyo.Constraint(rule=label_neg_num_constraint_max_func)

            # if constrain == 'label_neg_num_constraint_min':
            #     def label_neg_num_constraint_min_func(model):
            #         return sum(model.x[i]*model.lnn[i] for i in model.I) >= model.lnn_constrain_min
            #     model.lnn_constrain_min = pyo.Param(initialize=feed_dict['label_neg_num_constraint_min'])
            #     model.label_neg_num_constraint_min_func = pyo.Constraint(rule=label_neg_num_constraint_min_func)

        def obj_function(model):
            return sum(model.x[i]*model.c[i] for i in model.I)
            
        self.pymo_model = model
        model.objective = pyo.Objective(rule=obj_function, sense=pyo.maximize)
        opt = SolverFactory('glpk')
        opt.solve(model)
        return pyo.value(model.objective),[i for i in model.I if pyo.value(model.x[i])==1]

    def get_g_cc_ls(self, edge_attr_name='attn_sum'):
        def is_unlabel_ratio_valid(count_dict):
            if count_dict[-1] >= args.stop_cut_group_size * args.unlabel_ratio_max:
                return False
            else:
                return True
            
        args, g, df_g_data,  = self.args, self.graph_loader.g_tr, self.graph_loader.df_g_tr_data
        g_edge_items = nx.get_edge_attributes(g, edge_attr_name).items()
        self.sorted_edges = sorted(list(g_edge_items), key=lambda x: x[1], reverse=True)

        node_to_cluster = {}
        cluster_to_nodes = {}
        cluster_info = {}
        cluster_nb = 0
        unique_label_ls = df_g_data['_label'].unique()
        has_unk_label = (-1 in unique_label_ls)
        for i, (e, attn) in enumerate(self.sorted_edges):
            n1 = e[0]
            n2 = e[1]
                    
            if (n1 not in node_to_cluster) and (n2 not in node_to_cluster):
                cluster_nb += 1
                node_to_cluster[n1] = cluster_nb
                node_to_cluster[n2] = cluster_nb
                cluster_to_nodes[cluster_nb] = set({n1, n2})
                cluster_info[cluster_nb] = {'count':{lbl: 0 for lbl in unique_label_ls }}
                _n1_label = g.nodes[n1]['_label']
                _n2_label = g.nodes[n2]['_label']
                cluster_info[cluster_nb]['count'][_n1_label] += 1
                cluster_info[cluster_nb]['count'][_n2_label] += 1
                

            elif (n1 not in node_to_cluster) and (n2 in node_to_cluster):
                c2 = node_to_cluster[n2]
                _l2 = len(cluster_to_nodes[c2])
                _label = g.nodes[n1]['_label']
                cond1 = _l2 >= args.stop_cut_group_size
                if has_unk_label:
                    tmp_dict = {-1: cluster_info[c2]['count'][-1] + 1}
                    cond2 = (_label == -1) and (not is_unlabel_ratio_valid(tmp_dict))
                    assert is_unlabel_ratio_valid(cluster_info[c2]['count'])
                else:
                    cond2 = False
                if cond1 or cond2:
                    cluster_nb += 1
                    cluster_to_nodes[cluster_nb] = set({n1})
                    cluster_info[cluster_nb] = {'count':{lbl: 0 for lbl in unique_label_ls }}
                    cluster_info[cluster_nb]['count'][_label] += 1
                    node_to_cluster[n1] = cluster_nb # Laurence 20230304
                    continue

                node_to_cluster[n1] = c2
                cluster_to_nodes[c2] |= set({n1})
                cluster_info[c2]['count'][_label] += 1
                

            elif (n1 in node_to_cluster) and (n2 not in node_to_cluster):
                c1 = node_to_cluster[n1]
                _l1 = len(cluster_to_nodes[c1])
                _label = g.nodes[n2]['_label']
                
                cond1 = _l1 >= args.stop_cut_group_size
                if has_unk_label:
                    tmp_dict = {-1: cluster_info[c1]['count'][-1] + 1}
                    assert is_unlabel_ratio_valid(cluster_info[c1]['count'])
                    cond2 = (_label == -1) and (not is_unlabel_ratio_valid(tmp_dict))
                else:
                    cond2 = False
                if cond1 or cond2:
                    cluster_nb += 1
                    cluster_to_nodes[cluster_nb] = set({n2})
                    # cluster_info[cluster_nb] = {'count':{0:0, -1:0, 1:0}}
                    cluster_info[cluster_nb] = {'count':{lbl: 0 for lbl in unique_label_ls }}
                    cluster_info[cluster_nb]['count'][_label] += 1
                    node_to_cluster[n2] = cluster_nb # Laurence 20230304
                    continue
                
                node_to_cluster[n2] = c1
                cluster_to_nodes[c1] |= set({n2})
                cluster_info[c1]['count'][_label] += 1
                

            elif (n1 in node_to_cluster) and (n2 in node_to_cluster):
                c1 = node_to_cluster[n1]
                c2 = node_to_cluster[n2]
                _l1 = len(cluster_to_nodes[c1])
                _l2 = len(cluster_to_nodes[c2])
                if c1 == c2:
                    continue                
                
                if has_unk_label:
                    tmp_dict = {-1: cluster_info[c1]['count'][-1] + cluster_info[c2]['count'][-1]}
                    cond2 = is_unlabel_ratio_valid(tmp_dict)
                else:
                    cond2 = True
                # Laurence 20230316 align to suvi's >>>
                # if _l1 + _l2 <= args.stop_cut_group_size and cond2: 
                if _l1 <= args.stop_cut_group_size and _l2 <= args.stop_cut_group_size and cond2:
                # Laurence 20230316 align to suvi's <<<
                       
                    # Update node_to_cluster
                    for tmp_n2 in cluster_to_nodes[c2]:
                        node_to_cluster[tmp_n2] = c1
                    node_to_cluster[n2] = c1

                    # Update cluster_to_nodes
                    cluster_to_nodes[c1] = cluster_to_nodes[c1] | cluster_to_nodes[c2]
                    cluster_to_nodes[c2] = set({})

                    # Update cluster_info info
                    for _label in cluster_info[c2]['count']:
                        cluster_info[c1]['count'][_label] += cluster_info[c2]['count'][_label]
                    cluster_info[c2] = {'del':'del'}

                if has_unk_label:
                    assert is_unlabel_ratio_valid(cluster_info[c1]['count'])


        g_cc_ls = []
        new_cluster_id = 0
        new_cluster_to_nodes = {}
        new_cluster_info = {}
        for old_cluster_id, cc in cluster_to_nodes.items():
            if len(cc) > 0:
                new_cluster_to_nodes[new_cluster_id] = cc
                new_cluster_info[new_cluster_id] = cluster_info[old_cluster_id]
                g_cc_ls.append(cc)
                new_cluster_id += 1

        return g_cc_ls, new_cluster_info, new_cluster_to_nodes
    
    def add_residual_nodes(self, sorted_edges, selected_nodes, args):
        for e,_ in sorted_edges:
            if (e[0] not in selected_nodes) and (e[1] in selected_nodes) and (len(selected_nodes) < args.B):
                selected_nodes.append(e[0])
            elif (e[1] not in selected_nodes) and (e[0] in selected_nodes) and (len(selected_nodes) < args.B):
                selected_nodes.append(e[1])
            if  len(selected_nodes) >= args.B:
                break

    def set_df_cc_ls(self, g, g_cc_ls, edge_attr_name='attn_sum'):
        
        ignore_size_1_cc = self.args.ignore_size_1_cc
        ignore_cc_only_has_one_label = self.args.ignore_cc_only_has_one_label

        label_set = set(self.graph_loader.df_g_tr_data['_label'].unique())
        attn_ls = []
        label_num_ls_dict = {l: [] for l in label_set}
        for g_cc in g_cc_ls:
            subg_cc = g.subgraph(g_cc)
            subg_cc_attn = sum(list(nx.get_edge_attributes(subg_cc, edge_attr_name).values()))
            _label_ls = list(nx.get_node_attributes(subg_cc, '_label').values()) 
            for l in label_set:
                label_num_ls_dict[l].extend([0])

            for l in _label_ls:
                label_num_ls_dict[l][-1] += 1
            attn_ls.append(subg_cc_attn)

        cc_attn_map ={
            'cc':g_cc_ls,
            'node_num':[len(cc) for cc in g_cc_ls],
            edge_attr_name: attn_ls, 
        }
        for l in label_set:
            cc_attn_map[f'lbn_{l}'] = label_num_ls_dict[l]

        df_cc_map = pd.DataFrame(cc_attn_map)
        
        # Laurence 20230314 >>>
        if ignore_cc_only_has_one_label:
            df_cc_map = df_cc_map.query(f"node_num > `lbn_{-1}`") 
        if ignore_size_1_cc:
            df_cc_map = df_cc_map.query(f"node_num > 1") 
        
        df_cc_map = df_cc_map.reset_index(drop=True)
        # Laurence 20230314 <<<
        self.df_cc_map = df_cc_map


    def reduce_graph(self, edge_attr_name='attn_sum', return_meta=False):
        
        g_edge_items = nx.get_edge_attributes(self.graph_loader.g_tr, edge_attr_name).items()
        sorted_edges = sorted(list(g_edge_items), key=lambda x: x[1], reverse=True)
        g_cc_ls, new_cluster_info, new_cluster_to_nodes = self.get_g_cc_ls(edge_attr_name='attn_sum')
        self.set_df_cc_ls(self.graph_loader.g_tr, g_cc_ls)

        feed_dict = {
            'items':self.df_cc_map.index, 'values': self.df_cc_map[edge_attr_name], 
            'weights':self.df_cc_map.node_num, 'weight_constraint': self.args.B, 
            'lbn_0_constraint': self.args.lbn_0_constraint,
            'lbn_1_constraint': self.args.lbn_1_constraint,
            'lbn_2_constraint': self.args.lbn_2_constraint,
            'lbn_3_constraint': self.args.lbn_3_constraint,
            'lbn_4_constraint': self.args.lbn_4_constraint,
            'lbn_5_constraint': self.args.lbn_5_constraint,
            'lbn_6_constraint': self.args.lbn_6_constraint,
            'lbn_0_constraint_gt': self.args.lbn_0_constraint_gt,
            'lbn_1_constraint_gt': self.args.lbn_1_constraint_gt,            
            'lbn_2_constraint_gt': self.args.lbn_2_constraint_gt,            
            'lbn_3_constraint_gt': self.args.lbn_3_constraint_gt,            
            'lbn_4_constraint_gt': self.args.lbn_4_constraint_gt,            
            'lbn_5_constraint_gt': self.args.lbn_5_constraint_gt,      
            'lbn_6_constraint_gt': self.args.lbn_6_constraint_gt,            
            'lbn_n1_constraint': self.args.lbn_n1_constraint,

            'constraint_ls': [
                'weight_constraint',
                # 'lbn_0_constraint',
                # 'lbn_n1_constraint'
            ]
        }
        if self.args.lbn_0_constraint != 1_000_000:
            feed_dict['constraint_ls'] += ['lbn_0_constraint']
        if self.args.lbn_1_constraint != 1_000_000:
            feed_dict['constraint_ls'] += ['lbn_1_constraint']
        if self.args.lbn_2_constraint != 1_000_000:
            feed_dict['constraint_ls'] += ['lbn_2_constraint']
        if self.args.lbn_3_constraint != 1_000_000:
            feed_dict['constraint_ls'] += ['lbn_3_constraint']
        if self.args.lbn_4_constraint != 1_000_000:
            feed_dict['constraint_ls'] += ['lbn_4_constraint']
        if self.args.lbn_5_constraint != 1_000_000:
            feed_dict['constraint_ls'] += ['lbn_5_constraint']
        if self.args.lbn_6_constraint != 1_000_000:
            feed_dict['constraint_ls'] += ['lbn_6_constraint']                                                            
        if self.args.lbn_n1_constraint != 1_000_000:
            feed_dict['constraint_ls'] += ['lbn_n1_constraint']

        if self.args.lbn_0_constraint_gt != -1_000_000:
            feed_dict['constraint_ls'] += ['lbn_0_constraint_gt']
        if self.args.lbn_1_constraint_gt != -1_000_000:
            feed_dict['constraint_ls'] += ['lbn_1_constraint_gt']
        if self.args.lbn_2_constraint_gt != -1_000_000:
            feed_dict['constraint_ls'] += ['lbn_2_constraint_gt']
        if self.args.lbn_3_constraint_gt != -1_000_000:
            feed_dict['constraint_ls'] += ['lbn_3_constraint_gt']            
        if self.args.lbn_4_constraint_gt != -1_000_000:
            feed_dict['constraint_ls'] += ['lbn_4_constraint_gt']
        if self.args.lbn_5_constraint_gt != -1_000_000:
            feed_dict['constraint_ls'] += ['lbn_5_constraint_gt']
        if self.args.lbn_6_constraint_gt != -1_000_000:
            feed_dict['constraint_ls'] += ['lbn_6_constraint_gt']                                        

        for col in self.df_cc_map.columns:
            if "lbn_" in col:
                feed_dict[col] = self.df_cc_map[col]

        if self.df_cc_map['node_num'].sum() < self.args.B:
            # _node_ls = []
            # for cc in df_cc_map['cc'].values:
            #     _node_ls.extend(list(cc))
            _attn, selected_cc = self.df_cc_map[edge_attr_name].sum(), list(self.df_cc_map.index)
        else:
            # print(df_cc_map.node_num.sum())
            _attn, selected_cc = self.knapsack_flex(feed_dict)

        self.df_knapsack_result = self.df_cc_map.loc[selected_cc]

        selected_nodes = []
        for cc in self.df_knapsack_result['cc']:
            selected_nodes += list(cc)

        # Laurence 20230314 >>>
        if  len(selected_nodes) <= self.args.B:
            self.add_residual_nodes(sorted_edges, selected_nodes, self.args)
        # Laurence 20230314 <<<

        retain_attn = sum(nx.get_edge_attributes(self.graph_loader.g_tr.subgraph(selected_nodes), edge_attr_name).values())
        all_attn = sum(nx.get_edge_attributes(self.graph_loader.g_tr, edge_attr_name).values())
        R_retain = retain_attn / all_attn

        if return_meta:
            _meta = {"selected_nodes":selected_nodes, 'df_cc_map':self.df_cc_map}
            return R_retain, _meta
            
        else:
            return R_retain


    @DependencyController.track
    @DependencyController.alias("4.4.1")
    @experiment_track(exp_param_ls=["path"], exp_param_map={"path":"df_reduce_graph_rs_path"} )
    def set_df_reduce_graph_rs_path(self, path):
        self.df_reduce_graph_rs_path = path

    @DependencyController.track
    @DependencyController.alias("4.4.3")
    @experiment_track()
    def set_reduce_id(self, name):
        self.reduce_id = name    

    @DependencyController.require(["4.4.1","4.4.3","4.4.2","4.3.1"])    
    @DependencyController.track
    @DependencyController.alias("4.5")
    @pyip_prompt_confirm("Do you want to save the current ours-vs-other node selection solution to a file?", False, _is_last_prompt=False)
    @pyip_prompt_input("What is the suffix (e.g., 001 --> df_reduce_graph_rs_epplitic_tmp001.pkl) of the file to save?", 'suffix')
    @experiment_track(exp_param_ls=["path"], exp_param_map={"path":"df_reduce_graph_rs_path"} )
    def save_df_reduce_graph_rs(self, suffix=None, path=''):
        if suffix == None:
            suffix = self.reduce_id
        _df = pd.concat([self.df_other_reduce_graph_rs,
                        self.df_our_reduce_graph_rs], axis=0).reset_index(drop=True)
        _path = f"{path}df_reduce_graph_rs_{self.graph_loader.graph_id}_tmp{suffix}.pkl"
        print(f"Saved path: {_path}")
        _df.to_pickle(_path)
    # def save_df_reduce_graph_rs(self, suffix, path=''):
    #     _df = pd.concat([self.df_other_reduce_graph_rs,
    #                     self.df_our_reduce_graph_rs], axis=0).reset_index(drop=True)
    #     _path = f"{path}df_reduce_graph_rs_{self.graph_loader.graph_id}_tmp{suffix}.pkl"
    #     print(f"Saved path: {_path}")
    #     _df.to_pickle(_path)        


    @DependencyController.require("4.4.1")
    @DependencyController.track
    @DependencyController.alias("4.6")
    @pyip_prompt_input("What is the suffix (e.g., 001 --> df_reduce_graph_rs_epplitic_tmp001.pkl) of the file to load?", 'suffix')
    def load_df_reduce_graph_rs(self, suffix, **kwargs):
        _path = f"{self.df_reduce_graph_rs_path}df_reduce_graph_rs_{self.graph_loader.graph_id}_tmp{suffix}.pkl"
        _df = pd.read_pickle(_path)
        self.df_other_reduce_graph_rs = _df.query("ours==0")
        self.df_our_reduce_graph_rs = _df.query("ours==1")
        print(f"loaded path: {_path}")

    @DependencyController.require(["4.4.1","4.4.3"])
    @DependencyController.track
    @DependencyController.alias("4.7")
    def run_reduce_if_not_exist(self, **kwargs):
        _df_path = Path(self.df_reduce_graph_rs_path).joinpath(Path(f"df_reduce_graph_rs_{self.graph_loader.graph_id}_tmp{self.reduce_id}.pkl"))
        if not _df_path.exists():
            self.repeat_run_our_reduce_graph(**kwargs)
            self.repeat_run_other_reduce_graph(**kwargs)

    @DependencyController.require(["4.1","4.4.1","4.4.2","4.3.1","4.4.3"])
    @DependencyController.track
    @DependencyController.alias("4.8")
    def save_reduce_rs_if_not_exist(self, **kwargs):
        _df_path = Path(self.df_reduce_graph_rs_path).joinpath(Path(f"df_reduce_graph_rs_{self.graph_loader.graph_id}_tmp{self.reduce_id}.pkl"))
        # _df_path = Path(self.df_reduce_graph_rs_path).joinpath(Path(self.reduce_id))
        if not _df_path.exists():
            _df = pd.concat([self.df_other_reduce_graph_rs,
                            self.df_our_reduce_graph_rs], axis=0).reset_index(drop=True)
            
            _df.attrs['args'] = self.selected_arg_str
            print(f"Saved path: {_df_path}")
            _df.to_pickle(_df_path)

    @DependencyController.require(["4.4.1","4.4.3"])
    @DependencyController.track
    @DependencyController.alias("4.8.1")
    def check_if_reduce_rs_exist(self):
        _df_path = Path(self.df_reduce_graph_rs_path).joinpath(Path(f"df_reduce_graph_rs_{self.graph_loader.graph_id}_tmp{self.reduce_id}.pkl"))
        return _df_path.exists()



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, last_softmax=True):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.last_softmax = last_softmax

    def forward(self, x: Tensor, edge_index: Tensor, return_feat=False) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        feat = x
        x = F.dropout(x, p=self.dropout, training=self.training) # Laurence 20230404 # ⭐
        x = self.conv2(x, edge_index)
        
        if self.last_softmax:
            x = F.softmax(x, dim=-1) # Laurence 20230404 # ⭐
        if return_feat:
            return x, feat
        return x

# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
#         self.conv1 = GCNConv(hidden_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)
#         # torch.nn.init.kaiming_normal_(self.lin1.weight)
#         # torch.nn.init.kaiming_normal_(self.conv1.lin.weight)
#         # torch.nn.init.kaiming_normal_(self.conv2.lin.weight)


#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         # x: Node feature matrix of shape [num_nodes, in_channels]
#         # edge_index: Graph connectivity matrix of shape [2, num_edges]
#         x =  self.lin1(x).relu()
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index)
#         return x
    

@class_register
class GCN_Trainer(DependencyController):
    def __init__(self, graph_loader, graph_sample_caller, run_name, model_perf_rs_path=None, 
                 best_metric_type = 'acc', **kwargs):
        super().__init__(**kwargs)
        self.graph_loader = graph_loader
        self.graph_sample_caller = graph_sample_caller
        self.unkonwn_label = -1
        self.run_name = run_name
        self.df_model_multi_run_rs_existed = None
        self.best_metric_type = best_metric_type
        self.save_learned_feat = self.graph_sample_caller.args.save_learned_feat
        if model_perf_rs_path:
            self.set_model_perf_rs_path(model_perf_rs_path)



    @DependencyController.track
    @DependencyController.alias("5.1")
    def set_model_config(self):
        self.num_classes = len(set(self.graph_loader.df_g_tr_te_data['_label'].unique()) - {self.unkonwn_label})
        self.feat_dim = len(self.graph_loader.df_g_tr_te_data.iloc[0]['feat'])
        subg_te = self.graph_loader.g_te
        subg_te_relabel = nx.relabel_nodes(subg_te, dict(zip(np.array(subg_te.nodes), np.arange(len(subg_te.nodes) )))  )
        self.subg_te = subg_te
        _, self.data_feed_gcn_test = self.create_data_to_train_gcn(subg_te_relabel, ratio_tr=0.0)
        self.is_test_node_real = (self.data_feed_gcn_test['real'] == 1)
        if self.num_classes == 2:
            self.metric_acc_te = BinaryAccuracy()
            self.f1_score = BinaryF1Score(average='none')
            self.recall = BinaryRecall().to(device)
            self.precision = BinaryPrecision().to(device)
        else:
            self.metric_acc_te = Accuracy(task='multiclass', num_classes=self.num_classes)
            self.f1_score = F1Score(task="multiclass", num_classes=self.num_classes)
        self.metric_acc_te = self.metric_acc_te.to(device)
        self.f1_score = self.f1_score.to(device)

    @DependencyController.require("5.1")
    @DependencyController.track
    @DependencyController.alias("5.2")
    @experiment_track(exp_param_ls=["args"], exp_param_map={"args":"gcn_config"})
    def set_args(self, args=[], **kwargs):
        parser = argparse.ArgumentParser(description = 'Experiment GCN Trainingw')
        parser.add_argument('--early_stop', type=int, help='Early stop epochs.', default=20)
        parser.add_argument('--epoch', type=int, help='Number of epochs.', default=200)
        parser.add_argument('--hidden_dim', type=int, help='Hidden dimention of the GCN model.', default=16)
        parser.add_argument('--use_CE_weights', type=int, help='Whether to use crossentropy weights', default=1)
        parser.add_argument('--crossentropy_weigths', nargs='+', type=int, help='Classwise weights for the CE loss.', default=[0.7, 0.3])
        parser.add_argument('--lr', type=float, help='learning rate.', default=0.001)
        parser.add_argument('--dropout', type=float, help='GCN dropout rate.', default=0.5)
        parser.add_argument('--last_softmax', type=int, help='Whether to use softmax at last', default=1)
        args = parser.parse_args(args)
        self.hidden_dim = args.hidden_dim
        self.early_stop = args.early_stop
        self.epoch = args.epoch
        self.use_ce_weights = args.use_CE_weights
        self.crossentropy_weigths = args.crossentropy_weigths
        self.lr = args.lr
        self.dropout= args.dropout
        self.last_softmax = args.last_softmax
        self.args = args
        self.set_loss()

    
    def set_loss(self):
        if self.use_ce_weights:
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.crossentropy_weigths).to(device)) # Laurence 20230403 # ⭐
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def create_data_to_train_gcn(self, g_feed0, ratio_tr = 0.8):
        g_feed = g_feed0
        data_ours = {
            'x': Tensor(np.array(list(nx.get_node_attributes(g_feed, 'feat').values()))).to(device),
            'y': Tensor(np.array(list(nx.get_node_attributes(g_feed, '_label').values()))).long().to(device),
            # 'edge_index': to_undirected(Tensor(np.array(g_feed.edges).T).long().to(device)), # Laurence 20230404 # ⭐
            'edge_index': Tensor(np.array(g_feed.edges).T).long().to(device), # Laurence 20230404 # ⭐
            'real': Tensor(np.array(list(nx.get_node_attributes(g_feed, 'real').values()))).long().to(device),
            'test': Tensor(np.array(list(nx.get_node_attributes(g_feed, 'test').values()))).long().to(device),
        }
        # Laurence 20230406 >>>
        if data_ours['edge_index'].shape[0] > 0:
            data_ours['edge_index'] = to_undirected(data_ours['edge_index'])
        # Laurence 20230406 <<<

        data_ours['train_mask'] = [True] * int(len(data_ours['x']) * ratio_tr) + [False] * (len(data_ours['x']) - int(len(data_ours['x']) *ratio_tr))
        random.shuffle(data_ours['train_mask'])
        data_ours['train_mask'] = Tensor(data_ours['train_mask']).bool()
        return g_feed, data_ours

    def train_one_time(self, df_cmp, init_random_model_every, model_dummy):
        # times_to_run = pyip.inputNum("How many models you want to train?", min=1, max=100)
        # model_0 = GCN(self.feat_dim, self.hidden_dim, self.num_classes) # dummy model used to init others

        model_0 = model_dummy
        df_cmp_ori = df_cmp.copy()
        self.te_acc_ls = []
        self.te_f1_score_ls = []
        self.losses_ls = []
        self.losses_te_ls = []
        self.best_epoch_ls = []
        self.feat_te_ls = []
        self.target_te_ls = []

        # Set testing CCs info >>>
        self.cc_te_ls = list(filter(lambda x: len(x) > 1, nx.connected_components(self.subg_te))) # Laurence 20230406
        # g_cc_te_dict = {}
        self.data_feed_g_cc_te_dict = {}
        for ind, cc in enumerate(self.cc_te_ls):
            assert len(cc) > 1
            g_cc = self.subg_te.subgraph(cc)
            g_cc = nx.relabel_nodes(g_cc, dict(zip(np.array(g_cc.nodes), np.arange(len(g_cc.nodes) )))  )
            # g_cc_te_dict[ind] = g_cc
            _, data_feed_g_cc = self.create_data_to_train_gcn(g_cc, ratio_tr=0.0)
            self.data_feed_g_cc_te_dict[ind] = data_feed_g_cc
        self.is_test_node_real = (self.data_feed_gcn_test['real'] == 1)
        # Set testing CCs info <<<

        for i in range(df_cmp.shape[0]):
            # if i % init_random_model_every == 0:
            #     model_0 = GCN(self.feat_dim , self.hidden_dim, self.num_classes) # dummy model used to init others

            _g_selected = self.graph_loader.g_tr.subgraph(df_cmp.iloc[i]['selected_nodes'])
            self._g_selected = nx.relabel_nodes(_g_selected, dict(zip(list(_g_selected.nodes), range(len(_g_selected.nodes)) )))
            
            _, self.data_feed_gcn_subg = self.create_data_to_train_gcn(self._g_selected, ratio_tr=1.0)
            self._g_selected_node_supervised_ind = [True if self._g_selected.nodes[n]['real'] else False for n in self._g_selected.nodes ]
            self._g_selected_node_supervised_ind = np.array(self._g_selected_node_supervised_ind) & np.array(self.data_feed_gcn_subg['train_mask'].numpy().tolist())
            self._g_selected_node_supervised_ind = self._g_selected_node_supervised_ind.tolist()

            model = GCN(self.feat_dim , self.hidden_dim, self.num_classes, self.dropout, self.last_softmax).to(device)
            self.model = model
            model.load_state_dict(model_0.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            # Set training CCs info >>>
            self.cc_ls = list(filter(lambda x: len(x) > 1, nx.connected_components(self._g_selected)))
            g_cc_dict = {}
            self.data_feed_g_cc_dict = {}
            for ind, cc in enumerate(self.cc_ls):
                assert len(cc) > 1
                g_cc = self._g_selected.subgraph(cc)
                g_cc = nx.relabel_nodes(g_cc, dict(zip(list(g_cc.nodes), range(len(g_cc.nodes)) )))
                g_cc_dict[ind] = g_cc
                _, data_feed_g_cc = self.create_data_to_train_gcn(g_cc, ratio_tr=1.0)
                self.data_feed_g_cc_dict[ind] = data_feed_g_cc
            # Set training CCs info <<<

            def get_te_res(return_feat=False):
                pred_te_one_ls = []
                target_te_one_ls = []
                feat_te_one_ls = []
                for ind, cc in enumerate(self.cc_te_ls):
                    _valid = self.data_feed_g_cc_te_dict[ind]['real'] == 1
                    pred_te_one, feat_te_one = model(self.data_feed_g_cc_te_dict[ind]['x'], self.data_feed_g_cc_te_dict[ind]['edge_index'], return_feat=True )
                    pred_te_one = pred_te_one[_valid]
                    feat_te_one = feat_te_one[_valid]
                    target_te_one = self.data_feed_g_cc_te_dict[ind]['y'][_valid]
                    pred_te_one_ls.append(pred_te_one)
                    target_te_one_ls.append(target_te_one)
                    feat_te_one_ls.append(feat_te_one)
                
                pred_te = torch.cat(pred_te_one_ls, dim=0)
                target_te = torch.cat(target_te_one_ls, dim=0)
                feat_te = torch.cat(feat_te_one_ls, dim=0)
                if return_feat:
                    return pred_te, target_te, feat_te

                return pred_te, target_te

            
            best_val = 0
            best_model_weight_dict = None
            self.loss_ls = []
            early_stop_count = 0
            best_epoch = 0
            for epoch in range(self.epoch):
                model.train() # Laurence 20230404 ⭐
                for ind, _cc in enumerate(self.cc_ls):
                    optimizer.zero_grad()
                    _valid = self.data_feed_g_cc_dict[ind]['real'] == 1
                    pred_tr_filtered = model(self.data_feed_g_cc_dict[ind]['x'], self.data_feed_g_cc_dict[ind]['edge_index'])[_valid]
                    target_tr_filtered = self.data_feed_g_cc_dict[ind]['y'][_valid]
                    loss = self.criterion(pred_tr_filtered, target_tr_filtered)
                    loss.backward()
                    optimizer.step()
                self.loss_ls.append(loss.detach().cpu().numpy().item())
                model.eval() # Laurence 20230404 ⭐
                pred_te, target_te = get_te_res()

                

                # Laurence 20230228 >>>
                pred_val_label_te = torch.argmax(pred_te, dim=1)
                # metric_acc_te = BinaryAccuracy()
                te_acc = self.metric_acc_te(pred_val_label_te, target_te)
                te_f1 = self.f1_score(pred_val_label_te, target_te)
                if self.best_metric_type == 'acc':
                    _new_metric_val = te_acc
                elif self.best_metric_type == 'f1':
                    _new_metric_val = te_f1
                else:
                    raise ValueError(f"Not support {self.best_metric_type}")
                
                # if epoch % 50 == 0:
                #     if self.num_classes == 2:
                #         te_precision = self.precision(pred_val_label_te, target_te)
                #         te_recall = self.recall(pred_val_label_te, target_te)
                #         tr_f1 = self.f1_score(
                #             torch.argmax(pred_tr_filtered.detach(), dim=1), 
                #             target_tr_filtered)
                    
                #         print(f"Tr-f1: {tr_f1}, Te-best-metric[f1,precision,recall]: {_new_metric_val},{te_precision},{te_recall}")

                # Laurence 20230228 <<<

                if best_val < _new_metric_val:
                    best_epoch = epoch
                    early_stop_count = 0
                    best_val = _new_metric_val
                    best_acc = te_acc
                    best_f1 = te_f1
                    best_model_weight_dict = copy.deepcopy(model.state_dict())
                else:
                    early_stop_count += 1
                    if early_stop_count > self.early_stop:
                        break

            print(f"Found best_{self.best_metric_type}: {best_val} with epochs: {epoch}")
            
            model.load_state_dict(best_model_weight_dict)

            pred_te, target_te, feat_te = get_te_res(return_feat=True)
            pred_val_label_te = torch.argmax(pred_te, dim=1)
            te_loss = F.cross_entropy(pred_te, target_te)

            te_acc = self.metric_acc_te(pred_val_label_te, target_te)
            te_f1 = self.f1_score(pred_val_label_te, target_te)
            # print(f"{df_cmp.iloc[i]['name']} get better val_acc: {te_acc:.3f}")
            self.te_acc_ls.append(te_acc.cpu().numpy().item())
            self.te_f1_score_ls.append(te_f1.cpu().numpy().item())
            self.losses_ls.append(self.loss_ls)
            self.losses_te_ls.append(te_loss.cpu().item())  
            self.best_epoch_ls.append(best_epoch)
            self.feat_te_ls.append(feat_te.tolist())
            self.target_te_ls.append(target_te.tolist())

        self.df_cmp = df_cmp_ori.assign(te_acc  = self.te_acc_ls)
        self.df_cmp = self.df_cmp.assign(loss_te = self.losses_te_ls)
        self.df_cmp = self.df_cmp.assign(f1_te = self.te_f1_score_ls)
        self.df_cmp = self.df_cmp.assign(best_epoch = self.best_epoch_ls)
        if self.save_learned_feat:
            self.df_cmp = self.df_cmp.assign(feat_te = self.feat_te_ls)
            self.df_cmp = self.df_cmp.assign(target_te = self.target_te_ls)
        return pd.DataFrame(self.df_cmp)
    
    

    @DependencyController.track
    @DependencyController.alias("5.2.1")
    @experiment_track(exp_param_ls=["path"], exp_param_map={"path":"df_model_perf_rs_path"} )
    def set_model_perf_rs_path(self, path):
        self.df_model_performance_rs_path = f"{path}/df_model_perf_rs_{self.run_name}.pkl"
        self.df_model_feat_rs_path = f"{path}/df_model_feat_rs_{self.run_name}.pkl"

    
    @DependencyController.require("5.2")
    @DependencyController.track
    @DependencyController.alias("5.3")
    @pyip_prompt_input_num("How many models you want to train?", "times_to_run")
    @experiment_track(exp_param_ls=["times_to_run"], 
                      exp_param_map={"times_to_run":"runs_per_selection"})
    def prompt_multi_runs(self, times_to_run, use_ours=True, use_others=True, **kwargs):
        df_multi_runs = []
        # times_to_run = pyip.inputNum("How many models you want to train?", min=1, max=100)
        for t in tqdm.tqdm(range(times_to_run)):
            torch.manual_seed(t * 100)
            model_0 = GCN(self.feat_dim , self.hidden_dim, self.num_classes) # dummy model used to init others
            if use_others:
                _df = self.graph_sample_caller.df_other_reduce_graph_rs
                _every = _df['name'].unique().shape[0]
                df_other = self.train_one_time(_df, _every, model_0)
                df_multi_runs.append(df_other)

            if use_ours:
                _df = self.graph_sample_caller.df_our_reduce_graph_rs
                _every = _df['name'].unique().shape[0]
                df_our = self.train_one_time(_df, _every, model_0)            
                df_multi_runs.append(df_our)

        df_multi_runs = pd.concat(df_multi_runs).reset_index(drop=True)
        df_multi_runs.attrs['times_to_run'] = times_to_run
        self.set_df_model_multi_runs_rs(df_multi_runs)
        

    @DependencyController.track
    @DependencyController.alias("5.3.1")
    def set_df_model_multi_runs_rs(self, df):
        self.df_model_multi_runs_rs = df

    # @DependencyController.require("5.2.1")
    # @DependencyController.track
    # @DependencyController.alias("5.3.1")
    # def load_df_model_multi_runs_rs_by_suffix(self, sfx):
    #     pass

    @DependencyController.require(["5.2.1"])
    @DependencyController.track
    @DependencyController.alias("5.3.2")
    def prompt_multi_runs_if_not_exist(self, *args, **kwargs):
        _df_path = Path(self.df_model_performance_rs_path)
        _df_feat_path = Path(self.df_model_feat_rs_path)
        if _df_path.exists():
            self.df_model_multi_run_rs_existed = True
            self.set_df_model_multi_runs_rs(pd.read_pickle(_df_path))
        else:
            self.df_model_multi_run_rs_existed = False
            self.prompt_multi_runs(*args, **kwargs)

    @DependencyController.require(["5.3.1"])
    @DependencyController.track
    @DependencyController.alias("5.3.3")
    def save_multi_runs_rs_if_not_exist(self):
        if self.df_model_multi_run_rs_existed == False:
            _df_path = Path(self.df_model_performance_rs_path)
            assert not _df_path.exists()
            self.df_model_multi_runs_rs.to_pickle(_df_path)



    

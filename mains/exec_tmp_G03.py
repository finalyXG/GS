# %%
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
# from importlib import reload
import torch
from utils import exp_lib
from IPython.display import clear_output
import numpy as np
import pandas as pd
from utils import common
# from common import *
from pathlib import Path
import shutil
from utils.utils import get_args
from utils.config import process_config
from utils.dirs import create_dirs



if __name__ == '__main__':    
    # capture the config path from the run arguments
    # then process the json configuration file
    args = get_args()
    config = process_config(args.config)
    print(config)
    create_dirs([config.summary_dir, config.checkpoint_dir])
    
    attn_file = config["attn_file"]

    gcn_config = ["--early_stop","20","--hidden_dim","3", "--epoch", "100", "--use_CE_weights", "0", "--last_softmax", "0", "--lr", "0.01"]

    exp_args_ls = [
        # ("--B 500 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001"),
        # ("--B 400 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "002"),
        # ("--B 300 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "003"),
        # ("--B 200 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "004"),
        ("--B 1000 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_c16"),
        ("--B 900 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_c15"),
        ("--B 800 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_c14"),
        ("--B 700 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_c13"),
        ("--B 600 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_c12"),
        ("--B 500 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_c11"),
        ("--B 400 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_c10"),
        ("--B 300 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_c9"),
        ("--B 200 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_c8"),

        ("--B 190 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b17"),
        ("--B 180 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b16"),
        ("--B 170 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b15"),
        ("--B 160 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b14"),
        ("--B 150 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b13"),
        ("--B 140 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b12"),
        ("--B 130 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b11"),
        ("--B 120 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b10"),
        ("--B 110 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b9"),
        ("--B 100 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b8"),

        ("--B 60 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b4_vis"),
        ("--B 50 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b3_vis"),  
        ("--B 40 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b2_vis"),
        ("--B 30 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b1_vis"),  
        
        
        # ("--B 90 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b7"),
        # ("--B 80 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b6"),
        # ("--B 70 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b5"),
        # ("--B 60 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b4"),
        # ("--B 50 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "001_b3"),
        # ("--B 40 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "002_b2"),
        # ("--B 30 --lbn_0_constraint_gt 10 --lbn_1_constraint_gt 10 --stop_cut_group_size 10 --unlabel_ratio_max 10 ", "003_b1"),

    ]
    exp_arg_dict = {f"sfx_{suffix}":arg for arg, suffix in exp_args_ls}
    suffix_ls = ['001_b4_vis', '001_b3_vis', '001_b2_vis', '001_b1_vis']
    
        
    global_param_dict = {
        'df_model_perf_rs_path':f"./experiments/{config['exp_name']}/",
        'graph_name': config['graph_name'],
        'selected_attn_path': config["attn_file"],
        'repeat_ours_exp_num':1, 'repeat_others_exp_num':5, 'runs_per_selection':5,
        'graph_parent_path':"./dataset",
        "attn_parent_path": "GRPG_007_tr_te_attn_ind001.npy",
        "df_gs_path": config["df_gs_path"],
        "version": 2,
        'df_reduce_graph_rs_path': f"./experiments/{config['exp_name']}/",
        'args': gcn_config,
        'model_perf_rs_path':  f"./experiments/{config['exp_name']}/",
        "times_to_run":5,
    }


    for sfx in suffix_ls:
        print(sfx, end=',')
        graph_loader = exp_lib.GraphLoader(**global_param_dict)
        
        graphsage_caller = exp_lib.GraphSAGE_Caller(graph_loader, **global_param_dict)
        graph_sample_caller = exp_lib.GraphSample_Caller(
                graph_loader, graphsage_caller, reduce_id=sfx, **global_param_dict )
        
        graph_loader.load_graph(_disable_prompt=True, **global_param_dict)
        graphsage_caller.load_attn(_disable_prompt=True, **global_param_dict) 
            
        graph_sample_caller.set_args(exp_arg_dict[f"sfx_{sfx}"], _disable_prompt=True, **global_param_dict) #⭐
        
        graph_sample_caller.run_reduce_if_not_exist(_disable_prompt=True, **global_param_dict)
        
        if not graph_sample_caller.check_if_reduce_rs_exist():
            graph_sample_caller.save_reduce_rs_if_not_exist(_disable_prompt=True, **global_param_dict)
            
            
            
    for sfx in suffix_ls:
        graph_loader = exp_lib.GraphLoader(**global_param_dict)
        graphsage_caller = exp_lib.GraphSAGE_Caller(graph_loader, **global_param_dict)
        graph_sample_caller = exp_lib.GraphSample_Caller(
                graph_loader, graphsage_caller, reduce_id=sfx, **global_param_dict )
        graph_loader.load_graph(_disable_prompt=True, **global_param_dict)
        graph_sample_caller.set_args(exp_arg_dict[f"sfx_{sfx}"], _disable_prompt=True, **global_param_dict) #⭐
        
        gcn_trainer = exp_lib.GCN_Trainer(graph_loader, graph_sample_caller, run_name=sfx, **global_param_dict)
        
        graph_sample_caller.load_df_reduce_graph_rs(sfx, _disable_prompt=True)
        gcn_trainer.set_args(**global_param_dict)
                
        gcn_trainer.prompt_multi_runs_if_not_exist(use_ours=True, use_others=True, _disable_prompt=True, **global_param_dict)
        gcn_trainer.save_multi_runs_rs_if_not_exist()
        
        break
        # print(777/0)


import argparse
from datetime import datetime
import glob
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel(logging.ERROR)

PROJ_PATH = '/mnt/c/Users/Suvi Liu/Documents/project/NWA_AI'
if PROJ_PATH not in sys.path:
    sys.path.append(PROJ_PATH)

from graphsage.supervised_train import load_data, train, FLAGS
from eval_scripts import check_node_feat, post_graphsage_trained
from graphsage.minibatch import NodeMinibatchIterator
from networkx.readwrite import json_graph

log = logging.getLogger('nwa.final.analysis.com')
# sys.stdout = StreamToLogger(log,logging.INFO)
# sys.stderr = StreamToLogger(log,logging.ERROR)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

def lassert(args):
    assert args
    return ''


exp_config = {
    'global': {
        # 'train_prefix': "../example_data/data_2022-06-29_2322",
        # 'train_prefix': "../example_data/data_2022-06-01_0956",
        # 'train_prefix': "../example_data/livi-test-data-20220511",
        'gs_train_prefix': "../example_data/aaa",
        # 'gs_train_prefix': "../example_data/data_2022-06-01_0956",
        # 'gs_train_prefix': "../example_data/livi-test-data-20220511",
        # 'model_path': "/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/sup-example_data/graphsage_mean_small_0.0010_22-06-2022-22:16:31/weights.049-sb@0-77f1_mic-0.922-f1_mac-0.922",
        # 'model_path': "/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/sup-example_data/graphsage_mean_small_0.0010_20-05-2022-10:34:09/weights.003-sb@0-22f1_mic-0.969-f1_mac-0.706",
        # 'model_path': "/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/sup-example_data/graphsage_mean_small_0.0010_30-06-2022-01:36:55/weights.120-sb@0-56f1_mic-0.949-f1_mac-0.949",
        # 'model_path': "/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/sup-example_data/graphsage_mean_small_0.0010_05-07-2022-23:42:12/weights.086-sb@0-57f1_mic-0.000-f1_mac-0.000",
        'model_path':'/mnt/c/Users/Suvi Liu/Documents/project/NWA_AI/sup-example_data/graphsage_attn_small_0.0010_13-07-2022-10:14:53/weights.122-sb@0-58f1_mic-0.000-f1_mac-0.000',
        'PROJ_PATH' : PROJ_PATH,
        'yyyymmdd_HHMM': datetime.today().strftime('%Y%m%d_%H%M'),
        'NB_ITER': 3,
        'set_FLAGS': lambda : (
            FLAGS.mark_as_parsed(),
            setattr(FLAGS, 'train_prefix', exp_config['global']['gs_train_prefix']),
            setattr(FLAGS, 'sigmoid', False),
            setattr(FLAGS, 'train_return_model_once_ready', True),
        ),
        'set_minibatch_it': lambda : (
            train_data := load_data(FLAGS.train_prefix, remove_isolated_nodes=FLAGS.remove_isolated_nodes),
            exp_config['global'].setdefault('minibatch_it', post_graphsage_trained.get_tr_iter(train_data, FLAGS, NodeMinibatchIterator)),
            exp_config['global'].setdefault('train_data', train_data),
        ),
        'set_model': lambda : (
            exp_config['global'].setdefault('model', train(exp_config['global']['train_data'])), 
        ),
        'load_model_weight': lambda : (
            model := exp_config['global']['model'], 
            model.load_weights(exp_config['global']['model_path'])
        ),
        'set_df_G_data': lambda : (
            minibatch_it := exp_config['global']['minibatch_it'],
            G_data := json.load(open(FLAGS.train_prefix + "-G.json")),
            # Note that train/val/testing nodes are subset of "can_graph" nodes, as some can_graph nodes containing 95% fake nodes and no other real nodes, 
            # which are correct but will be remove from trai/val/testing nodes. 
            # can_graph_dict := {n: 1 for n in minibatch_it.train_nodes + minibatch_it.val_nodes + minibatch_it.test_nodes},
            can_graph_dict := {n: 1 for n in minibatch_it.G.nodes if (minibatch_it.G.nodes[n]['can_graph']==1)and(minibatch_it.G.nodes[n]['real']==True)},
            df_G_data := pd.DataFrame(),
            df_G_data := df_G_data.assign(
                id=[n['id'] for n in G_data['nodes'] if n.setdefault('real', False)],
                feat=[n['feat'] for n in G_data['nodes'] if n.setdefault('real', False)],
                label=[n['label'][1] for n in G_data['nodes'] if n.setdefault('real', False)],
                val = [n['val'] for n in G_data['nodes'] if n.setdefault('real', False)],
                test = [n['test'] for n in G_data['nodes'] if n.setdefault('real', False)],
                can_graph = [can_graph_dict.setdefault(n['id'], 0) for n in G_data['nodes'] if n.setdefault('real', False) ],
            ),
            # Laurence 20220706
            df_G_data := df_G_data.assign(
                node_feat_check_sum=lambda x: [sum(e) for e in x['feat']]
            ),
            exp_config['global'].setdefault('df_G_data', df_G_data)
        ),
        'set_df_gs': lambda: (
            model := exp_config['global']['model'],
            df_gs := post_graphsage_trained.get_df_gs_info(exp_config['global']['train_data'], FLAGS, NodeMinibatchIterator, model, nb_iter=exp_config['global']['NB_ITER'] ),
            exp_config['global'].setdefault('df_gs_info', df_gs),
        ),
        'set_df_combine_info': lambda : (
            graph_feat_agg_name := f"graph_feat_agg_{exp_config['global']['NB_ITER']}",
            graph_pred_raw_name := f"graph_pred_raw_{exp_config['global']['NB_ITER']}",
            df_combine_info := pd.merge(
                left=exp_config['global']['df_G_data'],
                right=exp_config['global']['df_gs_info'][['id','graph_feat','graph_pred','node_feat_check_sum',graph_feat_agg_name,graph_pred_raw_name]],
                on=['id','node_feat_check_sum'], how='left'),

            # df_combine_info := pd.merge(
            #     left=exp_config['global']['df_G_data'],
            #     right=exp_config['global']['df_gs_info'][['id','graph_feat','graph_pred',graph_feat_agg_name,graph_pred_raw_name]],
            #     on='id', how='left'),


            # Laurence 20220706
            df_combine_info := df_combine_info.assign(can_gs=lambda x: ~x['graph_feat'].isna()),
            lassert(df_combine_info['can_graph'].isna().sum() == 0),
            # exp_config['global'].setdefault('df_combine_info', df_combine_info),
            exp_config['global'].update({'df_combine_info': df_combine_info}),
        ),
        'pad_zero_df_combine_info': lambda : (
            graph_feat_agg_name := f"graph_feat_agg_{exp_config['global']['NB_ITER']}",
            df_combine_info := exp_config['global']['df_combine_info'],
            # pad_zero_idx := df_combine_info.query('can_graph==0').index,
            pad_zero_idx := df_combine_info.query('can_gs==False').index,
            pad_zero_idx_to_assert := df_combine_info.pipe(lambda x: x[x['graph_feat'].isna()]).index,
            pad_zero_ind_dict := {v:True for v in pad_zero_idx.values},
            # Laurence 20220706
            # graph_feat_dim := len(df_combine_info.query("can_graph==1")['graph_feat'].values[0]),
            graph_feat_dim := len(df_combine_info.query("can_gs==True")['graph_feat'].values[0]),
            zero_feat := [0.] * graph_feat_dim,
            graph_feat_agg_col_val := [e if pad_zero_ind_dict.setdefault(k, False)==False else zero_feat for k,e in df_combine_info[graph_feat_agg_name].items() ],
            graph_feat_col_val := [e if pad_zero_ind_dict.setdefault(k, False)==False else zero_feat for k,e in df_combine_info['graph_feat'].items() ],
            df_combine_info := df_combine_info.assign(**{graph_feat_agg_name: graph_feat_agg_col_val}),
            df_combine_info := df_combine_info.assign(graph_feat=graph_feat_col_val),
            exp_config['global'].update({'df_combine_info': df_combine_info}),
        ),
        'assert_df': lambda : (
            graph_feat_agg_name := f"graph_feat_agg_{exp_config['global']['NB_ITER']}",
            df_G_data := exp_config['global']['df_G_data'],
            df_gs_info := exp_config['global']['df_gs_info'],
            df_combine_info := exp_config['global']['df_combine_info'],
            # Laurence 20220706
            # lassert(df_G_data.query("(can_graph==1)").shape[0] == df_gs_info.query("is_graph==True").shape[0]),
            lassert(df_G_data.query("(can_graph==1)").shape[0] >= df_gs_info.query("is_graph==True").shape[0]),
            lassert(df_combine_info['graph_feat'].isna().sum() == 0), # Since we apply "pad_zero_df_combine_info"
            lassert(df_combine_info[graph_feat_agg_name].isna().sum() == 0),
        ),
        'set_node_feat_append_graph_feat_info': lambda : (
            df_combine_info := exp_config['global']['df_combine_info'],
            graph_feat_agg_name := f"graph_feat_agg_{exp_config['global']['NB_ITER']}",
            df_combine_info := df_combine_info.assign(feat_append_graph_feat = lambda x: x.pipe(lambda x: np.concatenate(
                [np.stack(x['feat'].values),
                np.stack(x['graph_feat'].values)], axis=-1).tolist())),
            df_combine_info := df_combine_info.assign(feat_append_graph_feat_agg = lambda x: x.pipe(lambda x: np.concatenate(
                [np.stack(x['feat'].values),
                np.stack(x[graph_feat_agg_name].values)], axis=-1).tolist())),
            exp_config['global'].update({'df_combine_info': df_combine_info}),
            
        ),
        'set_col_feat_LR_pred': lambda : (
            df_combine_info := exp_config['global']['df_combine_info'],
            logreg := check_node_feat.fit_lr(
                df_combine_info.query("test==False")['feat'].values.tolist(),
                df_combine_info.query("test==False")['label'].values.tolist(),
                **{'max_iter': 1000},
            ),
            df_combine_info := df_combine_info.assign(feat_LR_pred=logreg.predict_proba(df_combine_info['feat'].values.tolist()).tolist()),
            exp_config['global'].update({'df_combine_info': df_combine_info}),
        ),
        'set_col_FAGFA_pred': lambda : (
            df_combine_info := exp_config['global']['df_combine_info'],
            logreg_final := check_node_feat.fit_lr(
                df_combine_info.query("test==False")['feat_append_graph_feat_agg'].values.tolist(),
                df_combine_info.query("test==False")['label'].values.tolist(),
                **{'max_iter': 1000},
            ),
            df_combine_info := df_combine_info.assign(FAGFA_pred=logreg_final.predict_proba(df_combine_info['feat_append_graph_feat_agg'].values.tolist()).tolist()),
            exp_config['global'].update({'df_combine_info': df_combine_info}),
        ),
        'compute_gs_rdr_info': lambda : (
            df_combine_info := exp_config['global']['df_combine_info'],
            # Laurence 20220706
            # ms := df_combine_info.query("(can_graph==1)&(test==True)&(label==1.0)").pipe(lambda x: np.stack(x['graph_pred'])[:,1] ).min(),
            ms := df_combine_info.query("(can_gs==1)&(test==True)&(label==1.0)").pipe(lambda x: np.stack(x['graph_pred'])[:,1] ).min(),
            log.info(f'minimum "can graph" true-hit score from GS: {ms}'),
            rdr_nb := df_combine_info.query("(can_graph==1)&(test==True)").pipe(lambda x: np.stack(x['graph_pred'])[:,1] < ms).sum(),
            log.info(f'reduction number on "can graph" testing data: {rdr_nb}'),
            ms1 := df_combine_info.query("(test==True)&(label==1.0)").pipe(lambda x: np.stack(x['FAGFA_pred'])[:,1] ).min(),
            log.info(f'minimum true-hit score from GS-FAGFA_pred: {ms1}'),
            rdr_nb1 := df_combine_info.query("(test==True)").pipe(lambda x: np.stack(x['FAGFA_pred'])[:,1] < ms).sum(),
            log.info(f'reduction number on all testing data (< {ms1}): {rdr_nb1}'),
        ),
        'set_logging_info': lambda : (
            proj_path := exp_config['global']['PROJ_PATH'],
            yyyymmdd_HHMM := exp_config['global']['yyyymmdd_HHMM'],
            staging := f"{proj_path}/staging/tmp_t/",
            Path(f"{staging}{yyyymmdd_HHMM}").mkdir(parents=True, exist_ok=True),
            logging_path := f'{staging}{yyyymmdd_HHMM}/output.log',
            logging.basicConfig(filename=logging_path, filemode='a', level=logging.DEBUG),
            exp_config['global'].update({'staging_path': f'{staging}{yyyymmdd_HHMM}'}),
            hdlr := logging.StreamHandler(),
            fhdlr := logging.FileHandler(f"{logging_path}", mode='w', encoding='utf-8'),
            formatter := logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            hdlr.setFormatter(formatter),
            fhdlr.setFormatter(formatter),
            log.addHandler(hdlr),
            log.addHandler(fhdlr),
            log.setLevel(logging.DEBUG),
        ),
        'save_l2m_df': lambda : (
            df_combine_info := exp_config['global']['df_combine_info'],
            df_combine_info.to_json(f'{exp_config["global"]["staging_path"]}/l2m_graphsage.json')
        ),
        'set_exp_config_data': lambda : (
            exp_config['data'].update({'value': exp_config['global']['df_combine_info']}),
        ),

        'run_step_ls': [
            'set_FLAGS', 
            'set_minibatch_it', 
            'set_df_G_data',
            'set_model',
            'load_model_weight',
            'set_df_gs',
            'set_df_combine_info',
            'pad_zero_df_combine_info',
            'assert_df',
            'set_node_feat_append_graph_feat_info',
            'set_col_feat_LR_pred',
            'set_col_FAGFA_pred',
            'compute_gs_rdr_info',
            'set_logging_info',
            'save_l2m_df',
            'set_exp_config_data',
        ],
        # Note: the load_and_check_model_wegiths should place on top as it has "in-place" effect.
        'run_exp_ls' : [
            # 'load_and_check_model_weights',
            # 'one_LR_w_feat_on_all', 
            # 'one_LR_w_feat_on_can_graph',
            # 'one_LR_w_feat_on_can_not_graph',
            # 'one_LR_w_graph_feat_on_can_graph',
            # 'one_LR_w_all_feat_on_all',
            # 'one_LR_w_all_feat_on_can_graph',
            # 'one_LR_w_all_feat_on_can_not_graph',
            # 'one_LR_w_graph_agg_feat_on_can_graph',
            # 'one_LR_w_all_FAGFA_on_all'
        ],
        
    },
    'data':{
        'value': None,
        'type': 'data',
    },
    'one_LR_w_feat_on_all':{
        'help': 'Use one single logistic regression (LR) on all data (i.e., ignore whehter the data can form a  local graph or not). In particular, we use "feat".',
        'df': lambda : exp_config['data']['value'],
        'type': 'LR',
        'feat_tr': lambda df: df.query('test==False')['feat'].values.tolist(),
        'target_tr': lambda df: df.query('test==False')['label'].values.tolist(),
        'feat_te': lambda df: df.query('test==True')['feat'].values.tolist(),
        'target_te': lambda df: df.query('test==True')['label'].values.tolist(),
    },
    'one_LR_w_feat_on_can_graph':{
        'help': 'Use one single logistic regression (LR) on "can graph" data (i.e., can form a local graph). In particular, we use "feat"',
        'df': lambda : exp_config['data']['value'],
        'type': 'LR', 
        'feat_tr': lambda df: df.query('(test==False)&(can_graph==1)')['feat'].values.tolist(),
        'target_tr': lambda df: df.query('(test==False)&(can_graph==1)')['label'].values.tolist(),
        'feat_te': lambda df: df.query('(test==True)&(can_graph==1)')['feat'].values.tolist(),
        'target_te': lambda df: df.query('(test==True)&(can_graph==1)')['label'].values.tolist(),
    },
    'one_LR_w_feat_on_can_not_graph':{
        'help': 'Use one single logistic regression (LR) on "can not graph" data (i.e., can not form a local graph). In particular, we use "feat"',
        'df': lambda : exp_config['data']['value'],
        'type': 'LR', 
        'feat_tr': lambda df: df.query('(test==False)&(can_graph==0)')['feat'].values.tolist(),
        'target_tr': lambda df: df.query('(test==False)&(can_graph==0)')['label'].values.tolist(),
        'feat_te': lambda df: df.query('(test==True)&(can_graph==0)')['feat'].values.tolist(),
        'target_te': lambda df: df.query('(test==True)&(can_graph==0)')['label'].values.tolist(),
    },
    'load_and_check_model_weights':{
        'help':'Load model checkpoint, and then verify its validation by pre-saved data file (e.g., check_sb@0.npy. Note that the last element in sb_k will be used to load model weights. In particular, we use "feat".',
        'type': 'verify_model_weight',
        # 'model': model,
        # 'checkpoint_path': "/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/sup-example_data/graphsage_mean_small_0.0010_20-05-2022-10:34:09",
        # 'checkpoint_path': "/mnt/c/Users/Laurence_Liu/Documents/Regtics_proj/NWA_AI/sup-example_data/graphsage_mean_small_0.0010_06-07-2022-10:43:35", # Laurence 20220706
        # 'sb_k': [0], #  2, 4, 8
    },
    'one_LR_w_graph_feat_on_can_graph':{
        'help':'Use one single logistic regression (LR) on "can graph" data using graphsage feature. In particular, we use "feat".',
        'df': lambda : exp_config['data']['value'],
        'type': 'LR', 'max_iter': 10000,
        'feat_tr': lambda df: df.query('(test==False)&(can_graph==1)')['graph_feat'].values.tolist(),
        'target_tr': lambda df: df.query('(test==False)&(can_graph==1)')['label'].values.tolist(),
        'feat_te': lambda df: df.query('(test==True)&(can_graph==1)')['graph_feat'].values.tolist(),
        'target_te': lambda df: df.query('(test==True)&(can_graph==1)')['label'].values.tolist(),
    },
    'one_LR_w_all_feat_on_all':{
        'help':'Use one single logistic regression (LR) on "all" data (no matter can graph or not) using combined feature. In particular, we use "feat_append_graph_feat".',
        'df': lambda : exp_config['data']['value'],
        'type': 'LR', 'max_iter': 1000,
        'feat_tr': lambda df: df.query('(test==False)')['feat_append_graph_feat'].values.tolist(),
        'target_tr': lambda df: df.query('(test==False)')['label'].values.tolist(),
        'feat_te': lambda df: df.query('(test==True)')['feat_append_graph_feat'].values.tolist(),
        'target_te': lambda df: df.query('(test==True)')['label'].values.tolist(),
    },
    'one_LR_w_all_feat_on_can_graph':{
        'help':'Use one single logistic regression (LR) on "can graph" data using combined feature. In particular, we use "feat_append_graph_feat".',
        'df': lambda : exp_config['data']['value'],
        'type': 'LR', 'max_iter': 1000,
        'feat_tr': lambda df: df.query('(test==False)&(can_graph==1)')['feat_append_graph_feat'].values.tolist(),
        'target_tr': lambda df: df.query('(test==False)&(can_graph==1)')['label'].values.tolist(),
        'feat_te': lambda df: df.query('(test==True)&(can_graph==1)')['feat_append_graph_feat'].values.tolist(),
        'target_te': lambda df: df.query('(test==True)&(can_graph==1)')['label'].values.tolist(),
    },
    'one_LR_w_all_feat_on_can_not_graph':{
        'help':'Use one single logistic regression (LR) on "can not graph" data using combined feature. In particular, we use "feat_append_graph_feat".',
        'df': lambda : exp_config['data']['value'],
        'type': 'LR', 'max_iter': 1000,
        'feat_tr': lambda df: df.query('(test==False)&(can_graph==0)')['feat_append_graph_feat'].values.tolist(),
        'target_tr': lambda df: df.query('(test==False)&(can_graph==0)')['label'].values.tolist(),
        'feat_te': lambda df: df.query('(test==True)&(can_graph==0)')['feat_append_graph_feat'].values.tolist(),
        'target_te': lambda df: df.query('(test==True)&(can_graph==0)')['label'].values.tolist(),
    },
    'one_LR_w_graph_agg_feat_on_can_graph':{
        'help':'Use one single logistic regression (LR) on "can not graph" data using graph-aggregated feature. In particular, we use "graph_feat_agg_30".',
        'df': lambda : exp_config['data']['value'],
        'type': 'LR', 'max_iter': 1000,
        'feat_tr': lambda df: df.query('(test==False)&(can_graph==1)')[f"graph_feat_agg_{exp_config['global']['NB_ITER']}"].values.tolist(),
        'target_tr': lambda df: df.query('(test==False)&(can_graph==1)')['label'].values.tolist(),
        'feat_te': lambda df: df.query('(test==True)&(can_graph==1)')[f"graph_feat_agg_{exp_config['global']['NB_ITER']}"].values.tolist(),
        'target_te': lambda df: df.query('(test==True)&(can_graph==1)')['label'].values.tolist(),
    },
    'one_LR_w_all_FAGFA_on_all':{
        'help':'Use one single logistic regression (LR) on "all" data (no matter can graph or not) using combined feature. In particular, we use "feat_append_graph_feat_agg".',
        'df': lambda : exp_config['data']['value'],
        'type': 'LR', 'max_iter': 1000,
        'feat_tr': lambda df: df.query('(test==False)')['feat_append_graph_feat_agg'].values.tolist(),
        'target_tr': lambda df: df.query('(test==False)')['label'].values.tolist(),
        'feat_te': lambda df: df.query('(test==True)')['feat_append_graph_feat_agg'].values.tolist(),
        'target_te': lambda df: df.query('(test==True)')['label'].values.tolist(),
    },
    
    
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis Flow.')
    parser.add_argument('--save_to_folder_by_date', type=str, help='Date value yyyymmdd_HHMM as saved folder name', default=None)

    args = parser.parse_args()
    if args.save_to_folder_by_date:
        exp_config['global']['yyyymmdd_HHMM'] = args.save_to_folder_by_date

    for e in exp_config['global']['run_step_ls']:
        exp_config['global'][e]()
    
    run_exp_ls = exp_config['global']['run_exp_ls']

    for k in run_exp_ls:
        if k not in exp_config.keys():
            continue
        e = exp_config[k]
        log.info('+' * 100)
        log.info(f"Experiment name: {k}")
        log.info(f"Description: {e['help']}")

        if 'df' in e.keys():
            if e.keys() and type(e['df']).__name__ == 'function':
                df = e['df']()
            else:
                df = e['df']
            
        if e['type'] == 'LR':
            feat_tr = e['feat_tr'](df)
            feat_te = e['feat_te'](df)
            target_tr = e['target_tr'](df)
            target_te = e['target_te'](df)
            kwargs = {'max_iter': e.setdefault('max_iter', 100)}
            logreg = check_node_feat.fit_lr(feat_tr, target_tr, **kwargs)
            check_node_feat.lr_eval(logreg, feat_tr, feat_te, target_tr, target_te, log=log)
            
        # if e['type'] == 'verify_model_weight':
        #     assert 'train_data' in exp_config['global'].keys()
        #     model = train(exp_config['global']['train_data'])

        #     sb_k_ls = e['sb_k']
        #     # model = e['model']
        #     checkpoint_path = e['checkpoint_path']

        #     for sk in sb_k_ls:
        #         check_sb_data = np.load(f'{checkpoint_path}/check_sb@{sk}.npy', allow_pickle=True).item()
        #         check_sb_result = np.load(f'{checkpoint_path}/check_sb@{sk}_result.npy', allow_pickle=True)
        #         model_path = glob.glob(f"{checkpoint_path}/weights.{check_sb_result[-1]['epoch']:03d}-sb@{sk}*.index")[0].split('.index')[0]
        #         log.info(f"model_path (best with sb@{sk}): {model_path}")
        #         model.load_weights(model_path)
        #         model.check_sb_data = check_sb_data
        #         pred_rs = model.test_one_step(check_sb_data, return_node_feat=True)
        #         # assert (check_sb_result[0] == pred_rs[0].numpy()).all()
        #         log.info(f'Model (best with sb@{sk}) weights verified (passed).\n')

        log.info('+' * 100)




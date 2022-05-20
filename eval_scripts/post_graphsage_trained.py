
import argparse

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import tensorflow as tf
import tqdm


def get_tr_iter(train_data, FLAGS, NodeMinibatchIterator):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map  = train_data[4]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))
    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = {
        'labels' : tf.cast(tf.compat.v1.distributions.Bernoulli(probs=0.7).sample(sample_shape=(1, num_classes)), tf.float32),
        'batch' : tf.constant(list(G.nodes)[:1], dtype=tf.int32, name='batch1'),
        'dropout': tf.constant(0., dtype=tf.float32, name='batch1'),
        'batch_size' : tf.constant(FLAGS.batch_size, dtype=tf.float32, name='batch1'),
    }
    minibatch = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs)
    return minibatch


# Get dataframe of training and testing graphsage features, predictions and labels 
def get_df_gs_info(train_data, FLAGS, NodeMinibatchIterator, model, nb_iter=1):

    assert nb_iter >= 1
    df_tr_gs_info_ls = []
    df_val_gs_info_ls = []
    df_te_gs_info_ls = []
    
    for i in tqdm.tqdm(range(nb_iter), desc="Generating training info:"):
        minibatch_it = get_tr_iter(train_data, FLAGS, NodeMinibatchIterator)
        labels_ls = []
        pred_ls = []
        feat_ls = []
        node_id_ls = []
        while not minibatch_it.end():
            feed_dict, labels = minibatch_it.next_minibatch_feed_dict()
            outs = model.test_one_step(feed_dict, return_node_feat=True)
            pred_ls += outs[0].numpy().tolist()
            labels_ls += labels[:,1].tolist()
            feat_ls += outs[-1].numpy().tolist()
            node_id_ls += feed_dict['batch'].numpy().tolist()

        df_tr_gs_info = pd.DataFrame({
            'id':node_id_ls,
            'graph_feat':feat_ls, 
            'graph_pred':pred_ls, 
            'label':labels_ls})\
            .assign(is_train=True).sort_values(by='id')

        df_tr_gs_info_ls.append(df_tr_gs_info)
    graph_feat_agg = np.mean(np.stack([np.stack(df['graph_feat'].values) for df in df_tr_gs_info_ls]), axis=0).tolist()
    graph_pred_raw = np.stack([np.stack(df['graph_pred'].values) for df in df_tr_gs_info_ls])
    graph_pred_raw = np.transpose(graph_pred_raw, (1,0,2)).tolist()
    df_tr_gs_info = df_tr_gs_info.assign(**{f'graph_feat_agg_{nb_iter}': graph_feat_agg})
    df_tr_gs_info = df_tr_gs_info.assign(**{f'graph_pred_raw_{nb_iter}': graph_pred_raw})

    

    for i in tqdm.tqdm(range(nb_iter), desc="Generating val info:"):
        feed_dict_val, labels_val = minibatch_it.node_val_feed_dict(test=False)
        outs_val = model.test_one_step(feed_dict_val, return_node_feat=True)
        df_val_gs_info = pd.DataFrame({
            'id': feed_dict_val['batch'].numpy().tolist(),
            'graph_feat':outs_val[-1].numpy().tolist(), 
            'graph_pred':outs_val[0].numpy().tolist(),
            'label': labels_val[:,1].tolist()}).assign(is_train=False)

        df_val_gs_info_ls.append(df_val_gs_info)
    graph_feat_agg = np.mean(np.stack([np.stack(df['graph_feat'].values) for df in df_val_gs_info_ls]), axis=0).tolist()
    graph_pred_raw = np.stack([np.stack(df['graph_pred'].values) for df in df_val_gs_info_ls])
    graph_pred_raw = np.transpose(graph_pred_raw, (1,0,2)).tolist()
    df_val_gs_info = df_val_gs_info.assign(**{f'graph_feat_agg_{nb_iter}': graph_feat_agg})
    df_val_gs_info = df_val_gs_info.assign(**{f'graph_pred_raw_{nb_iter}': graph_pred_raw})

    
    for i in tqdm.tqdm(range(nb_iter), desc="Generating test info:"):
        feed_dict_te, labels_te = minibatch_it.node_val_feed_dict(test=True)
        outs_te = model.test_one_step(feed_dict_te, return_node_feat=True)
        df_te_gs_info = pd.DataFrame({
            'id': feed_dict_te['batch'].numpy().tolist(),
            'graph_feat':outs_te[-1].numpy().tolist(),
            'graph_pred':outs_te[0].numpy().tolist(),
            'label': labels_te[:,1].tolist()}).assign(is_train=False)
        df_te_gs_info_ls.append(df_te_gs_info)
    graph_feat_agg = np.mean(np.stack([np.stack(df['graph_feat'].values) for df in df_te_gs_info_ls]), axis=0).tolist()
    graph_pred_raw = np.stack([np.stack(df['graph_pred'].values) for df in df_te_gs_info_ls])
    graph_pred_raw = np.transpose(graph_pred_raw, (1,0,2)).tolist()
    df_te_gs_info = df_te_gs_info.assign(**{f'graph_feat_agg_{nb_iter}': graph_feat_agg})
    df_te_gs_info = df_te_gs_info.assign(**{f'graph_pred_raw_{nb_iter}': graph_pred_raw})



    df_gs_info = pd.concat([
        df_tr_gs_info, 
        df_val_gs_info, 
        df_te_gs_info
        ], 
        axis=0).reset_index(drop=True).assign(is_graph=True)

    return df_gs_info

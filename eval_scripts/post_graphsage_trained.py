
import argparse

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tqdm
from graphsage.supervised_train import load_data, train, FLAGS



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
def get_df_gs_info(train_data, FLAGS, NodeMinibatchIterator, model, nb_iter=1, include_test=True):

    assert nb_iter >= 1
    df_tr_gs_info_ls = []
    df_val_gs_info_ls = []
    df_te_gs_info_ls = []
    # Laurence 2022076
    minibatch_it = get_tr_iter(train_data, FLAGS, NodeMinibatchIterator)
    idx2id = {v: k for k, v in minibatch_it.id2idx.items()}

    # Laurence 20220713
    idx2id[len(idx2id)] = -1
    
    for i in tqdm.tqdm(range(nb_iter), desc="Generating training info:"):
        # Laurence 20220706
        minibatch_it = get_tr_iter(train_data, FLAGS, NodeMinibatchIterator)
        labels_ls = []
        pred_ls = []
        feat_ls = []
        node_feat_ls = []
        node_id_ls = []
        node_neigh_id_ls_dict = {}
        attn_w_ls_dict = {}
        while not minibatch_it.end():
            feed_dict, labels = minibatch_it.next_minibatch_feed_dict()
            outs = model.test_one_step(feed_dict, return_node_feat=True, return_sampled_nodes=True, return_others=True)
            pred_ls += outs[0].numpy().tolist()
            labels_ls += np.argmax(labels, axis=1).tolist()
            feat_ls += outs[3].numpy().tolist()
            node_id_ls += feed_dict['batch'].numpy().tolist()
            # Laurence 20220713
            samples_idx = outs[2][0]
            bs = tf.shape(samples_idx[0]).numpy()[0]
            acc_shape = [bs]
            acc_mul = bs
            hop = 0
            for node_idx_ls in samples_idx[1:]:
                tmp_len = len(node_idx_ls)
                acc_shape += [tmp_len // acc_mul]
                acc_mul = tmp_len
                node_idx_ls_np = [idx2id[n] for n in node_idx_ls.numpy()]
                node_id_tr = tf.reshape(node_idx_ls_np, acc_shape)
                node_neigh_id_ls_dict.setdefault(hop, [])
                node_neigh_id_ls_dict[hop] += node_id_tr.numpy().tolist()
                hop += 1

            # Laurence 20220715: add attention
            attn_w = outs[4]['attn_w']
            attn_w_reshape = {}
            bs = outs[0].shape[0]
            for k,attn_by_layer in attn_w.items():
                agg_name = f'attn_agg_{k}'
                acc_mul = 1
                front_shape = []
                attn_w_reshape[agg_name] = {}
                for hop, attn_by_hop in attn_by_layer.items():
                    hop_name = f'hop_{hop}'
                    divide = attn_by_hop.shape[0] // acc_mul
                    acc_mul *= divide
                    front_shape.append(divide)
                    new_shape = front_shape + attn_by_hop.shape[1:]
                    attn_w_reshape[agg_name][hop_name] = tf.reshape(attn_by_hop, new_shape).numpy().tolist()
                    col_name = f'{agg_name}_{hop_name}'
                    attn_w_ls_dict.setdefault(col_name, [])
                    attn_w_ls_dict[col_name] += attn_w_reshape[agg_name][hop_name] 
                    # print(front_shape, acc_mul, new_shape)


        node_feat_ls = [minibatch_it.G.nodes[n]['feat'] for n in node_id_ls]
        node_feat_cs_ls = [sum(e) for e in node_feat_ls]
        df_tr_gs_info = pd.DataFrame({
            'id': [idx2id[n] for n in node_id_ls], # Laurence 20220706
            'graph_feat':feat_ls, 
            'graph_pred':pred_ls, 
            'node_feat': node_feat_ls,
            'node_feat_check_sum': node_feat_cs_ls,
            'label':labels_ls})\
            .assign(is_train=True).sort_values(by='id')

        # Laurence 20220713
        for k, v in node_neigh_id_ls_dict.items():
            df_tr_gs_info = df_tr_gs_info.assign(**{f'neigh_{k}': v})

        # Laurence 20220715
        for k, v in attn_w_ls_dict.items():
            df_tr_gs_info = df_tr_gs_info.assign(**{k: v})



        df_tr_gs_info_ls.append(df_tr_gs_info)
    graph_feat_agg = np.mean(np.stack([np.stack(df['graph_feat'].values) for df in df_tr_gs_info_ls]), axis=0).tolist()
    graph_pred_raw = np.stack([np.stack(df['graph_pred'].values) for df in df_tr_gs_info_ls])
    graph_pred_raw = np.transpose(graph_pred_raw, (1,0,2)).tolist()
    df_tr_gs_info = df_tr_gs_info.assign(**{f'graph_feat_agg_{nb_iter}': graph_feat_agg})
    df_tr_gs_info = df_tr_gs_info.assign(**{f'graph_pred_raw_{nb_iter}': graph_pred_raw})

    
    df_val_gs_info = pd.DataFrame()
    # Laurence 20220715: Disable validation dataframe
    # if len(minibatch_it.val_nodes) > 0:
    #     for i in tqdm.tqdm(range(nb_iter), desc="Generating val info:"):
    #         feed_dict_val, labels_val = minibatch_it.node_val_feed_dict(test=False)
    #         outs_val = model.test_one_step(feed_dict_val, return_node_feat=True)
    #         df_val_gs_info = pd.DataFrame({
    #             # 'id': feed_dict_val['batch'].numpy().tolist(), # Laurence 20220706
    #             'id': [idx2id[n] for n in feed_dict_val['batch'].numpy().tolist()],
    #             'graph_feat':outs_val[-1].numpy().tolist(), 
    #             'graph_pred':outs_val[0].numpy().tolist(),
    #             'label': labels_val[:,1].tolist()}).assign(is_train=False)

    #         df_val_gs_info_ls.append(df_val_gs_info)
    #     graph_feat_agg = np.mean(np.stack([np.stack(df['graph_feat'].values) for df in df_val_gs_info_ls]), axis=0).tolist()
    #     graph_pred_raw = np.stack([np.stack(df['graph_pred'].values) for df in df_val_gs_info_ls])
    #     graph_pred_raw = np.transpose(graph_pred_raw, (1,0,2)).tolist()
    #     df_val_gs_info = df_val_gs_info.assign(**{f'graph_feat_agg_{nb_iter}': graph_feat_agg})
    #     df_val_gs_info = df_val_gs_info.assign(**{f'graph_pred_raw_{nb_iter}': graph_pred_raw})

    df_te_gs_info = pd.DataFrame()
    if include_test:
        # Set current adj matrix for testing >>>
        for l in model.layer_infos:  # Laurence 20220713
            l.neigh_sampler.adj_info = minibatch_it.test_adj
        # Set current adj matrix for testing <<<
        for i in tqdm.tqdm(range(nb_iter), desc="Generating test info:"):
            feed_dict_te, labels_te = minibatch_it.node_val_feed_dict(test=True)
            outs_te = model.test_one_step(feed_dict_te, return_node_feat=True, return_sampled_nodes=True, return_others=True)

            # Laurence 20220706
            node_id_ls = [idx2id[n] for n in feed_dict_te['batch'].numpy().tolist()]
            node_feat_ls = [minibatch_it.G.nodes[n]['feat'] for n in node_id_ls]
            node_feat_cs_ls = [sum(e) for e in node_feat_ls]

            node_neigh_id_ls_dict = {}
            # Laurence 20220713
            samples_idx = outs_te[2][0]
            bs = tf.shape(samples_idx[0]).numpy()[0]
            acc_shape = [bs]
            acc_mul = bs
            hop = 0
            for node_idx_ls in samples_idx[1:]:
                tmp_len = len(node_idx_ls)
                acc_shape += [tmp_len // acc_mul]
                acc_mul = tmp_len
                node_idx_ls_np = [idx2id[n] for n in node_idx_ls.numpy()]
                node_id_tr = tf.reshape(node_idx_ls_np, acc_shape)
                node_neigh_id_ls_dict.setdefault(hop, [])
                node_neigh_id_ls_dict[hop] += node_id_tr.numpy().tolist()
                hop += 1

            attn_w_ls_dict = {}
            # Laurence 20220715: add attention
            attn_w = outs_te[4]['attn_w']
            attn_w_reshape = {}
            bs = outs_te[0].shape[0]
            for k,attn_by_layer in attn_w.items():
                agg_name = f'attn_agg_{k}'
                acc_mul = 1
                front_shape = []
                attn_w_reshape[agg_name] = {}
                for hop, attn_by_hop in attn_by_layer.items():
                    hop_name = f'hop_{hop}'
                    divide = attn_by_hop.shape[0] // acc_mul
                    acc_mul *= divide
                    front_shape.append(divide)
                    new_shape = front_shape + attn_by_hop.shape[1:]
                    attn_w_reshape[agg_name][hop_name] = tf.reshape(attn_by_hop, new_shape).numpy().tolist()
                    col_name = f'{agg_name}_{hop_name}'
                    attn_w_ls_dict.setdefault(col_name, [])
                    attn_w_ls_dict[col_name] += attn_w_reshape[agg_name][hop_name] 
                    # print(front_shape, acc_mul, new_shape)

            df_te_gs_info = pd.DataFrame({
                # 'id': feed_dict_te['batch'].numpy().tolist(), # Laurence 20220706
                'id': [n for n in node_id_ls],
                'graph_feat':outs_te[3].numpy().tolist(),
                'graph_pred':outs_te[0].numpy().tolist(),
                'node_feat': node_feat_ls,
                'node_feat_check_sum': node_feat_cs_ls,
                'label': labels_te[:,1].tolist()}).assign(is_train=False)

            # Laurence 20220713
            for k, v in node_neigh_id_ls_dict.items():
                df_te_gs_info = df_te_gs_info.assign(**{f'neigh_{k}': v})

            # Laurence 20220715
            for k, v in attn_w_ls_dict.items():
                df_te_gs_info = df_te_gs_info.assign(**{k: v})
                
            
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

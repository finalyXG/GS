from __future__ import division
from __future__ import print_function

import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 🚩: disable the import warnings and infos in developmet 
import time
import tensorflow as tf


import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data

from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
tf.get_logger().setLevel('ERROR')

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

tf.compat.v1.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 200, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 15, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 5, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 256, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

# Add-hod
flags.DEFINE_boolean('remove_isolated_nodes', True, 'whether to remove nodes with degree=0')
flags.DEFINE_list('k_of_sb', [0,2,4,8], 'The value of k of sb@k')
flags.DEFINE_boolean('train_return_model_once_ready', False, 'Whether to return model without training in train()')
flags.DEFINE_boolean('disable_concat_to_ori_feature', False, 'Whether to disable concatenating learned aggregated features to the original node feature')
flags.DEFINE_float('T', 1.0, 'Temperature used in graphsage attention')

# For analysis flow
# Laurence 20220809
flags.DEFINE_string('save_to_folder_by_date', '', 'Date value yyyymmdd_HHMM as saved folder name')


datetime_now = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
# os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = tf.where(y_pred > 0.5, 1, 0).numpy()
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def calc_acc_precision_recall(y_true, y_pred):
    is_multi_label = True if y_true.shape[1] > 2 else False
    average = 'micro' if is_multi_label else 'binary'    
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = tf.where(y_pred > 0.5, 1, 0).numpy()
    
    s1 = metrics.accuracy_score(y_true, y_pred)
    s2 = metrics.precision_score(y_true, y_pred, average=average)
    s3 = metrics.recall_score(y_true, y_pred, average=average)
    s4 =metrics.f1_score(y_true, y_pred, average=average)
    return s1,s2,s3,s4

def calc_microavg_eval_measures(tp, fn, fp):
    tp_sum = sum(tp.values()).item()
    fn_sum = sum(fn.values()).item()
    fp_sum = sum(fp.values()).item()

    p = tp_sum*1.0 / (tp_sum+fp_sum)
    r = tp_sum*1.0 / (tp_sum+fn_sum)
    if (p+r)>0:
        f1 = 2.0 * (p*r) / (p+r)
    else:
        f1 = 0
    return p, r, f1    


# Define model evaluation function
def evaluate(model, minibatch_iter, epoch, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(test=True)
    node_outs_val = model.test_one_step(feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    # Laurence 20220705
    mic, mac = 0, 0
    '''
    # Laurence 20220426
    Add calc_rdr
    '''
    # outnode_outs_val[0].numpy()[:,1]
    # feed_dict_te, labels_te = minibatch_iter.node_val_feed_dict(test=True) # Laurence 20220705
    feed_dict_te, labels_te = feed_dict_val, labels
    node_outs_te = model.test_one_step(feed_dict_te, return_sampled_nodes=True)
    pred_te = node_outs_te[0].numpy()[:,1]
    labels_te = feed_dict_te['labels'].numpy()[:,1]
    df_te = pd.DataFrame({'pred_te':pred_te,'target': labels_te})
    sorted_pos_score = df_te.query('target==1.0')['pred_te'].sort_values().values
    rdno_at_0 = df_te.query(f'pred_te<{sorted_pos_score[0]}').shape[0]
    rdno_at_1 = df_te.query(f'pred_te<{sorted_pos_score[1]}').shape[0] - 1
    rdno_at_2 = df_te.query(f'pred_te<{sorted_pos_score[2]}').shape[0] - 2
    rdno_at_4 = df_te.query(f'pred_te<{sorted_pos_score[4]}').shape[0] - 4
    rdno_at_8 = df_te.query(f'pred_te<{sorted_pos_score[8]}').shape[0] - 8
    rdno_at_16 = df_te.query(f'pred_te<{sorted_pos_score[16]}').shape[0] - 16
    rdr_info = {'sb@0':rdno_at_0, 
                'sb@1':rdno_at_1, 
                'sb@2':rdno_at_2, 
                'sb@4':rdno_at_4,
                'sb@8':rdno_at_8,
                'sb@16':rdno_at_16,
                }
    rdr_info4cb = copy.deepcopy(rdr_info)
    rdr_info4cb.update({'f1_mic': mic, 'f1_mac': mac,})

    for cb in model.checkpoint_cb_ls:
        cb.epochs_since_last_save += 1
        old_best = cb.best
        cb._save_model(epoch, batch=None, logs=rdr_info4cb)

        if cb.save_best_only:
            current = rdr_info4cb.get(cb.monitor)
            if cb.best != old_best:
                # It means that we just save a new model version, 
                # then we need to save some helper files to verity the model weights.
                sample_to_check_model_weight_path = '/'.join(cb.filepath.split('/')[:-1])
                feed_dict_te_np = {k: v.numpy() for k, v in feed_dict_te.items()}
                feed_dict_te_np.update({
                    'sample': node_outs_te[2][0],
                    'support_sizes': node_outs_te[2][1],
                })
                feed_dict_te_np_result = model.test_one_step(feed_dict_te_np)
                feed_dict_te_np_result = [v.numpy() for v in feed_dict_te_np_result]
                feed_dict_te_np_result.append({'epoch':epoch + 1}) # Align epoch from graphsage to checkpoint. CAN NOT CUT. 
                feed_dict_te_np_result = np.array(feed_dict_te_np_result, dtype=object)
                np.save(sample_to_check_model_weight_path + f'/check_{cb.monitor}.npy',feed_dict_te_np )
                np.save(sample_to_check_model_weight_path + f'/check_{cb.monitor}_result.npy',feed_dict_te_np_result )





    return node_outs_val[1].numpy(), mic, mac, (time.time() - t_test), rdr_info

def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}_{datetime_now}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate,
            datetime_now=datetime_now)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def incremental_evaluate(model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = model.test_one_step(feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    acc, precision, recall, f1 = calc_acc_precision_recall(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], acc, precision, recall, f1, (time.time() - t_test)

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        # 'labels' : tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        # 'batch' : tf.compat.v1.placeholder(tf.int32, shape=(None), name='batch1'),
        # 'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
        # 'batch_size' : tf.compat.v1.placeholder(tf.int32, name='batch_size'),
        # 'labels' : tf.ones(shape=(1, num_classes), dtype=tf.float32, name='labels'),
        # 'batch' : tf.compat.v1.placeholder(tf.int32, shape=(None), name='batch1'),
        # 'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
        # 'batch_size' : tf.compat.v1.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data, test_data=None):

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map  = train_data[4]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = {
        'labels' : tf.cast(tf.compat.v1.distributions.Bernoulli(probs=0.7).sample(sample_shape=(1, num_classes)), tf.float32),
        'batch' : tf.constant(list(G.nodes)[:1], dtype=tf.int32, name='batch1'),
        'dropout': tf.constant(FLAGS.dropout, dtype=tf.float32, name='dropout'),
        'batch_size' : tf.constant(FLAGS.batch_size, dtype=tf.float32, name='batch_size'),
    }
    minibatch = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs)
    # adj_info_ph = tf.constant(minibatch.adj.shape)

    adj_info = tf.constant(minibatch.adj, name="adj_info")
    val_adj_info = tf.constant(minibatch.test_adj, name="val_adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos, 
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
                        
    elif FLAGS.model == 'graphsage_attn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="attn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos, 
                                    aggregator_type="maxpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss = FLAGS.sigmoid,
                                    identity_dim = FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.compat.v1.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    if FLAGS.train_return_model_once_ready:
        return model # Laurence 20220513

    log_dir()
    # Initialize checkpoint
    checkpoint_cb_ls =  []
    for k in FLAGS.k_of_sb:
        one_cb = tf.keras.callbacks.ModelCheckpoint(
                    # filepath = log_dir() + f'weights.{{epoch:03d}}' + f'-sb@{k}-{{sb@{k}}}' + f'f1_mic-{{f1_mic:.3f}}' + f'-f1_mac-{{f1_mac:.3f}}',
                    filepath = log_dir() + f'weights.{{epoch:03d}}' + f'val_precision-{{val_precision:.3f}}' + f'-val_recall-{{val_recall:.3f}}' + f'-val_f1-{{val_f1:.3f}}',
                    save_weights_only=True,
                    # monitor=f'sb@{k}',
                    monitor=f'val_f1',
                    mode='max',
                    verbose=1,
                    save_freq='epoch',
                    save_best_only=True)
        checkpoint_cb_ls.append(one_cb)
        one_cb.model = model
    model.checkpoint_cb_ls = checkpoint_cb_ls
  
    # summary_writer = tf.summary.create_file_writer(log_dir())
     
    # Init variables
    
    # Train model
    
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    # train_adj_info = tf.compat.v1.assign(adj_info, minibatch.adj)
    # val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)
    for epoch in range(FLAGS.epochs): 
        minibatch.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({'dropout': FLAGS.dropout})

            t = time.time()
            # Training step
            outs = model.train_one_step(feed_dict)
            train_cost = outs[1]

            if iter % FLAGS.validate_iter == 0:
                # Validation
                sampler.adj_info = val_adj_info
                if FLAGS.validate_batch_size == -1:
                    val_cost, val_f1_mic, val_f1_mac, val_acc, val_precision, val_recall, val_f1, duration = incremental_evaluate(model, minibatch, FLAGS.batch_size, test=True)
                    rdr_info = {
                        'val_precision':val_precision, 
                        'val_recall':val_recall, 
                        'val_f1':val_f1, 
                        'val_f1_mic':val_f1_mic,
                        'val_f1_mac':val_f1_mac,
                    }
                    assert len(model.checkpoint_cb_ls) == 1
                    cb = model.checkpoint_cb_ls[0]
                    cb.epochs_since_last_save += 1
                    cb._save_model(epoch, batch=None, logs=rdr_info)
                    rdr_info = 0.0
                else:
                    val_precision, val_recall, val_f1 = 0.0, 0.0, 0.0
                    val_cost, val_f1_mic, val_f1_mac, duration, rdr_info = evaluate(model, minibatch, epoch, FLAGS.validate_batch_size)
                sampler.adj_info = adj_info
                epoch_val_costs[-1] += val_cost

            # if total_steps % FLAGS.print_every == 0:
            #     with summary_writer.as_default(step=total_steps):
            #         tf.summary.histogram(name='distribution', data=outs[0])

    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[0])
                print("Iter:", '%04d' % iter, 
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic), 
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac), 
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_acc=", "{:.5f}".format(val_acc),
                      "val_precision=", "{:.5f}".format(val_precision),
                      "val_recall=", "{:.5f}".format(val_recall),
                      "val_f1=", "{:.5f}".format(val_f1),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic), 
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac), 
                      "SB=",rdr_info,
                      "time=", "{:.5f}".format(avg_time))
 
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
                break
    
    # print("Optimization Finished!")
    # sampler.adj_info = val_adj_info
    # val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(model, minibatch, FLAGS.batch_size)
    # print("Full validation stats:",
    #               "loss=", "{:.5f}".format(val_cost),
    #               "f1_micro=", "{:.5f}".format(val_f1_mic),
    #               "f1_macro=", "{:.5f}".format(val_f1_mac),
    #               "time=", "{:.5f}".format(duration))
    # with open(log_dir() + "val_stats.txt", "w") as fp:
    #     fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} time={:.5f}".
    #             format(val_cost, val_f1_mic, val_f1_mac, duration))

    print("Writing test set stats to file (don't peak!)")
    val_cost, val_f1_mic, val_f1_mac, val_acc, val_precision, val_recall, val_f1, duration = incremental_evaluate(model, minibatch, FLAGS.batch_size, test=True)
    with open(log_dir() + "test_stats.txt", "w") as fp:
        fp.write("loss={:.5f} acc={:.5f} f1_micro={:.5f} f1_macro={:.5f}".
                format(val_cost, val_acc, val_f1_mic, val_f1_mac))

def main(argv=None):
    print("Loading training data..")

    train_data = load_data(FLAGS.train_prefix, remove_isolated_nodes=FLAGS.remove_isolated_nodes)
    print("Done loading training data..")
    train(train_data)

if __name__ == '__main__':
    tf.compat.v1.app.run()

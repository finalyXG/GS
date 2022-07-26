import tensorflow as tf

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import AttnAggregator, MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif aggregator_type == 'attn':
            self.aggregator_cls = AttnAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.compat.v1.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build_by_run_once()


    def build_by_run_once(self):
        batch_build = 1
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos, batch_build)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        if FLAGS.model == 'graphsage_attn':
            self.outputs1, self.aggregators, _ = self.aggregate(samples1, [self.features], self.dims, num_samples, support_sizes1, batch_build, name='agg', concat=self.concat, model_size=self.model_size)
        else:
            self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples, support_sizes1, batch_build, name='agg', concat=self.concat, model_size=self.model_size)

        dim_mult = 2 if self.concat else 1

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        # Laurence 20220406: In tf2.0, seems no need to take care of grads_and_vars in build step.
        # grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
        #         for grad, var in grads_and_vars]
        # self.grad, _ = clipped_grads_and_vars[0]
        # self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()

    def _loss(self, node_preds=None, labels=None, is_return=False):
        # Weight decay loss
        self.loss = 0
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        if node_preds is None:
            node_preds = self.node_preds

        if labels is None:
            labels = self.placeholders['labels']
        
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=node_preds,
                    labels=labels))
        else:
            self.loss += tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(
                    logits=node_preds,
                    labels=tf.stop_gradient(labels)))

        # Laurence 20220705
        # tf.compat.v1.summary.scalar('loss', self.loss)
        if is_return:
            return self.loss

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)

    def predict_input(self, input):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(input)
        else:
            return tf.nn.softmax(input)
        


    #Laurence 20220406
    def train_one_step(self, feed_dict):

        with tf.GradientTape() as tape:
            samples1, support_sizes1 = self.sample(feed_dict['batch'], self.layer_infos, feed_dict['batch_size'])
            num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

            if FLAGS.model == 'graphsage_attn':
                outputs1, _, _ = self.aggregate(samples1, [self.features], self.dims, num_samples,
                    support_sizes1, batch_size=feed_dict['batch_size'], aggregators=self.aggregators, concat=self.concat, model_size=self.model_size)
            else:
                outputs1, _ = self.aggregate(samples1, [self.features], self.dims, num_samples,
                    support_sizes1, batch_size=feed_dict['batch_size'], aggregators=self.aggregators, concat=self.concat, model_size=self.model_size)

            dim_mult = 2 if self.concat else 1
            # node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes,
            #         dropout=self.placeholders['dropout'],
            #         act=lambda x : x)
            # TF graph management
            node_preds = self.node_pred(outputs1)
            preds = self.predict_input(node_preds)
            loss = self._loss(node_preds, feed_dict['labels'], is_return=True)

        gradients = [tf.clip_by_value(g, -5.0, 5.0) for g in tape.gradient(loss, self.trainable_variables)]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return preds, loss

    #Laurence 20220407
    def test_one_step(self, feed_dict, return_sampled_nodes=False, return_node_feat=False):
        if 'sample' in feed_dict.keys() and 'support_sizes' in feed_dict.keys():
            samples1 = feed_dict['sample']
            support_sizes1 = feed_dict['support_sizes']
        else:
            samples1, support_sizes1 = self.sample(feed_dict['batch'], self.layer_infos, feed_dict['batch_size'])
            # samples2, support_sizes2 = self.sample(feed_dict['batch'], self.layer_infos, feed_dict['batch_size'])
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]

        outputs1, _ = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, batch_size=feed_dict['batch_size'], aggregators=self.aggregators, concat=self.concat, model_size=self.model_size)

        dim_mult = 2 if self.concat else 1
        # node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, 
        #         dropout=self.placeholders['dropout'],
        #         act=lambda x : x)
        # TF graph management
        node_preds = self.node_pred(outputs1)
        preds = self.predict_input(node_preds)
        loss = self._loss(node_preds, feed_dict['labels'], is_return=True)

        rs = [preds, loss]
        if return_sampled_nodes:
            rs.append((samples1, support_sizes1))
        if return_node_feat:
            rs.append(outputs1)
        return rs

        # if return_sampled_nodes:
        #     return preds, loss, (samples1, support_sizes1)
        # else:       
        #     return preds, loss



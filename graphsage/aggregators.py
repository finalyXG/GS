import tensorflow as tf

from .layers import Layer, Dense
from .inits import glorot, zeros

class AttnAggregator(Layer): 
    def __init__(self, input_dim, output_dim, num_heads=4, neigh_input_dim=None,
        dropout=0.,bias=False, act=tf.nn.relu,
        name=None, concat=False, **kwargs):
        super(AttnAggregator, self).__init__(**kwargs) 
        
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        # if name is not None:
        #     name = '/' + name
        # else:
        #     name = ''

        # init = tf.initializers.GlorotUniform()
        # self.vars_neigh_weights = tf.Variable(init(shape=[neigh_input_dim, output_dim]),
        #                                         trainable=True, name=f'{name}/neigh_weights')
        # self.vars_self_weights = tf.Variable(init(shape=[input_dim, output_dim]),
        #                                         trainable=True, name=f'{name}/self_weights')
        # if self.bias:
        #     self.vars_bias = tf.Variable(tf.initializers.zeros()(shape=[self.output_dim]), name=f'{name}/bias')

        # if self.logging:
        #     self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim # previously d_model: the num of units before spliting
        self.num_heads = num_heads
        assert output_dim % self.num_heads == 0

        self.depth = output_dim//self.num_heads
        self.d = tf.keras.layers.Dense(output_dim)
        self.wk = tf.keras.layers.Dense(output_dim)
        self.wq = tf.keras.layers.Dense(output_dim)
        self.wv = tf.keras.layers.Dense(output_dim)

        self.layer_norm = tf.keras.layers.LayerNormalization()
    def split_heads(self, x, batch_size):
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm=[0,2,1,3]) # (batch_size,num_heads,seq_length,depth)

    def pad_mask(self,feat):
        # check weather all the feat of certain neighbr are zeros
        tmp = tf.cast(tf.math.equal(tf.math.count_nonzero(feat,-1,keepdims=True),0),tf.float32)
        res = tf.transpose(tmp, perm=[0,1,3,2])
        return res
    
    def scaled_dot_product_attention(self,Q,K,V):

        matmul_qk = tf.matmul(Q,K,transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1],tf.float32)
        scaled_qk = matmul_qk/tf.math.sqrt(dk)
        mask = self.pad_mask(K)
        scaled_qk += mask*-1e9 

        attention_weight = tf.nn.softmax(scaled_qk,axis=-1) # (batch size, num_heads, seq_length, seq_length)
        output = tf.matmul(attention_weight,V) # (batch size, num_heads, seq_length, depth)
        return output, attention_weight

    # def _call(self, inputs, **kwargs):
    def __call__(self, inputs, **kwargs):
        self_vecs, neigh_vecs = inputs
        if 'dropout' in kwargs.keys():
            neigh_vecs = tf.nn.dropout(neigh_vecs, rate=1 - (1-self.dropout))
            self_vecs = tf.nn.dropout(self_vecs, rate=1 - (1-self.dropout))

        # self_feat is Q, neigh feat is K & V
        batch_size = tf.shape(self_vecs)[0]
        q = self.wq(self_vecs) # (batch size, unit)
        k = self.wk(neigh_vecs) # (batch size, neighbor size, unit)
        v = self.wv(neigh_vecs) # (batch size, neighbor size, unit)
        q = q[:,tf.newaxis,:]

        Q = self.split_heads(q,batch_size)
        K = self.split_heads(k,batch_size)
        V = self.split_heads(v,batch_size)

        attention, weight = self.scaled_dot_product_attention(Q,K,V)
        attention = tf.transpose(attention, perm=[0,2,1,3]) # (batch size, seq_length, num_heads, depth)
        attention_concat = tf.reshape(attention,(-1,self.output_dim))

        if not self.concat:
            res = tf.add_n([tf.reshape(q,(-1,self.output_dim)),attention_concat])
        else:
            res = tf.concat([tf.reshape(q,(-1,self.output_dim)),attention_concat], axis=1)

        # res = tf.reshape(attention_concat,(-1,self.output_dim))
        return self.act(self.layer_norm(self.d(res))), weight # (seq_length, output_dim)


class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, 
            name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        init = tf.initializers.GlorotUniform()
        self.vars_neigh_weights = tf.Variable(init(shape=[neigh_input_dim, output_dim]),
                                                trainable=True, name=f'{name}/neigh_weights')
        self.vars_self_weights = tf.Variable(init(shape=[input_dim, output_dim]),
                                                trainable=True, name=f'{name}/self_weights')
        if self.bias:
            self.vars_bias = tf.Variable(tf.initializers.zeros()(shape=[self.output_dim]), name=f'{name}/bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs, **kwargs):
        self_vecs, neigh_vecs = inputs

        if 'dropout' in kwargs.keys():
            neigh_vecs = tf.nn.dropout(neigh_vecs, rate=1 - (1-self.dropout))
            self_vecs = tf.nn.dropout(self_vecs, rate=1 - (1-self.dropout))
        neigh_means = tf.reduce_mean(input_tensor=neigh_vecs, axis=1)
       
        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars_neigh_weights)

        from_self = tf.matmul(self_vecs, self.vars_self_weights)
         
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars_bias
       
        return self.act(output)

class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        init = tf.initializers.GlorotUniform()
        self.vars_neigh_weights = tf.Variable(init(shape=[neigh_input_dim, output_dim]),
                                                trainable=True, name=f'{name}/neigh_weights')

        if self.bias:
            self.vars_bias = tf.Variable(tf.initializers.zeros()(shape=[self.output_dim]), name=f'{name}/bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, rate=1 - (1-self.dropout))
        self_vecs = tf.nn.dropout(self_vecs, rate=1 - (1-self.dropout))
        means = tf.reduce_mean(input_tensor=tf.concat([neigh_vecs, 
            tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)
       
        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars_neigh_weights)

        # bias
        if self.bias:
            output += self.vars_bias
       
        return self.act(output)


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(input=neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(input_tensor=neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class MeanPoolingAggregator(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(input=neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_mean(input_tensor=neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class TwoMaxLayerPoolingAggregator(Layer):
    """ Aggregates via pooling over two MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim_1,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))
        self.mlp_layers.append(Dense(input_dim=hidden_dim_1,
                                 output_dim=hidden_dim_2,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))


        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim_2, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(input=neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim_2))
        neigh_h = tf.reduce_max(input_tensor=neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None,  concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                        name='neigh_weights')
           
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                                        name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        dims = tf.shape(input=neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(input_tensor=tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(input_tensor=used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.compat.v1.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
        batch_size = tf.shape(input=rnn_outputs)[0]
        max_len = tf.shape(input=rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        output = tf.add_n([from_self, from_neighs])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


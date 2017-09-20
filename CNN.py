# coding=utf-8
import tensorflow as tf
from tf_seq2seq import embedding_attention_decoder,sequence_loss
class CNN_Encoder:
    def __init__(self,max_len,num_word,embedding_size,filter_sizes,num_filters):
        #max input length
        self.max_len = max_len
        #word vocab size
        self.num_word = num_word
        # cnn input
        self.input_x = tf.placeholder(tf.int32, [None, self.max_len], name="input_x")
        #embedding size
        self.embedding_size  = embedding_size
        # embedding
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.num_word, self.embedding_size ], -1.0, 1.0),
                name="embedding")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        #drop out keep prob
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #filter sizes
        self.filter_sizes = filter_sizes
        #num_filters
        self.num_filters = num_filters
        #conv
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        #concat
        num_filters_total = num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        #drop out layer
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        #output layer
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, 128],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[128]), name="b")
            self.cnn_output = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

        self.session = tf.Session()

class CNN2LSTM:
    def __init__(self,max_len,num_words,embedding_size):
        # learning rate
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        #decoder max len
        self.max_len = max_len
        #decider num_words
        self.num_words = num_words
        #decoder embedding size
        self.embedding_size = embedding_size

        #decoder input placeholder
        #list of placeholder
        self.dec_inputs = []
        for i in range(self.max_len):
            self.dec_inputs.append(tf.placeholder(tf.int32, [None], name=('dec_inp-%d' % i)))
        #target same as the deocoder input unless the last one (null token)
        self.targets = [self.dec_inputs[i + 1] for i in range(len(self.dec_inputs) - 1)]
        self.targets.append(tf.placeholder(tf.int32, [None], name=('target-pad')))
        #cost weights
        self.cost_weights = [tf.ones_like(trg, tf.float32, name='cost_weights')
                             for trg in self.targets]
        #rnn cell
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.embedding_size)
        self.state_size = tf.constant(self.cell.state_size)
        #encoder
        self.encoder = CNN_Encoder(max_len=16, num_word=39, embedding_size=self.embedding_size, filter_sizes=[1,2,3, 4,5],
                          num_filters=50)
        #encoder output
        self.encoder_output = tf.tuple([self.encoder.cnn_output,self.encoder.cnn_output])

        self.attention_memory = self.encoder.embedded_chars
        #decoder

        with tf.variable_scope("seq2seq") as scope:
            #feed previous
            self.dec_outputs, self.dec_states = embedding_attention_decoder(self.dec_inputs,self.encoder_output
                                                                            ,self.attention_memory,self.cell,self.num_words
                                                                            ,self.embedding_size,num_heads=16,output_size=self.num_words,feed_previous=True
                                                                            )
            scope.reuse_variables()
            #non feed previous
            #training using reference as input
            self.outputs, self.states = embedding_attention_decoder(self.dec_inputs,self.encoder_output
                                                                            ,self.attention_memory,self.cell,self.num_words
                                                                           ,self.embedding_size,num_heads=16,output_size=self.num_words)
        #cost
        self.cost = sequence_loss(self.outputs, self.targets,
                                     self.cost_weights, self.num_words)


        #optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        #train func
        self.train_func = self.optimizer.minimize(self.cost)

        self.session = tf.Session()
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver(tf.global_variables())
        self.check_point_path = "train"







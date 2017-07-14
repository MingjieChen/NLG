# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import core as layers_core
from System_Config import system_config
sys_config = system_config()
embedding_dim = sys_config.embedding_dim
batch_size = sys_config.batch_size
learning_rate = sys_config.learning_rate
hidden_size = sys_config.hidden_size
start_token = sys_config.start
end_token = sys_config.end
class LSTM_decoder:
    def __init__(self,sequence_length,num_words,index2words,memory,h0):
        ### parameters
        self.start_token = tf.constant([start_token] * batch_size,shape=[batch_size], dtype=tf.int32,name="start_tokens")
        self.end_token  =tf.constant([end_token] * batch_size ,shape=[batch_size],dtype= tf.int32,name="end_tokens")
        self.h0 = h0
        self.h0_stack = tf.stack([self.h0,self.h0])
        self.index2words = index2words
        self.num_words = num_words
        self.sequence_length = sequence_length
        self.x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.batch_len = tf.placeholder(tf.int32,[],name="batch_len")
        self.memory = memory
        self.mask = tf.placeholder(tf.float32,[None,None],name="input_mask")

        self.embeddings = tf.Variable(tf.random_normal([self.num_words,embedding_dim]),name="embeddings")

        #self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.x), perm=[1, 0, 2])

        ### reference placeholder


        #with tf.device("/cpu:0"):
            #self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.x), perm=[1, 0, 2])
        ### recurrent_unit
        self.g_recurrent_unit = self.create_recurrent_unit()# maps h_tm1 to h_t for generator
        ### outputunit
        self.g_output_unit = self.create_output_unit() # maps h_t to o_t (output token logits)
        ### generated prob and words
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.batch_len,
                                             dynamic_size=False, infer_shape=True,name="gen_o")
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.batch_len,
                                             dynamic_size=False, infer_shape=True,name="gen_x")
        next_token = self.start_token
        self.num_attention_unit = 256
        self.memory_layer = layers_core.Dense(
            self.num_attention_unit, name="memory_layer", use_bias=True,activation=tf.tanh)
        self.query_layer = layers_core.Dense(
            self.num_attention_unit, name="query_layer", use_bias=True,activation=tf.tanh)
        self.processed_memory = self.memory_layer(memory)

        def _g_recurrence(i, x_t, h_tm1, gen_o, gen_x,next_token):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            previous_hidden_state, c_prev = tf.unstack(h_t)# hidden_memory_tuple

            query = self.query_layer(previous_hidden_state)
            query = array_ops.expand_dims(query, 1)
            score = math_ops.matmul(query, self.processed_memory, transpose_b=True)
            score = array_ops.squeeze(score, [1])
            alignments = tf.nn.softmax(score)
            expanded_alignments = array_ops.expand_dims(alignments, 1)
            context = math_ops.matmul(expanded_alignments, self.memory)
            context = array_ops.squeeze(context, [1])
            new_h_t = tf.tanh(tf.concat([previous_hidden_state,context],axis=-1))
            o_t = self.g_output_unit(new_h_t)  # batch x vocab , logits not prob
            prob = tf.nn.softmax(o_t)

            #log_prob = tf.log(tf.nn.softmax(o_t))

            next_token = tf.cast(tf.reshape(tf.argmax(prob,axis=-1),[batch_size]),tf.int32)
            #next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.embeddings, next_token)  # batch x emb_dim
            #gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.num_words, 1.0, 0.0),
                                                             #tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_o = gen_o.write(i,prob)
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o, gen_x,next_token

        _, _, _, self.gen_o, self.gen_x,next_token = control_flow_ops.while_loop(
            cond= lambda  i,_1, _2, _3, _4,_5: i<self.batch_len,
            #use end token or not it is a question
            #cond=lambda i, _1, _2, _3, _4,_5: math_ops.logical_and(i < self.batch_len,math_ops.reduce_all(math_ops.logical_not(tf.equal(next_token,self.end_token)))) ,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.embeddings,self.start_token), self.h0_stack, gen_o, gen_x,next_token))
        self.generations = tf.reshape(self.gen_x.concat(), [batch_size, self.batch_len])
        self.gen_prob = tf.reshape(self.gen_o.concat(),[batch_size,self.batch_len,self.num_words])
        self.ref = tf.one_hot(self.x,self.num_words,1.0,0.0)
        #self.loss =-tf.reduce_sum(tf.reduce_sum(tf.log(self.gen_prob)*self.ref,axis=-1)*self.mask)/(tf.cast(self.batch_len,dtype=tf.float32)*tf.cast(batch_size,dtype=tf.float32))
        self.loss = -tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.log(self.gen_prob) * self.ref, axis=-1) * self.mask,axis=-1)/tf.reduce_sum(self.mask,axis=-1))
        self.opt = tf.train.AdamOptimizer(learning_rate=0.5e-6)
        grads_vars = self.opt.compute_gradients(self.loss)
        capped_grads_vars = [(tf.clip_by_value(g, -5, 5), v)
                             for g, v in grads_vars]  # gradient capping
        self.train_op = self.opt.apply_gradients(capped_grads_vars, tf.Variable(0, trainable=False))
        self.saver = tf.train.Saver(tf.global_variables())

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([embedding_dim, hidden_size]),name="Wi")
        self.Ui = tf.Variable(self.init_matrix([hidden_size, hidden_size]),name="Ui")
        self.bi = tf.Variable(self.init_matrix([hidden_size]),name="bi")

        self.Wf = tf.Variable(self.init_matrix([embedding_dim, hidden_size]),name="Wf")
        self.Uf = tf.Variable(self.init_matrix([hidden_size, hidden_size]),name="Uf")
        self.bf = tf.Variable(self.init_matrix([hidden_size]),name="bf")

        self.Wog = tf.Variable(self.init_matrix([embedding_dim, hidden_size]),name="Wog")
        self.Uog = tf.Variable(self.init_matrix([hidden_size, hidden_size]),name="Uog")
        self.bog = tf.Variable(self.init_matrix([hidden_size]),"bog")

        self.Wc = tf.Variable(self.init_matrix([embedding_dim, hidden_size]),"Wc")
        self.Uc = tf.Variable(self.init_matrix([hidden_size, hidden_size]),name="Uc")
        self.bc = tf.Variable(self.init_matrix([hidden_size]),name="bc")


        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
                )

            # Output Gate
            o = tf.sigmoid(
                    tf.matmul(x, self.Wog) +
                    tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                    tf.matmul(x, self.Wc) +
                    tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])# batch * (2*hiddensize)

        return unit

    def create_output_unit(self):
        self.Wo = tf.Variable(self.init_matrix([hidden_size+sys_config.readin_hidden_size, self.num_words]),name="Wo")
        self.bo = tf.Variable(self.init_matrix([self.num_words]),name="bo")


        def unit(hidden_memory_tuple):
            #hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_memory_tuple, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def train(self, sess,x,mask,batch_len,vectors,encoder):
        feed_dict = {self.x:x,self.mask :mask,self.batch_len : batch_len,
                     encoder.family_friendly_X: [vec["familyfriendly"] for vec in vectors]
            , encoder.eatType_X: [vec["eattype"] for vec in vectors]
            , encoder.food_X: [vec["food"] for vec in vectors]
            , encoder.near_X: [vec["near"] for vec in vectors]
            , encoder.name_X: [vec["name"] for vec in vectors]
            , encoder.area_X: [vec["area"] for vec in vectors]
            , encoder.priceRange_X: [vec["pricerange"] for vec in vectors]
            , encoder.customer_rating_X: [vec["customer_rating"] for vec in vectors]
                     }
        loss,_,gen = sess.run([self.loss,self.train_op,self.generations],feed_dict)
        #outputs = sess.run(self.batch_len,feed_dict)
        return loss,gen

    def test(self, sess,x,mask,batch_len,vectors,encoder):
        feed_dict = {self.x:x,self.mask :mask,self.batch_len : batch_len,
                     encoder.family_friendly_X: [vec["familyfriendly"] for vec in vectors]
            , encoder.eatType_X: [vec["eattype"] for vec in vectors]
            , encoder.food_X: [vec["food"] for vec in vectors]
            , encoder.near_X: [vec["near"] for vec in vectors]
            , encoder.name_X: [vec["name"] for vec in vectors]
            , encoder.area_X: [vec["area"] for vec in vectors]
            , encoder.priceRange_X: [vec["pricerange"] for vec in vectors]
            , encoder.customer_rating_X: [vec["customer_rating"] for vec in vectors]
                     }
        loss,generations = sess.run([self.loss,self.generations],feed_dict)
        #outputs = sess.run(self.batch_len,feed_dict)
        return loss,generations

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)


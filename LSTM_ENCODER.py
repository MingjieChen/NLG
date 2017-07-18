# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
num_attr_value = 39
embedding_dim = 128
batch_size = 32
hidden_dim = 128
max_en_len = 16
class LSTM_encoder:
    def __init__(self):
        self.en_embedding = tf.Variable(tf.random_normal([num_attr_value,embedding_dim]),name="encoder_embedding")
        self.x = tf.placeholder(tf.int32,[batch_size,max_en_len],name="encoder_input")
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.en_embedding, self.x), perm=[1, 0, 2])
        self.start_token = tf.constant([0] * batch_size, shape=[batch_size], dtype=tf.int32,
                                       name="start_tokens")
        ### recurrent_unit
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        ### outputunit
        self.h0 = tf.zeros([batch_size, hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=max_en_len,
                                             dynamic_size=False, infer_shape=True, name="en_gen_o")
        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=max_en_len,infer_shape=True)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)
        def _g_recurrence(i, x_t, h_tm1, gen_o):
            h_t = self.g_recurrent_unit(x_t, h_tm1)
            x_tp1 = ta_emb_x.read(i)
            previous_hidden_state, c_prev = tf.unstack(h_t)




            gen_o = gen_o.write(i, previous_hidden_state)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_o
        _,_,self.last_state,self.gen_o = control_flow_ops.while_loop(cond =
                                                                lambda  i,_1,_2,_3:i<max_en_len,body=_g_recurrence,
                                                                loop_vars=(tf.constant(0, dtype=tf.int32),
                                                                           tf.nn.embedding_lookup(self.en_embedding,
                                                                                                  self.start_token), self.h0,gen_o ))
        self.memory = tf.reshape(self.gen_o.concat(),[batch_size,max_en_len,hidden_dim])









    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([embedding_dim, hidden_dim]), name="en_Wi")
        self.Ui = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]), name="en_Ui")
        self.bi = tf.Variable(self.init_matrix([hidden_dim]), name="en_bi")

        self.Wf = tf.Variable(self.init_matrix([embedding_dim, hidden_dim]), name="en_Wf")
        self.Uf = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]), name="en_Uf")
        self.bf = tf.Variable(self.init_matrix([hidden_dim]), name="en_bf")

        self.Wog = tf.Variable(self.init_matrix([embedding_dim, hidden_dim]), name="en_Wog")
        self.Uog = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]), name="en_Uog")
        self.bog = tf.Variable(self.init_matrix([hidden_dim]), "en_bog")

        self.Wc = tf.Variable(self.init_matrix([embedding_dim, hidden_dim]), "en_Wc")
        self.Uc = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]), name="en_Uc")
        self.bc = tf.Variable(self.init_matrix([hidden_dim]), name="en_bc")

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

            return tf.stack([current_hidden_state, c])  # batch * (2*hiddensize)

        return unit



    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def init_vector(self, shape):
        return tf.zeros(shape)



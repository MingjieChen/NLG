# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tf_seq2seq import sequence_loss#, embedding_attention_decoder
#from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.contrib.rnn import EmbeddingWrapper, RNNCell, OutputProjectionWrapper

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access
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
    def __init__(self,max_len,num_words,embedding_size,encoder_max_len,encoder_input_size):
        self.encoder_max_len = encoder_max_len
        self.encoder_input_size = encoder_input_size
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
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.embedding_size
                                                 #,input_size = self.embedding_size
                                                 )
        self.state_size = tf.constant(self.cell.state_size)
        #encoder
        self.encoder = CNN_Encoder(max_len=self.encoder_max_len, num_word=self.encoder_input_size, embedding_size=self.embedding_size, filter_sizes=[1,2,3, 4,5],
                          num_filters=50)
        #encoder output
        self.encoder_output = tf.tuple([self.encoder.cnn_output,self.encoder.cnn_output])

        self.attention_memory = self.encoder.embedded_chars
        #decoder

        with tf.variable_scope("seq2seq") as scope:
            #feed previous
            self.dec_outputs, self.dec_states = embedding_attention_decoder(self.dec_inputs,self.encoder_output
                                                                            ,self.attention_memory,self.cell,self.num_words
                                                                            ,self.embedding_size,num_heads=self.encoder_max_len,output_size=self.num_words,feed_previous=True
                                                                            )
            scope.reuse_variables()
            #non feed previous
            #training using reference as input
            self.outputs, self.states = embedding_attention_decoder(self.dec_inputs,self.encoder_output
                                                                            ,self.attention_memory,self.cell,self.num_words
                                                                           ,self.embedding_size,num_heads=self.encoder_max_len,output_size=self.num_words)

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
        self.check_point_path = "CNN_train"

def embedding_rnn_decoder(decoder_inputs, initial_state, cell, num_symbols,embedding_size,
                          output_projection=None, feed_previous=False,
                          scope=None):


    """RNN decoder with embedding and a pure-decoding option.
    Args:
        decoder_inputs: a list of 1D batch-sized int32 Tensors (decoder inputs).
        initial_state: 2D Tensor [batch_size x cell.state_size].
        cell: RNNCell defining the cell function.
        num_symbols: integer, how many symbols come into the embedding.
        output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [cell.output_size x num_symbols] and B has
        shape [num_symbols]; if provided and feed_previous=True, each fed
        previous output will first be multiplied by W and added B.
        feed_previous: Boolean; if True, only the first of decoder_inputs will be
        used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
        In effect, this implements a greedy decoder. It can also be used
        during training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
        If False, decoder_inputs are used as given (the standard decoder case).
        scope: VariableScope for the created subgraph; defaults to
        "embedding_rnn_decoder".
    Returns:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x cell.output_size] containing the generated outputs.
        states: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        Each item is a 2D Tensor of shape [batch_size x cell.state_size].
    Raises:
        ValueError: when output_projection has the wrong shape.
    """
    if output_projection is not None:
        proj_weights = ops.convert_to_tensor(
            output_projection[0], dtype=dtypes.float32)
        proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                        num_symbols])
        proj_biases = ops.convert_to_tensor(
            output_projection[1], dtype=dtypes.float32)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with vs.variable_scope(scope or "embedding_rnn_decoder"):
        with ops.device("/cpu:0"):
            embedding = vs.get_variable("embedding", [num_symbols, embedding_size])

        def extract_argmax_and_embed(prev, _):
            """Loop_function that extracts the symbol from prev and embeds it."""
            if output_projection is not None:
                prev = nn_ops.xw_plus_b(
                prev, output_projection[0], output_projection[1])
            prev_symbol = array_ops.stop_gradient(math_ops.argmax(prev, 1))
            return embedding_ops.embedding_lookup(embedding, prev_symbol)

        loop_function = None
        if feed_previous:
            loop_function = extract_argmax_and_embed

        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return rnn_decoder(emb_inp, initial_state, cell,output_size = num_symbols,
                       loop_function=loop_function)

def rnn_decoder(decoder_inputs, initial_state,cell,output_size ,loop_function=None,
                scope=None):
    """RNN decoder for the sequence-to-sequence model.
    Args:
        decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].

        cell: RNNCell defining the cell function and size.
        output_size: output size int32.
        loop_function: if not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
        Signature -- loop_function(prev, i) = next
            * prev is a 2D Tensor of shape [batch_size x cell.output_size],
            * i is an integer, the step number (when advanced control is needed),
            * next is a 2D Tensor of shape [batch_size x cell.input_size].
        scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
    Returns:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x cell.output_size] containing generated outputs.
        states: The state of each cell in each time-step. This is a list with
        length len(decoder_inputs) -- one item for each time-step.
        Each item is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
        states can be the same. They are different for LSTM cells though.)
    """

    with vs.variable_scope(scope or "rnn_decoder"):
        states = [initial_state]
        outputs = []

        prev = None


        for i in range(len(decoder_inputs)):
            if i > 0:
                vs.get_variable_scope().reuse_variables()
            inp = decoder_inputs[i]
            if loop_function is not None and prev is not None:
                with vs.variable_scope("loop_function", reuse=True):
                    #We do not propagate gradients over the loop function.
                    inp = array_ops.stop_gradient(loop_function(prev, i))



            output, new_state = cell(inp, states[-1])

            output = linear(output,output_size,True)
            outputs.append(output)
            states.append(new_state)

            if loop_function is not None:
                prev = array_ops.stop_gradient(output)
    return outputs, states

def attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, num_heads=1, loop_function=None,
                      dtype=dtypes.float32, scope=None):
    """RNN decoder with attention for the sequence-to-sequence model.
    Args:
        decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: RNNCell defining the cell function and size.
        output_size: size of the output vectors; if None, we use cell.output_size.
        num_heads: number of attention heads that read from attention_states.
        loop_function: if not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
        Signature -- loop_function(prev, i) = next
            * prev is a 2D Tensor of shape [batch_size x cell.output_size],
            * i is an integer, the step number (when advanced control is needed),
            * next is a 2D Tensor of shape [batch_size x cell.input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
    Returns:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of shape
        [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either i-th decoder_inputs or
        loop_function(output {i-1}, i)) as follows. First, we run the cell
        on a combination of the input and previous attention masks:
        cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
            new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
            output = linear(cell_output, new_attn).
        states: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        Each item is a 2D Tensor of shape [batch_size x cell.state_size].
    Raises:
        ValueError: when num_heads is not positive, there are no inputs, or shapes
        of attention_states are not set.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with vs.variable_scope(scope or "attention_decoder"):
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(
            attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = vs.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(vs.get_variable("AttnV_%d" % a, [attention_vec_size]))

        states = [initial_state]


    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      for a in range(num_heads):
        with vs.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(
              v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
              [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds


    outputs = []
    prev = None
    batch_attn_size = array_ops.stack([batch_size, attn_size])

    attention_cell = tf.random_normal(batch_attn_size,name="attention_cell")

    attention_cells = [attention_cell]
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
             for _ in range(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
        a.set_shape([None, attn_size])
    with vs.variable_scope("Attn_cell"):
        attention_cell_W1 = vs.get_variable("attention_cell_W1", [attn_size, attn_size])
        attention_cell_W2 = vs.get_variable("attention_cell_W2", [attn_size, attn_size])
    for i in range(len(decoder_inputs)):
        if i > 0:
            vs.get_variable_scope().reuse_variables()
        inp = decoder_inputs[i]
        # If loop_function is set, we use it instead of decoder_inputs.
        if loop_function is not None and prev is not None:
            with vs.variable_scope("loop_function", reuse=True):
                inp = array_ops.stop_gradient(loop_function(prev, i))
        # Merge input and previous attentions into one vector of the right size.
        input_size = inp.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from input: %s" % inp.name)
        x = linear([inp] + attns, input_size, True)
        # Run the RNN.
        cell_output, new_state = cell(x, states[-1])
        states.append(new_state)
        # Run the attention mechanism.
        attns = attention(new_state)
        with vs.variable_scope("Attn_cell"):

            attention_cell = tf.add(tf.matmul(attention_cells[-1],attention_cell_W1),tf.matmul(linear(attns,attn_size,True),attention_cell_W2))
            attention_cells.append(attention_cell)
        with vs.variable_scope("AttnOutputProjection"):
            output = linear([cell_output] + attns+[attention_cell], output_size, True)
        if loop_function is not None:
            # We do not propagate gradients over the loop function.
            prev = array_ops.stop_gradient(output)
        outputs.append(output)

    return outputs, states

def embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, num_symbols, embedding_size, num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False, dtype=dtypes.float32,
                                scope=None):
  """RNN decoder with embedding and attention and a pure-decoding option.
  Args:
    decoder_inputs: a list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: RNNCell defining the cell function.
    num_symbols: integer, how many symbols come into the embedding.
    num_heads: number of attention heads that read from attention_states.
    output_size: size of the output vectors; if None, use cell.output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x output_size] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
  Raises:
    ValueError: when output_projection has the wrong shape.
  """
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                        num_symbols])
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with vs.variable_scope(scope or "embedding_attention_decoder"):
    with ops.device("/cpu:0"):
      embedding = vs.get_variable("embedding", [num_symbols, embedding_size])

    def extract_argmax_and_embed(prev, _):
      """Loop_function that extracts the symbol from prev and embeds it."""
      if output_projection is not None:
        prev = nn_ops.xw_plus_b(
            prev, output_projection[0], output_projection[1])
      prev_symbol = array_ops.stop_gradient(math_ops.argmax(prev, 1))
      emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
      return emb_prev

    loop_function = None
    if feed_previous:
      loop_function = extract_argmax_and_embed

    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    return attention_decoder(
        emb_inp, initial_state, attention_states, cell, output_size=output_size,
        num_heads=num_heads, loop_function=loop_function)



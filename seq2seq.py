# coding=utf-8
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq,sequence_loss
import os
import numpy as np
import random,re
from System_Config  import system_config
from data_process_E2E import Config,Data_Instance
import pickle
from bleu import sentence_bleu,SmoothingFunction
import logging
log_file = "train.log"
logger = logging.getLogger(log_file)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
#ch = logging.StreamHandler()
#ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#ch.setFormatter(formatter)
fh.setFormatter(formatter)
#logger.addHandler(ch)
logger.addHandler(fh)
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
f = open("raw_data.pickle","rb")
config = pickle.load(f)
for attr, value in config.attr_values.items():
    config.attr_values[attr] = list(value)

word2index = {word : ind for ind,word in enumerate(config.words)}

index2word = {index : word for word, index in word2index.items()}

batch_size = 32
num_words = len(config.words)
max_epoch = 100
max_len =80
information = ["@start@","@unk@attr","@unk@value"]
for attr, values in config.attr_values.items():

    if attr not in information:
        information.append(attr)
    for value in values:

        if attr+"="+value not in information:
            information.append(attr+"="+value)


info_dict = {x: index for index ,x in enumerate(information)}


class Seq2seq:
    def __init__(self,num_words):
        self.enc_inputs = []
        self.dec_inputs = []
        self.max_da_len = 16
        self.da_dict_size = 39
        self.max_sentence_len = max_len
        self.tree_dict_size = num_words
        self.emb_size = 128
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.emb_size)
        for i in range(self.max_da_len):
            #one time step batch
            enc_input = tf.placeholder(tf.int32, [None], name=('enc_inp-%d' % i))
            self.enc_inputs.append(enc_input)
        self.dec_inputs = []
        for i in range(self.max_sentence_len):
            self.dec_inputs.append(tf.placeholder(tf.int32, [None], name=('dec_inp-%d' % i)))
        self.targets = [self.dec_inputs[i + 1] for i in range(len(self.dec_inputs) - 1)]
        self.targets.append(tf.placeholder(tf.int32, [None], name=('target-pad')))
        with tf.variable_scope("seq2seq") as scope:
            rnn_func = embedding_attention_seq2seq
            self.outputs, self.states = rnn_func(
                self.enc_inputs,
                self.dec_inputs, self.cell,
                self.da_dict_size, self.tree_dict_size,
                self.emb_size,
                scope=scope)
            scope.reuse_variables()
            self.dec_outputs, self.dec_states = rnn_func(
                self.enc_inputs, self.dec_inputs, self.cell,
                self.da_dict_size, self.tree_dict_size,
                self.emb_size,
                feed_previous=True, scope=scope)
        self.cost_weights = [tf.ones_like(trg, tf.float32, name='cost_weights')
                             for trg in self.targets]
        self.tf_cost = sequence_loss(self.outputs, self.targets,
                                             self.cost_weights, self.tree_dict_size)
        self.dec_cost = sequence_loss(self.dec_outputs, self.targets,
                                              self.cost_weights, self.tree_dict_size)
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.cost = self.tf_cost
        self.train_func = self.optimizer.minimize(self.cost)

        self.session = tf.Session()
        init = tf.global_variables_initializer()

        self.session.run(init)

        self.saver = tf.train.Saver(tf.global_variables())
    def train(self,feed_dict):
       output, _, cost = self.session.run([self.dec_outputs,self.train_func, self.cost], feed_dict=feed_dict)
       return output, cost

    def generate(self,feed_dict):
        output,cost = self.session.run([self.dec_outputs,self.cost],feed_dict = feed_dict)
        return output,cost

def vectorize(data):
    input_vecotors = []
    input_reference = []
    input_extra_references = []
    input_delmap = []
    max_length = 0
    for di in data:
        vec = []
        for attr in config.attr_values.keys():

            if attr in di.attr_values:
                value = di.attr_values[attr]
                vec.append(info_dict[attr])
                vec.append(info_dict[attr+"="+value])
        input_vecotors.append(vec)

        while len(vec)<16:
            vec.insert(0, 1)
            vec.insert(0, 0)
        ref = [word2index[w] for w in di.reference.split()]
        ref.insert(0,1)


        input_reference.append(ref)
        input_extra_references.append(list(di.extral_references))
        input_delmap.append(di.delexical_map)
    return input_vecotors, input_reference, input_extra_references,input_delmap


def next_batch(input_vectors, input_reference,input_extra_references ,map,batch_size):
    index = random.sample(list(np.arange(len(input_vectors))), batch_size)
    batch_vectors = [input_vectors[i] for i in index]

    batch_reference = [input_reference[i] for i in index]
    batch_extral_ref = [input_extra_references[i] for i in index]
    batch_map = [map[i] for i in index]
    for r in batch_reference:

        while len(r)< max_len:
            r.append(0)
        if len(r)!=max_len:
            print(len(r))
            print(max_len)
            print(r)
            exit(0)

    batch_vectors = np.array(batch_vectors).T
    batch_reference = np.array(batch_reference).T
    return batch_vectors, batch_reference,batch_map,batch_extral_ref

def evaluate(test_generations,extal_reference,del_map):
    #test_generations  = np.array(test_generations)*mask
    bleu_score = 0.0
    for ind,w in enumerate(test_generations):
        gen = []
        end = False
        for i in w:
            if i == 2 :
                end = True
            if not end and i!= 0:
                gen.append(index2word[i])

        reference = [[i.strip() for i in s.split()] for s in extal_reference[ind]]
        replaced_gen = []
        for w in gen:
            if w in del_map[ind]:
                replaced_gen.extend(del_map[ind][w].split())
            else:
                replaced_gen.append(w)
        '''
        print("gen : ",gen)
        for i in reference:
            print("ref : ",i)
        print("\n")
        '''
        if len(replaced_gen)<=4:
            bleu = sentence_bleu(reference,replaced_gen,smoothing_function=SmoothingFunction().method2,weights=(0.5,0.5))
        else:
            bleu = sentence_bleu(reference, replaced_gen,smoothing_function=SmoothingFunction().method2)
        bleu_score+=bleu
    bleu_score = bleu_score/len(test_generations)
    return bleu_score
t_input_vecotors, t_input_reference, t_input_extra_references,t_input_delmap=vectorize(config.train_instances)
v_input_vecotors, v_input_reference, v_input_extra_references,v_input_delmap=vectorize(config.dev_instances)
model = Seq2seq(num_words)
logger.info("start training")
for epo in range(max_epoch):
    B = []
    L = []

    for _ in range(100):
        v_batch_vectors, v_batch_reference, v_batch_map, v_batch_extral_ref = next_batch(v_input_vecotors, v_input_reference,
                                                                                 v_input_extra_references,
                                                                                 v_input_delmap, batch_size)
        v_feed_dict = {model.learning_rate : 1e-6}
        for i in range(len(model.enc_inputs)):
            v_feed_dict[model.enc_inputs[i]] = v_batch_vectors[i]
        for i in range(len(model.dec_inputs)):
            v_feed_dict[model.dec_inputs[i]] = v_batch_reference[i]
        v_feed_dict[model.targets[-1]] = [0]*batch_size
        v_output,v_cost = model.generate(v_feed_dict)
        v_result = np.argmax(v_output, axis=-1).T
        v_bleu = evaluate(v_result, v_batch_extral_ref, v_batch_map)
        B.append(v_bleu)
        L.append(v_cost)
    print("epoch {} valid bleu score {} valid loss {} ".format(epo, np.mean(B), np.mean(L)))
    logger.info("epoch {} valid bleu score {} valid loss {} ".format(epo, np.mean(B), np.mean(L)))




    for n in range(1000):
        batch_vectors, batch_reference,batch_map,batch_extral_ref = next_batch(t_input_vecotors
                                                                               ,t_input_reference,t_input_extra_references
                                                                               ,t_input_delmap,batch_size)

        feed_dict = {model.learning_rate : 1e-6}
        for i in range(len(model.enc_inputs)):
            feed_dict[model.enc_inputs[i]] = batch_vectors[i]
        for i in range(len(model.dec_inputs)):
            feed_dict[model.dec_inputs[i]] = batch_reference[i]
        feed_dict[model.targets[-1]] = [0]*batch_size
        output,cost=model.train(feed_dict)

        result =  np.argmax(output,axis=-1).T
        bleu = evaluate(result,batch_extral_ref,batch_map)

        if n % 200 == 0:
            print("epoch :{}, train loss :{}".format(epo, cost))
            print("bleu score :{}".format(bleu))
            logger.info("epoch :{}, train loss :{}".format(epo, cost))
            logger.info("bleu score :{}".format(bleu))



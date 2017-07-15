# coding=utf-8


import tensorflow as tf
import numpy as np
import random,re
from System_Config  import system_config
from data_process_E2E import Config,Data_Instance
import pickle
import os
from MLP import MLP_Encoder
from LSTM import LSTM_decoder
#import nltk
from bleu import sentence_bleu,SmoothingFunction
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
f = open("raw_data.pickle","rb")
config = pickle.load(f)
for attr, value in config.attr_values.items():
    config.attr_values[attr] = list(value)

word2index = {word : ind for ind,word in enumerate(config.words)}

index2word = {index : word for word, index in word2index.items()}

batch_size = 32
num_words = len(config.words)

sys_config = system_config()
max_epoch = 200



def vectorize(data):
    input_vecotors = []
    input_reference = []
    input_extra_references = []
    input_delmap = []
    max_length=0
    for di in data:




        vec = {}



        for attr in config.attr_values.keys():
            vec[attr] = np.zeros(len(config.attr_values[attr]))

        for attr in di.attr_values:
            value = di.attr_values[attr]

            vec[attr][config.attr_values[attr].index(value)] = 1.0
        input_vecotors.append(vec)

        ref = [word2index[w] for w in di.reference.split()]
        if len(ref) > max_length:
            max_length = len(ref)

        #for extral_ref in di.extral_references:
            #external_ref.append([config.words[w] for w in extral_ref.split()])
        input_reference.append(ref)
        #input_extra_references.append(external_ref)

        input_extra_references.append(list(di.extral_references))
        input_delmap.append(di.delexical_map)
    return input_vecotors, input_reference, input_extra_references,input_delmap,max_length


def next_batch(input_vectors, input_reference,input_extra_references ,map,batch_size,max_len):

    #max_len = 0

    index = random.sample(list(np.arange(len(input_vectors))), batch_size)
    batch_vectors = [input_vectors[i] for i in index]
    batch_mask = []
    batch_reference = [input_reference[i] for i in index]
    batch_extral_ref = [input_extra_references[i] for i in index]
    batch_map = [map[i] for i in index]
    #for s in batch_reference:
        #if len(s)> max_len:
            #max_len = len(s)


    for r in batch_reference:

        while len(r)< max_len:
            r.append(0)
        if len(r)!=max_len:
            print(r)
            exit(0)

    for r in batch_reference:
        mask = []
        for ind in r:
            if ind==0:
                mask.append(0)
            else:
                mask.append(1)
        batch_mask.append(mask)





    return batch_vectors, batch_reference,batch_mask,max_len,batch_map,batch_extral_ref

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
logger.info("vectorize train data")
train_vecotors, train_reference,train_extral_reference ,train_map, sequence_length = vectorize(config.train_instances)
logger.info("vectorize dev data")
valid_vectors,valid_reference,valid_extral_reference,valid_map,valid_sequence_length = vectorize(config.dev_instances)
logger.info("initial encoder")
encoder = MLP_Encoder()
logger.info("initial decoder")
decoder = LSTM_decoder(sequence_length, num_words, index2word,encoder.memory,encoder.mlp_out)
logger.info("start training")




with tf.Session() as sess:
    init = tf.global_variables_initializer()
    epoch = 0
    sess.run(init)

    for i in range(max_epoch):
        for n in range(1000):
            vectors, ref, mask, batch_len ,batch_map,batch_ref= next_batch(train_vecotors, train_reference, train_extral_reference, train_map,
                                                       batch_size,sequence_length)





            train_loss,gen = decoder.train(sess, ref, mask, batch_len, vectors, encoder)

            if n%200==0:
                print("epoch :{}, train loss :{}".format(i,train_loss))
                print("bleu score :{}".format(evaluate(gen,batch_ref,batch_map)))
                logger.info("epoch :{}, train loss :{}".format(i,train_loss))
                logger.info("bleu score :{}".format(evaluate(gen,batch_ref,batch_map)))


        B = []
        L = []
        for x in range(130):
            t_vectors,t_ref,t_mask,t_batch_len,t_map,t_batch_ref = next_batch(valid_vectors,valid_reference,valid_extral_reference,valid_map,batch_size,valid_sequence_length)

            test_loss,test_generation = decoder.test(sess,t_ref,t_mask,t_batch_len,t_vectors,encoder)

            b = evaluate(test_generation,t_batch_ref,t_map)
            B.append(b)
            L.append(test_loss)
        print("epoch {} valid bleu score {} valid loss {} ".format(i,np.mean(B),np.mean(L)))
        logger.info("epoch {} valid bleu score {} valid loss {} ".format(i,np.mean(B),np.mean(L)))























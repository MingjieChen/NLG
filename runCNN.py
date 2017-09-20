# coding=utf-8
from CNN import CNN_Encoder,CNN2LSTM
from System_Config  import system_config
from data_process_E2E import Config,Data_Instance
import pickle
import numpy as np
import tensorflow as tf
import random
from bleu import sentence_bleu, SmoothingFunction
import logging
random.seed(123)
information = ['@start@', '@unk@attr', '@unk@value', 'eattype', 'eattype=restaurant',
               'eattype=pub', 'eattype=coffee shop', 'food', 'food=japanese', 'food=chinese'
    , 'food=italian', 'food=indian', 'food=fast food', 'food=french', 'food=english', 'name',
               'name=@x@name', 'customer_rating', 'customer_rating=3 out of 5',
               'customer_rating=1 out of 5', 'customer_rating=low', 'customer_rating=high',
               'customer_rating=5 out of 5', 'customer_rating=average', 'familyfriendly',
               'familyfriendly=no', 'familyfriendly=yes', 'near', 'near=@x@near', 'area',
               'area=city centre', 'area=riverside', 'pricerange', 'pricerange=moderate',
               'pricerange=less than £20', 'pricerange=high', 'pricerange=£20-25',
               'pricerange=more than £30', 'pricerange=cheap']
info_dict = {x: index for index ,x in enumerate(information)}
def vectorize(data,config,word2index):
    #word index


    input_vectors = []
    input_reference = []
    input_extra_references = []
    input_delmap = []
    for di in data:
        vec = []
        for attr in config.attr_values.keys():

            if attr in di.attr_values:
                value = di.attr_values[attr]
                vec.append(info_dict[attr])
                vec.append(info_dict[attr + "=" + value])
        vec = sorted(vec)

        while len(vec) < 16:
            vec.insert(0, 1)
            vec.insert(0, 0)
        input_vectors.append(vec)

        ref = [word2index[w] for w in di.reference.split()]
        ref.insert(0, 1)

        input_reference.append(ref)
        input_extra_references.append(list(di.extral_references))
        input_delmap.append(di.delexical_map)
    return input_vectors, input_reference, input_extra_references, input_delmap
def next_batch(input_vectors, input_reference,input_extra_references ,map,max_len,batch_size):
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

    batch_reference = np.array(batch_reference).T
    return batch_vectors, batch_reference,batch_extral_ref,batch_map
def train(model,config,batch_size,word2index,index2word):
    #train on one batch
    train_vec,train_ref,train_extra,train_map = vectorize(config.train_instances,config,word2index)
    batch_vec,batch_ref,batch_extra,batch_map = next_batch(train_vec,train_ref,train_extra,train_map,80,32)
    feed_dict = {model.learning_rate: 5e-4,model.encoder.input_x:batch_vec}
    feed_dict[model.encoder.dropout_keep_prob] = 1.0
    for i in range(len(model.dec_inputs)):
        feed_dict[model.dec_inputs[i]] = batch_ref[i]
    feed_dict[model.targets[-1]] = [0] * batch_size
    result,_ ,cost = model.session.run([model.dec_outputs,model.train_func,model.cost] ,feed_dict)

    generation = np.argmax(result, axis=-1).T
    B = []
    for ind,r in enumerate(generation):

        gen = []
        end = False
        for i in r:
            if i == 2:
                end = True
            if not end and i != 0:
                gen.append(index2word[i])

        replaced_gen = []
        for w in gen:
            if w in batch_map[ind]:
                replaced_gen.extend(batch_map[ind][w].split())
            else:
                replaced_gen.append(w)
        reference = [[i.strip() for i in s.split()] for s in batch_extra[ind]]
        bleu = sentence_bleu(reference, replaced_gen, smoothing_function=SmoothingFunction().method2)
        B.append(bleu)
    BLEU = np.mean(B)
    return cost,BLEU

def evaluate(logger,model,config,word2index,max_len,index2word):
    dev_vec, dev_ref, dev_extra, dev_map = vectorize(config.dev_instances, config, word2index)
    #padding
    for r in dev_ref:
        while len(r)<max_len:
            r.append(0)
    dev_ref_T = np.array(dev_ref).T

    feed_dict = {model.learning_rate: 5e-4, model.encoder.input_x: dev_vec}
    feed_dict[model.encoder.dropout_keep_prob] = 1.0
    for i in range(len(model.dec_inputs)):
        feed_dict[model.dec_inputs[i]] = dev_ref_T[i]
    feed_dict[model.targets[-1]] = [0] * len(dev_vec)


    result = model.session.run(model.dec_outputs, feed_dict)
    #result = model.session.run(model.encoder.h_pool_flat,feed_dict)


    generation = np.argmax(result,axis=-1).T
    #for each generation
    string_or_bleu = False
    B = []
    for ind,r in enumerate(generation):

        gen = []
        end = False
        for i in r:
            if i == 2:
                end = True
            if not end and i != 0:
                gen.append(index2word[i])

            replaced_gen = []
            for w in gen:
                if w in dev_map[ind]:
                    replaced_gen.extend(dev_map[ind][w].split())
                else:
                    replaced_gen.append(w)

        string = ""
        if string_or_bleu:
            for w in replaced_gen:
                string += w + " "
            print(string)
        else:
            reference = [[i.strip() for i in s.split()] for s in dev_extra[ind]]
            bleu = sentence_bleu(reference, replaced_gen, smoothing_function=SmoothingFunction().method2)
            B.append(bleu)

    if not string_or_bleu:
        print("valid bleu score {}".format(np.mean(B)))
        logger.info("valid bleu score {}".format(np.mean(B)))









def run():
    #set logger
    log_file = "train.log"
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    sequence_length = 80
    embedding_size = 128
    max_epoch = 5
    pretrain = False
    save = True
    #read in data
    print("read in data")
    f = open("raw_data.pickle", "rb")
    config = pickle.load(f)
    print("done")
    #word index
    word2index = {word: ind for ind, word in enumerate(config.words)}
    index2word = {index: word for word, index in word2index.items()}
    print("compile model")
    logger.info("compile model")
    model = CNN2LSTM(sequence_length,len(config.words),embedding_size)
    if pretrain:
        ckpt = tf.train.get_checkpoint_state("train")
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            model.saver.restore(model.session, ckpt.model_checkpoint_path)
            print("model restored")
            logger.info("model restored")
    else:
        model.session.run(model.init)
    print("done")
    logger.info("done")

    print("start training")
    logger.info("start training")
    for epo in range(max_epoch):
        evaluate(logger, model, config, word2index, sequence_length, index2word)
        #1000 batch per one epo
        for bat in range(1000):
            cost,BLEU = train(model,config,32,word2index,index2word)
            if bat %200==0:
                print("train cost {}, train BLEU {}".format(cost,BLEU))
                logger.info("train cost {}, train BLEU {}".format(cost,BLEU))


        if save:
            model.saver.save(model.session, os.path.join(model.check_point_path, "model.ckpt"))



if __name__ == '__main__':
    run()





















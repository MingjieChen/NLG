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


            self.dec_outputs, self.dec_states = rnn_func(
                self.enc_inputs, self.dec_inputs, self.cell,
                self.da_dict_size, self.tree_dict_size,
                self.emb_size,
                feed_previous=True, scope=scope)
            scope.reuse_variables()
            self.outputs, self.states = rnn_func(
                self.enc_inputs,
                self.dec_inputs, self.cell,
                self.da_dict_size, self.tree_dict_size,
                self.emb_size,
                scope=scope)
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
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver(tf.global_variables())
        self.check_point_path = "train"
    def train(self,sess,feed_dict):
       output, _, cost = sess.run([self.dec_outputs,self.train_func, self.cost], feed_dict=feed_dict)
       return output, cost

    def generate(self,sess,feed_dict):
        output,cost = sess.run([self.dec_outputs,self.dec_cost],feed_dict = feed_dict)
        return output,cost

def vectorize(data):
    input_vecotors = []
    input_reference = []
    input_extra_references = []
    input_delmap = []
    for di in data:
        vec = []
        for attr in config.attr_values.keys():

            if attr in di.attr_values:
                value = di.attr_values[attr]
                vec.append(info_dict[attr])
                vec.append(info_dict[attr+"="+value])
        vec = sorted(vec)
        while len(vec)<16:
            vec.insert(0, 1)
            vec.insert(0, 0)
        input_vecotors.append(vec)


        ref = [word2index[w] for w in di.reference.split()]
        ref.insert(0,1)


        input_reference.append(ref)
        input_extra_references.append(list(di.extral_references))
        input_delmap.append(di.delexical_map)
    return input_vecotors, input_reference, input_extra_references,input_delmap


def next_batch(input_vectors, input_reference,input_extra_references ,map):
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

def evaluate(test_generations,extral_reference,del_map):
    bleu_score = 0.0
    for ind,w in enumerate(test_generations):
        gen = []
        end = False
        for i in w:
            if i == 2 :
                end = True
            if not end and i!= 0:
                gen.append(index2word[i])

        reference = [[i.strip() for i in s.split()] for s in extral_reference[ind]]
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
def single_valid(model,epo):
    B=[]
    L=[]
    previous = ""
    f = open('result.txt','w')
    for n in range(len(v_input_vecotors)):
    #for n in range(1):
        vec = v_input_vecotors[n]
        ref = v_input_reference[n]
        while len(ref) < max_len:
            ref.append(0)
        map = v_input_delmap[n]
        extra_ref = v_input_extra_references[n]
        feed_dict = {model.learning_rate:lr}
        for i in range(len(ref)):
            feed_dict[model.dec_inputs[i]] = [ref[i]]
        for i in range(len(model.enc_inputs)):
            feed_dict[model.enc_inputs[i]] = [vec[i]]
        output = model.session.run(model.dec_outputs[0:len(ref)],feed_dict=feed_dict)
        result = np.argmax(output, axis=-1).T


        gen = []
        end = False
        for i in result[0]:
            if i == 2:
                end = True
            if not end and i != 0:
                gen.append(index2word[i])
        reference = [[i.strip() for i in s.split()] for s in extra_ref]
        replaced_gen = []
        for w in gen:
            if w in map:
                replaced_gen.extend(map[w].split())
            else:
                replaced_gen.append(w)
        string  =""

        for w in replaced_gen:
            string+=w+" "
        bleu = sentence_bleu(reference, replaced_gen, smoothing_function=SmoothingFunction().method2)
        if previous!= string:
            previous = string
            #f.write(string +"\n")
            print(string)



            #f.write(str(bleu)+"\n")
            print(bleu)
        '''
        for s in reference:
            sen = ""
            for w in s:
                sen+=w+" "
            f.write(sen+"\n")
            print(sen)
        print("\n")
        f.write("\n")
        '''
        B.append(bleu)
    print("valid bleu score is {}".format(np.mean(B)))


def valid(model,epo):
    batch_vectors = []
    batch_references = []
    batch_extral_references = []
    batch_delmap = []
    B=[]
    L=[]
    for n in range(len(v_input_vecotors)):
    #for n in range(batch_size):
        if len(batch_vectors)<batch_size:
            batch_vectors.append(v_input_vecotors[n])
            batch_references.append(v_input_reference[n])
            batch_delmap.append(v_input_delmap[n])
            batch_extral_references.append(v_input_extra_references[n])
        if len(batch_vectors)==batch_size:

            for r in batch_references:

                while len(r) < max_len:
                    r.append(0)
                if len(r) != max_len:
                    print(len(r))
                    print(max_len)
                    print(r)
                    exit(0)

            batch_vectors = np.array(batch_vectors).T
            batch_references = np.array(batch_references).T

            feed_dict = {model.learning_rate: lr}
            for i in range(len(model.enc_inputs)):
                feed_dict[model.enc_inputs[i]] = batch_vectors[i]
            for i in range(len(model.dec_inputs)):
                feed_dict[model.dec_inputs[i]] = batch_references[i]
            feed_dict[model.targets[-1]] = [0] * batch_size
            output, cost = model.generate(model.session,feed_dict)


            result = np.argmax(output, axis=-1).T


            batch_vectors = []
            batch_references = []
            batch_extral_references = []
            batch_delmap = []
            B.append(bleu)
            L.append(cost)

    print("valid epoch :{},  valid loss :{}, bleu score :{}".format(epo, np.mean(L),np.mean(B)))
    logger.info("valid epoch :{},  valid loss :{}, bleu score :{}".format(epo, np.mean(L),np.mean(B)))


    logger.info("valid epoch :{},  valid loss :{}, bleu score :{}".format(epo, np.mean(L),np.mean(B)))
def expert_policy(batch_generation, extral_reference,del_map):
    batch_target = []
    for ind,w in enumerate(batch_generation):
        if ind>=1 :
            break
        gen = []
        end = False
        for i in w:
            if i == 2 :
                end = True
            if not end and i!= 0:
                gen.append(index2word[i])

        reference = [[i.strip() for i in s.split()] for s in extral_reference[ind]]
        for s in reference:
            s.insert(0,"@start@")
            s.append("@end@")
        replaced_gen = []
        for w in gen:
            if w in del_map[ind]:
                replaced_gen.extend(del_map[ind][w].split())
            else:
                replaced_gen.append(w)
        replaced_gen.insert(0,"@start@")
        available_words = set()
        for sent in extral_reference[ind]:
            for word in sent.split():
               available_words.add(word.strip())
        available_words.add("@end@")
        target = replaced_gen[:1]
        while("@end@" not in target and len(target)< 79 ):
            best_score = 0
            best_action = ""

            for action in available_words:
                temp_target = target[:]
                temp_target.append(action)

                while len(temp_target) < 5:
                    temp_target.insert(0,"@start@")



                bleu = sentence_bleu(reference, temp_target, smoothing_function=SmoothingFunction().method2,weights=(0.2,0.2,0.2,0.2,0.2))
                if bleu > best_score:
                    best_score = bleu
                    best_action = action


            target.append(best_action)

        replaced_gen.append("@end@")
        bleu = sentence_bleu(reference, target, smoothing_function=SmoothingFunction().method2)
        ref_bleu = sentence_bleu(reference, replaced_gen, smoothing_function=SmoothingFunction().method2)
        print(target)
        print(bleu)
        print(replaced_gen)
        print(ref_bleu)
        for s in reference:
            print(s)
        print("\n")

        best_bleu = bleu
        best_target = target[:]
        if ref_bleu>bleu:
            best_target = replaced_gen[:]
            best_bleu = ref_bleu

        best_target.insert(0,"@start@")
        target_string  = ""
        for wo in best_target:
            target_string+= wo+" "
        for attr,value in del_map[ind].items():
            if target_string.__contains__(value):
                target_string = target_string.replace(value,attr)
        after_best_target = target_string.strip().split()
        while(len(after_best_target)<max_len):
            after_best_target.append("@null@")
        if len(after_best_target)>max_len:
            print(len(after_best_target))
            print(after_best_target)
            print(best_target)
            print(best_bleu)
            exit(0)
        target_ind = [word2index[w] if w in word2index else 0 for w in best_target ]

        batch_target.append(target_ind)
    return batch_target
def dagger():
    for n in range(1):
        vectors, reference,map,extral_ref = next_batch(t_input_vecotors
                                            ,t_input_reference,t_input_extra_references
                                            ,t_input_delmap)
        # learned policy 's generation
        feed_dict = {model.learning_rate :lr}
        for i in range(len(model.enc_inputs)):
            feed_dict[model.enc_inputs[i]] = vectors[i]
        for i in range(len(model.dec_inputs)):
            feed_dict[model.dec_inputs[i]] = reference[i]
        feed_dict[model.targets[-1]] = [0]*batch_size
        output,cost=model.generate(model.session,feed_dict)

        result =  np.argmax(output,axis=-1).T
        bleu = evaluate(result, extral_ref, map)

        #train using dagger


        targets = expert_policy(result,extral_ref,map)
        '''
        dagger_targets = np.array(targets).T

        dagger_feed = {model.learning_rate :lr}
        for i in range(len(model.enc_inputs)):
            dagger_feed[model.enc_inputs[i]] = vectors[i]
        for i in range(len(model.dec_inputs)):
            dagger_feed[model.dec_inputs[i]] = dagger_targets[i]
        dagger_feed[model.targets[-1]] = [0]*batch_size
        dagger_output, dagger_cost = model.train(model.session, dagger_feed)
        dagger_result = np.argmax(dagger_output,axis=-1).T
        dagger_bleu = evaluate(dagger_result,extral_ref,map)



        if n % 20 == 0:
            print("learned policy bleu {}, loss {}".format(bleu,cost))
            print("dagger bleu {}, loss {}".format(dagger_bleu, dagger_cost))
            logger.info("learned policy bleu {}, loss {}".format(bleu,cost))
            logger.info("dagger bleu {}, loss {}".format(dagger_bleu, dagger_cost))
            valid(model, 0)
        '''







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
max_epoch = 1
max_len =80
lr = 1e-6
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
t_input_vecotors, t_input_reference, t_input_extra_references,t_input_delmap=vectorize(config.train_instances)
v_input_vecotors, v_input_reference, v_input_extra_references,v_input_delmap=vectorize(config.dev_instances)
if not os.path.isdir("train"):
        os.makedirs("train")
pretrain =True
just_test = False
if pretrain :
    ckpt = tf.train.get_checkpoint_state("train")
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        if just_test:

            test_model = Seq2seq(num_words)
            test_model.session.run(test_model.init)
            #test_model.saver.restore(sess, ckpt.model_checkpoint_path)



            test_model.saver.restore(test_model.session, "train/model.ckpt")
            print("model restored")
            logger.info("model restored")

            #for i in range(1):
                #valid(test_model,i)
            exit(0)

        else:
            model = Seq2seq(num_words)
            model.session.run(model.init)
            model.saver.restore(model.session, ckpt.model_checkpoint_path)
            print("model restored")
            logger.info("model restored")
    else:
        print("can not load")
        exit(0)
else:
    model = Seq2seq(num_words)
    model.session.run(model.init)
    print("init model")
    logger.info("init model")
logger.info("start training")
#valid(model,0)
single_valid(model,0)
#dagger()

'''
for epo in range(max_epoch):

    valid(model,epo)

    #model.saver.save(model.session,os.path.join(model.check_point_path, "model.ckpt"))

    dagger()

    for n in range(1000):
        batch_vectors, batch_reference,batch_map,batch_extral_ref = next_batch(t_input_vecotors
                                                                               ,t_input_reference,t_input_extra_references
                                                                               ,t_input_delmap)

        feed_dict = {model.learning_rate :lr}
        for i in range(len(model.enc_inputs)):
            feed_dict[model.enc_inputs[i]] = batch_vectors[i]
        for i in range(len(model.dec_inputs)):
            feed_dict[model.dec_inputs[i]] = batch_reference[i]
        feed_dict[model.targets[-1]] = [0]*batch_size
        output,cost=model.train(model.session,feed_dict)

        result =  np.argmax(output,axis=-1).T
        bleu = evaluate(result,batch_extral_ref,batch_map)

        if n % 200 == 0:
            print("epoch :{}, train loss :{}".format(epo, cost))
            print("bleu score :{}".format(bleu))
            logger.info("epoch :{}, train loss :{}".format(epo, cost))
            logger.info("bleu score :{}".format(bleu))
'''




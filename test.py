# coding=utf-8

from bleu import sentence_bleu, SmoothingFunction
import pickle
from data_process_E2E import Config,Data_Instance


hypothesis = ["open", "the"]
reference1 = ["open", "file"]
reference2 = ["open","the"]
#the maximum is bigram, so assign the weight into 2 half.
BLEUscore = sentence_bleu([reference1,reference2], hypothesis
                                                    ,smoothing_function=SmoothingFunction().method2
                                                    ,weights=(0.5,0.5))
f = open("raw_data.pickle","rb")
config = pickle.load(f)
for di in config.train_instances:
    gen = []
    for w in di.reference.split():
        if w in di.delexical_map:
            gen.extend(di.delexical_map[w].split())
        else:
            gen.append(w)
    print(gen)



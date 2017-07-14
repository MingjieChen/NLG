# coding=utf-8
import tensorflow as tf
import numpy as np
import random
from data_process_E2E import Config,Data_Instance
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from System_Config import system_config
sys_config = system_config()


num_family_friendly = sys_config.num_family_friendly
num_eatType = sys_config.num_eatType
num_food = sys_config.num_food
num_near = sys_config.num_near
num_priceRange = sys_config.num_priceRange
num_area = sys_config.num_area
num_customer_rating = sys_config.num_customer_rating
num_name = sys_config.num_name
readin_hidden_size = sys_config.readin_hidden_size
mlp_hidden_size = sys_config.mlp_hidden_size


class MLP_Encoder:
    ###placeholder
    def __init__(self):
        self.family_friendly_X = tf.placeholder("float", [None, num_family_friendly])
        self.eatType_X = tf.placeholder("float", [None, num_eatType])
        self.food_X = tf.placeholder("float", [None, num_food])
        self.near_X = tf.placeholder("float", [None, num_near])
        self.name_X = tf.placeholder("float", [None, num_name])
        self.priceRange_X = tf.placeholder("float", [None, num_priceRange])
        self.area_X = tf.placeholder("float", [None, num_area])
        self.customer_rating_X = tf.placeholder("float", [None, num_customer_rating])

        ### read in parameters
        self.family_friendly_W = tf.Variable(tf.random_normal([num_family_friendly, readin_hidden_size]))
        #self.family_friendly_b = tf.Variable(tf.random_normal([readin_hidden_size]))
        self.eatType_W = tf.Variable(tf.random_normal([num_eatType, readin_hidden_size]))
        #self.eatType_b = tf.Variable(tf.random_normal([readin_hidden_size]))
        self.food_W = tf.Variable(tf.random_normal([num_food, readin_hidden_size]))
        #self.food_b = tf.Variable(tf.random_normal([readin_hidden_size]))
        self.near_W = tf.Variable(tf.random_normal([num_near, readin_hidden_size]))
        #self.near_b = tf.Variable(tf.random_normal([readin_hidden_size]))
        self.name_W = tf.Variable(tf.random_normal([num_name, readin_hidden_size]))
        #self.name_b = tf.Variable(tf.random_normal([readin_hidden_size]))
        self.priceRange_W = tf.Variable(tf.random_normal([num_priceRange, readin_hidden_size]))
        #self.priceRange_b = tf.Variable(tf.random_normal([readin_hidden_size]))
        self.area_W = tf.Variable(tf.random_normal([num_area, readin_hidden_size]))
        #self.area_b = tf.Variable(tf.random_normal([readin_hidden_size]))
        self.customer_rating_W = tf.Variable(tf.random_normal([num_customer_rating, readin_hidden_size]))
        #self.customer_rating_b = tf.Variable(tf.random_normal([readin_hidden_size]))
        ### read in
        '''
        self.family_friendly_readin = tf.add(tf.matmul(self.family_friendly_X, self.family_friendly_W), self.family_friendly_b)
        self.eatType_readin = tf.add(tf.matmul(self.eatType_X, self.eatType_W), self.eatType_b)
        self.food_readin = tf.add(tf.matmul(self.food_X, self.food_W), self.food_b)
        self.near_readin = tf.add(tf.matmul(self.near_X, self.near_W), self.near_b)
        self.name_readin = tf.add(tf.matmul(self.name_X, self.name_W), self.name_b)
        self.priceRange_readin = tf.add(tf.matmul(self.priceRange_X, self.priceRange_W), self.priceRange_b)
        self.area_readin = tf.add(tf.matmul(self.area_X, self.area_W), self.area_b)
        self.customer_rating_readin = tf.add(tf.matmul(self.customer_rating_X, self.customer_rating_W), self.customer_rating_b)
        '''
        self.family_friendly_readin = tf.matmul(self.family_friendly_X, self.family_friendly_W)

        self.eatType_readin = tf.matmul(self.eatType_X, self.eatType_W)
        self.food_readin = tf.matmul(self.food_X, self.food_W)
        self.near_readin = tf.matmul(self.near_X, self.near_W)
        self.name_readin = tf.matmul(self.name_X, self.name_W)
        self.priceRange_readin = tf.matmul(self.priceRange_X, self.priceRange_W)
        self.area_readin = tf.matmul(self.area_X, self.area_W)
        self.customer_rating_readin = tf.matmul(self.customer_rating_X, self.customer_rating_W)


        ### concatenate all readin vectors
        self.con_vec = tf.concat(
            [self.family_friendly_readin, self.eatType_readin, self.food_readin, self.near_readin, self.name_readin, self.priceRange_readin
                , self.area_readin, self.customer_rating_readin], axis=-1)
        self.memory  = tf.reshape(self.con_vec,[sys_config.batch_size,sys_config.num_attributes,sys_config.readin_hidden_size])
        ### mlp parameters
        self.mlp_W = tf.Variable(tf.random_normal([8*readin_hidden_size,mlp_hidden_size]))
        self.mlp_b = tf.Variable(tf.random_normal([mlp_hidden_size]))
        ### mlp output
        self.mlp_out = tf.tanh(tf.add(tf.matmul(self.con_vec,self.mlp_W),self.mlp_b))
    def get_mlp_output(self,sess,vectors):
        feed_dict = {
            self.family_friendly_X: [vec["familyfriendly"] for vec in vectors]
            , self.eatType_X: [vec["eatType"] for vec in vectors]
            , self.food_X: [vec["food"] for vec in vectors]
            , self.near_X: [vec["near"] for vec in vectors]
            , self.name_X: [vec["name"] for vec in vectors]
            , self.area_X: [vec["area"] for vec in vectors]
            , self.priceRange_X: [vec["priceRange"] for vec in vectors]
            , self.customer_rating_X: [vec["customer_rating"] for vec in vectors]
        }

        result = sess.run(self.mlp_out,feed_dict = feed_dict)
        return result






























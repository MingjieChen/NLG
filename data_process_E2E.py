# coding=utf-8

import sys
import pickle
import re


class Config:
    def __init__(self):
        self.trainfile_path = "e2e_traindev/trainset.csv"
        self.devfile_path = "e2e_traindev/devset.csv"
        self.save_data_path = "raw_data.pickle"
        self.attributes = []
        self.attr_values = {}
        self.words = ["@null@","@start@","@end@"]
        self.train_instances = []
        self.dev_instances = []
        self.X_TOKEN = "@x@"
        self.debug = True

class Preprocess_Data:
    def process(self,file_path,config):
        num=0

        instances = []
        for line in open(file_path):
            if config.debug:
                if num >10000:
                    break
            if line.__contains__("\","):
                [left_part,right_part] = line.split("\",")
                if len(right_part.split())==1:
                    continue
                if left_part.startswith("\""):
                    left_part = left_part[1:]
                if right_part.startswith("\""):
                    right_part = right_part[1:].strip().lower()
                if right_part.endswith("\""):
                    right_part = right_part[:-1].lower().strip()
                right_part = right_part.lower().strip()
                attr_values = {}
                delexical_map = {}
                reference = right_part
                for pair in left_part.split(","):
                    if pair.__contains__("["):
                        attr = pair[0:pair.index("[")].strip().replace(" ","_").lower()
                        #if attr=="familyFriendly":
                            #attr = "family_friendly"
                        if attr not in config.attributes:
                            config.attributes.append(attr)
                        if attr not in config.attr_values:
                            config.attr_values[attr]=set()
                        value = pair[pair.index("[")+1:pair.index("]")].lower()
                        if attr=="name" or attr=="near":
                            del_value = config.X_TOKEN+attr
                            if del_value not in delexical_map:
                                delexical_map[del_value] = value
                            delexical_map[del_value] = value
                            value = del_value
                        if attr not in attr_values:
                            attr_values[attr] = value
                        attr_values[attr] = value
                        config.attr_values[attr].add(value)
                if reference.__contains__("fitzbilies"):
                    reference = reference.replace("fitzbilies","fitzbillies")
                if reference.__contains__("fitzbilles"):
                    reference = reference.replace("fitzbilles","fitzbillies")
                if reference.__contains__("0f"):
                    reference = reference.replace("0f", "of")



                for del_value in delexical_map:
                    if reference.__contains__(delexical_map[del_value]):
                        reference = reference.replace(str(delexical_map[del_value]),del_value)
                #if reference.__contains__("@X@names"):
                    #reference = reference.replace("@X@names","@X@name s")
                reference = reference.replace(".", " . ")
                reference = reference.replace(",", " , ")
                reference = reference.replace("\"", " \" ")
                reference = reference.replace("'"," ' ")
                reference = reference.replace(";", " ; ")
                reference = reference.replace(":", " : ")
                reference = reference.replace("!", " ! ")
                reference = reference.replace("?", " ? ")
                reference = reference.replace("-", " - ")
                reference = reference.strip()
                reference = reference.replace("  "," ")
                reference = reference.replace(", ,"," , ")
                reference = reference.replace(". ."," . ")
                if not reference.endswith("."):
                    reference+=" ."

                reference = reference+" @end@"


                #if not reference.__contains__("@x@"):
                    #print(reference)
                for word in reference.split():
                    #if re.match("[,.':;?\"]",word):
                        #if word!= "." and word!="," and word!="'":
                           #print(word)


                    if word.__contains__("fitz"):
                        word = "fitzbillies"

                    if word == "5@x@name":
                        word = "@x@name"
                        reference = reference.replace("5@x@name","5 @x@name")

                    if word.__contains__("@x@name") and word!="@x@name" :
                        reference = reference.replace(word,"@x@name")
                        real_value = word.replace("@x@name",delexical_map["@x@name"])
                        word = "@x@name"
                        delexical_map[word] = real_value


                    if word.__contains__("@x@near") and word!="@x@near" :
                        reference = reference.replace(word,"@x@near")
                        real_value = word.replace("@x@near", delexical_map["@x@near"])
                        word = "@x@near"
                        delexical_map[word] = real_value

                    m1 = re.match(r"([0-9]+)([a-z]+)([0-9]+)", word)
                    m2 = re.match(r"([a-z]+)([0-9]+)",word)
                    m3 = re.match(r"([a-z]+)(£[0-9]+)",word)
                    m4 = re.match(r"(£)([a-z]+)",word)
                    m5 = re.match(r"([0-9]+)([a-z]+)",word)
                    if m1:

                        replace_word = ""
                        for w in m1.groups():
                            w = w.strip()
                            replace_word += " " + w
                            if w not in config.words:
                                config.words.append(w)


                        reference=reference.replace(word,replace_word)

                    elif m2:

                        replace_word = ""
                        for w in m2.groups():
                            w = w.strip()
                            replace_word += " " + w
                            if w not in config.words:
                                config.words.append(w)
                        reference = reference.replace(word, replace_word)

                    elif m3:
                        replace_word = ""
                        for w in m3.groups():
                            w = w.strip()
                            replace_word += " " + w
                            if w not in config.words:
                                config.words.append(w)
                        reference = reference.replace(word, replace_word)
                    elif m4:

                        replace_word = ""
                        for w in m4.groups():
                            w = w.strip()
                            replace_word += " " + w
                            if w not in config.words:
                                config.words.append(w)
                        reference = reference.replace(word, replace_word)
                    elif m5 :

                        replace_word = ""
                        for w in m5.groups():
                            w = w.strip()
                            replace_word += " " + w
                            if w not in config.words:
                                config.words.append(w)
                        reference = reference.replace(word, replace_word)





                    else:
                        word =word.strip()
                        if word not in config.words:
                            config.words.append(word)
                reference = reference.replace("  "," ")
                di = Data_Instance(attr_values,reference,delexical_map)
                di.extral_references.add(di.post_process_ref())
                '''
                for extral_di in instances:
                    if di.abstract_mr()== extral_di.abstract_mr():

                        extral_di.extral_references.add(di.post_process_ref())
                        di.extral_references.union(extral_di.extral_references)
                        #di.extral_references.add(di.reference)
                        #di.extral_references.add(extral_di.reference)
                        extral_di.extral_references.union(di.extral_references)

                #print(num)
                '''


                instances.append(di)
                num+=1


        return instances

class Data_Instance:
    def __init__(self,attr_values,reference,delexical_map):
        self.attr_values = attr_values
        self.reference = reference
        self.delexical_map = delexical_map
        self.extral_references = set()
    def abstract_mr(self):
        abstract_string = ""
        for attr in self.attr_values.keys():
            value = self.attr_values[attr]
            if value in self.delexical_map:
                value = self.delexical_map[value]
            abstract_string+= attr+"="+value+" ,"
        return abstract_string
    def post_process_ref(self):
        ref =self.reference
        for del_value in self.delexical_map:

            value = self.delexical_map[del_value]

            if self.reference.__contains__(del_value):

                ref = ref.replace(del_value,value)
        if ref.strip().endswith("@end@"):
            ref = ref.replace("@end@","").strip()
        if ref.__contains__("@x@"):
            print(ref)
            exit(0)
        return ref






if __name__=='__main__':
    config = Config()
    train = Preprocess_Data()
    dev = Preprocess_Data()
    config.train_instances = train.process(config.trainfile_path,config)
    #config.dev_instances =dev.process(config.devfile_path,config)






    #f = open(config.save_data_path, "wb")
    #pickle.dump(config, f)
    #f.close()




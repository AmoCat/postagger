#!/usr/bin/env python
#coding:utf8
from dataset import read_dataset
import collections
from collections import Counter
import numpy as np
import os
import cPickle as pkl

class Data_processor(object):
    def __init__(self, embedding_size, embedding_name, mode = "train_"):
        self.embedding_size = embedding_size
        self.embedding_name = embedding_name
        if mode == "train_":
            self.words, self.labels = self.read_sentences()
            self.dic = self.word_frequency_dic()
            self.preprocess(self.dic) #首尾加<B><E>,替换UNK
            self.embedding_nums = self.word_embedding()
            self.convert_words_to_int(embedding_name, self.embedding_nums)
            #os.system("./word2vec -train fsentences.txt -output word_embedding -size 100")
            self.label_nums = self.hash_label(self.labels)  #得到标签数量
            print "embedding_nums: ",self.embedding_nums,", label_nums: ",self.label_nums
        elif mode == "test_":
            pass

    def read_sentences(self, out_filename='sentences.txt', mode = "train_"):
        "words: a dict of words"
        "labels: a dict of labels"
        "read data and convert it to sentences in ./sentences.txt"
        f_name = "penn." + mode.rstrip('_') + ".pos.gz"
        train_dataset = read_dataset(f_name)
        out = open(mode + out_filename,'w')
        words = ' '.join([' '.join(tup[0]) for tup in train_dataset]).split()
        labels = ' '.join([' '.join(tup[1]) for tup in train_dataset]).split()
        out.write('\n'.join([' '.join(tup[0]) for tup in train_dataset]))
        out.close()
        return words,labels

    def word_frequency_dic(self, mode = "train_"):
        "get a dict of word frequencies"
        c = Counter()
        f_name = "penn." + mode.rstrip('_') + ".pos.gz"
        train_dataset = read_dataset(f_name)
        #统计每个词出现的次数
        for t in train_dataset:
            for word in t[0]:
                c[word] += 1
        return c

    def preprocess(self, dic,in_filename = 'sentences.txt', out_filename = 'fsentences.txt', mode = "train_"):
        "Replace the low-frequency word(<5) as \'UNK\'"
        "add \'<B>\' at the beginning of sentences,\'<>\' at the end of sentences"
        "write the processed sentences into ./fsentences.txt"
        with open(mode + in_filename) as sentences,open(mode + out_filename,'w') as out:
            for line in sentences:
                words = line.split()
                s = "<B> "
                for word in words:
                    if dic[word] < 5:
                        s += 'UNK '
                    else:
                        s += word + ' '
                s += '<E> '
                s.rstrip()
                out.write(s+'\n')

    def word_embedding(self, sentences = "fsentences.txt", output = "word_embedding", size = "50", mode = "train_"):
        '''
        get word embedding
        if no error , return nums of embeddings
        '''
        res = os.system("./word2vec -train " + mode + sentences + " -output origin_embedding -size " + size)
        with open("origin_embedding", 'r') as f:
            data = f.read().strip().split('\n')
            first_line = data[0].split()
        with open(mode + output, 'w') as f:
            for line in data[1:]:
                f.write(line + '\n')
        return int(first_line[0])

    '''
    def convert_words_to_int(labels,vocabulary_size):
        count = [['UNK', -1]]
        count.extend(collections.Counter(labels).most_common())
        dictionary = dict()
        for label, _ in count:
        dictionary[label] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
        return count,dictionary,reverse_dictionary
    '''

    def convert_words_to_int(self, embedding_name, nums, size = 50, mode = "train_"):
        dictionary = {} #dict(word, int)
        embedding = np.ndarray([nums,size],np.float32 ) #dict(word, embedding)
        with open(mode + embedding_name) as f:
            for line in f:
                words = line.split()
                dictionary[words[0]] = len(dictionary)
                for i in range(size):
                    embedding[len(dictionary)-1][i] = np.float32(words[i+1])
            reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
            pkl.dump(embedding, open(mode + 'embedding_dict', 'w'))
            pkl.dump(dictionary, open(mode + 'word_to_int_dict', 'w'))
            pkl.dump(reverse_dictionary, open(mode + 'int_to_word_dict', 'w'))

    def hash_label(self, labels, mode = "train_"):
        dictionary = dict()
        count = list()
        count.extend(collections.Counter(labels).most_common())
        for label, _ in count:
            dictionary[label] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        pkl.dump(dictionary, open(mode + 'lab_to_int_dic', 'w'))
        pkl.dump(reverse_dictionary, open(mode + 'int_to_lab_dic', 'w'))
        return len(count)

    def gen_label_vector(self, label_size, label, lab_to_int_name = 'lab_to_int_dict', mode = "train_"):
        vector = np.zeros([label_size], dtype = np.float32)
        lab_to_int = pkl.load(open(mode + lab_to_int_name, 'r'))
        index = lab_to_int[label]
        vector[index] = 1.0
        return vector

if __name__ == '__main__':
    data_processor = Data_processor(50, "word_embedding", "train")


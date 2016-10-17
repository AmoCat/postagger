#!/usr/bin/env python
#coding:utf8
from dataset import read_dataset
from collections import Counter

def read_data(out_filename='./sentences.txt'):
    train_dataset = read_dataset('./penn.train.pos.gz')
    out = open(out_filename,'w')
    out.write('\n'.join([' '.join(tup[0]) for tup in train_dataset]))
    out.close()

def word_frequency_dic():
    c = Counter()
    train_dataset = read_dataset('./penn.train.pos.gz')
    #统计每个词出现的次数
    for t in train_dataset:
        for word in t[0]:
            c[word] += 1
    return c

def word_frequency(dic,in_filename = './sentences.txt',out_filename = './fsentences.txt'):
    with open(in_filename) as sentences,open(out_filename,'w') as out:
        for line in sentences:
            words = line.split()
            s = "<B> "
            for word in words:
                if dic[word]<5:
                    s += 'UNK '
                else:
                    s += word+' '
            s += '<E> '
            s.rstrip()
            out.write(s+'\n')

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

if __name__ == '__main__':
    read_dataset()
    dic = word_frequency_dic()
    word_frequency(dic)

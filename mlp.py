#coding:utf-8
import tensorflow as tf
import numpy as np
from dataset import read_dataset
import cPickle as pkl

class MLP(object):
    def __init__(self, embedding_size, label_size):
        self.learning_rate = 0.001
        self.embedding_size = embedding_size
        self.label_size = label_size
        self.n_input = 3 * self.embedding_size
        self.n_hidden_1 = 20
        self.n_hidden_2 = 17
        self.words, self.labs = self.get_data()
        self.nums_data = len(self.words)
        self.num_exapmles = 12000
        self.batch_size = 100

    def multilayer_perception(self, embedding_size, label_size):
        x = tf.placeholder("float", [None, embedding_size])
        y_ = tf.placeholder("float", [None, label_size])

        weights = {
            "h1" : tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            "h2" : tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            "out" : tf.Variable(tf.random_normal([self.n_hidden_2, self.label_size]))
        }

        biases = {
            "b1" : tf.Variable(tf.random_normal([self.n_hidden_1])),
            "b2" : tf.Variable(tf.random_normal([self.n_hidden_2])),
            "out" : tf.Variable(tf.random_normal([self.label_size]))
        }

        pred = self.mlp_core(x, weights, biases)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_))
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)

        init = tf.initialize_all_variables()
        
        correct_prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        with tf.Session() as sess:
            sess.run(init)
            for i in range(120):
                batch_x, batch_y = self.next_batch(self.batch_size)
                optimizer.run(feed_dict = {x:batch_x, y_:batch_y})
                train_accuracy = accuracy.eval(feed_dict = {x:batch_x, y_:batch_y})
                print "step %d, training accuracy %g" % (i+1, train_accuracy)


    def mlp_core(self, x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
        layer_2 = tf.nn.relu(layer_2)

        out_layer = tf.matmul(layer_2, weights["out"]) + biases["out"]
        return out_layer
    


    def get_data(self):
        train_data_set = read_dataset("./penn.train.pos.gz")
        word_to_int = pkl.load(open("word_to_int_dict", 'r'))
        lab_to_int = pkl.load(open("lab_to_int_dic", 'r'))
        int_words_list = []
        int_labels_list = []
        for t in train_data_set:
            words = t[0]
            labels = t[1]
            int_words = []
            for w in words:
                if word_to_int.has_key(w):
                    int_words.append(word_to_int[w])
                else:
                    int_words.append(word_to_int['UNK'])
            int_labels = [lab_to_int[l] for l in labels]
            int_words_list.append(int_words)
            int_labels_list.append(int_labels)
        print int_words_list[0],int_labels_list[0]
        return int_words_list,int_labels_list

if __name__ == "__main__":
    batch_size = 100
    mlp = MLP(50, 45)
    words, labs = mlp.get_data(batch_size)

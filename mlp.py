#coding:utf-8
import tensorflow as tf
import numpy as np
from dataset import read_dataset
import cPickle as pkl

class Batch(object):
    def gen_window(self, int_words_list, int_labels_list):
        embedding = pkl.load(open("embedding_dict", 'r')).tolist()
        word_to_int = pkl.load(open("word_to_int_dict", 'r'))
        int_start = word_to_int['<B>']
        int_end = word_to_int['<E>']
        the_int = word_to_int['the']
        words_windows = []
        labels_windows = []
        for sentence in int_words_list:
            for i in range(0,len(sentence)):
                window = []
                window.extend(embedding[i-1] if i > 0 else embedding[int_start])
                window.extend(embedding[i])
                window.extend(embedding[i+1] if i < len(sentence) - 1 else embedding[int_end])
                words_windows.append(window)

        for sentence in int_labels_list:
            for index in sentence[0:]:
                window = []
                for id in range(45):
                    window.append(0.0)
                window[i] = 1.0
                labels_windows.append(window)

        x = np.array(words_windows, np.float32)
        y_ = np.array(labels_windows, np.float32)
        l = len(x)
        perm = np.arange(l)
        np.random.shuffle(perm)
        x = x[perm]
        y_ = y_[perm]
        return x, y_, l

    def get_data(self):
        '''
        获得两个列表：
        int_words_list:包含每个句子的词对应int的列表
        int_labels_list:包含每个句子的label对应的int的列表
        例：
        int_words_list[0]:
        [57, 37, 306, 1098, 1771, 7, 16, 18, 1, 19, 27, 544,\
        12, 9135, 10629, 82, 16, 1, 1, 7651, 3, 1, 11, 1, 784,\
        2, 19, 1, 90, 8160, 80, 2, 3, 875, 7, 1, 2, 1932, 24,\
         8595, 1, 2, 31, 10310, 1530, 8, 1, 1, 6]
        int_labels_list[1]:
        [1, 3, 2, 8, 0, 1, 23, 3, 0, 24, 1, 2, 20, 2, 2, 35, 23,\
         15, 5, 18, 3, 0, 1, 2, 2, 6, 24, 0, 12, 5, 34, 6, 3, 0,\
          1, 2, 6, 15, 1, 2, 2, 6, 10, 9, 15, 13, 2, 2, 7]
        '''
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
        return int_words_list,int_labels_list

class MLP(object):
    def __init__(self, embedding_size, label_size):
        self.num_exapmles = 39832
        self.batch_size = 100
        self._index_in_epoch = 0
        self.learning_rate = 0.001
        self.embedding_size = embedding_size
        self.window_size = 3 * self.embedding_size
        self.label_size = label_size
        self.n_input = self.window_size
        self.n_hidden_1 = 20
        self.n_hidden_2 = 17
        self.words, self.labs = self.get_data()
        self.nums_data = len(self.words)
        self.x, self.y_ , self.window_nums = self.gen_window(self.words, self.labs)
        print "nums of windows: ", self.window_nums
        self.multilayer_perception(self.embedding_size, self.label_size)

    def multilayer_perception(self, embedding_size, label_size):
        x = tf.placeholder("float", [None, self.window_size])
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
            total_batch = self.window_nums/self.window_size
            for i in range(total_batch):
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
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.window_nums:
            end = self.window_nums - 1
        else:
            end = self._index_in_epoch
        return self.x[start:end],self.y_[start:end]


    def gen_window(self, int_words_list, int_labels_list):
        embedding = pkl.load(open("embedding_dict", 'r')).tolist()
        word_to_int = pkl.load(open("word_to_int_dict", 'r'))
        int_start = word_to_int['<B>']
        int_end = word_to_int['<E>']
        the_int = word_to_int['the']
        words_windows = []
        labels_windows = []
        for sentence in int_words_list:
            for i in range(0,len(sentence)):
                window = []
                window.extend(embedding[i-1] if i > 0 else embedding[int_start])
                window.extend(embedding[i])
                window.extend(embedding[i+1] if i < len(sentence) - 1 else embedding[int_end])
                words_windows.append(window)

        for sentence in int_labels_list:
            for index in sentence[0:]:
                window = []
                for id in range(45):
                    window.append(0.0)
                window[i] = 1.0
                labels_windows.append(window)

        x = np.array(words_windows, np.float32)
        y_ = np.array(labels_windows, np.float32)
        l = len(x)
        perm = np.arange(l)
        np.random.shuffle(perm)
        x = x[perm]
        y_ = y_[perm]
        return x, y_, l

    def get_data(self):
        '''
        获得两个列表：
        int_words_list:包含每个句子的词对应int的列表
        int_labels_list:包含每个句子的label对应的int的列表
        例：
        int_words_list[0]:
        [57, 37, 306, 1098, 1771, 7, 16, 18, 1, 19, 27, 544,\
        12, 9135, 10629, 82, 16, 1, 1, 7651, 3, 1, 11, 1, 784,\
        2, 19, 1, 90, 8160, 80, 2, 3, 875, 7, 1, 2, 1932, 24,\
         8595, 1, 2, 31, 10310, 1530, 8, 1, 1, 6]
        int_labels_list[1]:
        [1, 3, 2, 8, 0, 1, 23, 3, 0, 24, 1, 2, 20, 2, 2, 35, 23,\
         15, 5, 18, 3, 0, 1, 2, 2, 6, 24, 0, 12, 5, 34, 6, 3, 0,\
          1, 2, 6, 15, 1, 2, 2, 6, 10, 9, 15, 13, 2, 2, 7]
        '''
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
        return int_words_list,int_labels_list

if __name__ == "__main__":
    batch_size = 100
    mlp = MLP(50, 45)

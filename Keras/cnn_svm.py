
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from NLP_part.word_embedding.util import TextLoader
from NLP_part.word_embedding.get_word2vector import getword2vec_to_network



#config of model
label_size = 2
seq_length = 150

num_epochs = 50
learning_rate = 0.001
decay_rate = 0.9
batch_size = 4








class Model():
    def __init__(self,word_embedding):
        self.word_embedding = word_embedding
        tf.reset_default_graph()
        # 初始化偏置向量
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # 二维卷积运算，步长为1，输出大小不变
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        # 池化运算，将卷积特征缩小为1/2
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 归一化操作
        def norm(name, l_input, lsize=4):
            return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

        # 初始化权值向量
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        self.input_data = tf.placeholder(tf.int32, [None, seq_length], name='inputs')
        self.targets = tf.placeholder(tf.float64, [None,2])  # target is class label

        # 计算输出，采用的函数是softmax（输入的时候是one hot编码）
        # y = tf.nn.softmax(tf.matmul(self.input_data, W) + b)
        self.data = tf.nn.embedding_lookup(self.word_embedding, self.input_data)
        data = tf.expand_dims(self.data, 3)
        print('input_data shape is :',data.get_shape())
        input_data = tf.reshape(data, [-1, 30, 25, 1])
        # 第一个卷积层，5x5的卷积核，输出向量是32维
        w_conv1 = weight_variable([11, 11, 1, 128])
        b_conv1 = bias_variable([128])
        with tf.variable_scope('CNN_First_layer'):
            h_conv1 = tf.nn.relu(conv2d(input_data, w_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
        # 第二个卷积层，5x5的卷积核，输出向量是32维
        w_conv2 = weight_variable([11, 11, 128, 256])
        b_conv2 = bias_variable([256])
        with tf.variable_scope('CNN_Second_layer'):
            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
        # 第三个卷积层，5x5的卷积核，输出向量是32维
        w_conv3 = weight_variable([11, 11, 256, 32])
        b_conv3 = bias_variable([32])
        with tf.variable_scope('CNN_Third_layer'):
            h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)
        print('h_pool3 shape is :', h_pool3.get_shape())

        # 全连接层的w和b
        w_fc1 = weight_variable([4 * 4 * 32, 256])
        b_fc1 = bias_variable([256])
        # 此时输出的维数是256维
        h_pool2_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 32])
        with tf.variable_scope('Fully_connect_layer'):
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        self.h_fc1 = h_fc1
        # h_fc1是提取出的256维特征，很关键。后面就是用这个输入到SVM中

        # 设置dropout，否则很容易过拟合
        self.keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # 输出层，在本实验中只利用它的输出反向训练CNN，至于其具体数值我不关心
        w_fc2 = weight_variable([256, 2])
        b_fc2 = bias_variable([2])
        with tf.variable_scope('Fully_connect_layer'):
            self.prob = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
            self.y_conv = tf.nn.softmax(self.prob,name='prob')
            print('self.prob shape :',self.prob.get_shape())
            print(self.prob)
        with tf.variable_scope('cost'):
            # aa = tf.cast(self.targets,tf.argmax(self.prob,1),tf.int64)
            # self.cost = tf.nn.l2_loss(aa)
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets,logits=tf.argmax(self.prob,1)))
        tf.summary.scalar('loss', self.cost)

        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)  # Adam Optimizer
        print('self.y_conv shape is :',self.y_conv.get_shape())
        self.correct_pred = tf.equal(tf.argmax(self.y_conv, 1), self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()




def train():
    word_embedding = getword2vec_to_network()
    model = Model(word_embedding)
    saver = tf.train.Saver()
    loader = TextLoader(batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./tf_log/', sess.graph)
        print('start loop .......')
        for e in range(num_epochs):
            sess.run(tf.assign(model.lr, learning_rate * (decay_rate ** e)))
            for b in range(loader.num_batches):
                x, y = loader.next_batch(b)
                # print('trouble shooting tag')
                feed = {model.input_data: x, model.targets: y,model.keep_prob:0.7}
                train_loss, _, accuracy,summary_, h_fc1 = sess.run([model.cost, model.optimizer, model.accuracy,model.merged,model.h_fc1], feed_dict=feed)
                writer.add_summary(summary_, global_step=b)
                print('第{}批次，第{}次循环时,train_loss ={},accuracy ={}'.format(e,b,train_loss,accuracy))
                if e == num_epochs-1 and b == loader.num_batches-1:
                    saver.save(sess,'model/liver.model')


if __name__ == '__main__':
    train()
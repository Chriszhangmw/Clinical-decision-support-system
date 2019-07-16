


import numpy as np
import csv
import re
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils import TextLoader_test

numSamples = 110
maxVacubNum = 200

def get_test():
    data_loader = TextLoader_test('./input_200_test.csv', 1, 200, 'samples_vocabs.pkl', encoding='utf8')

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('model_2/gan/gan.model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('model_2/gan/'))
        inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
        y = tf.get_default_graph().get_tensor_by_name('softmaxLayer/y:0')
        for k in range(data_loader.num_batches):
            x,_ = data_loader.next_batch(k)
            result = sess.run(y,{inputs:x})
            print(result)


if __name__ == "__main__":
    input_dataSet = np.load('idsMatrix_200_test.npy')
    k = 0
    for i in input_dataSet:
        if i[-1] == 1:
            k+=1
    print(k)
    get_test()











import re
import pickle as pickle
import gensim
import jieba.posseg
import  numpy as np
from hy_config import  train_config
from sentence_cut import cut_non
from sklearn.externals import joblib
import tensorflow as tf
import numpy as np
import os
import threading
import time
import datetime
import  _thread

retrive=('n','v','l','ng','nl','nz','f','vn','a','i','vg','tg','t')
stopwords = [line.rstrip() for line in open('D:/Eclipse_workplace/web_tensorflow/stopwords.txt','r',encoding = 'utf-8')]
maxVacubNum = 150
model = gensim.models.KeyedVectors.load_word2vec_format("D:/Eclipse_workplace/web_tensorflow/books.model.bin", binary=True)


config = train_config()
path2 = 'D:/Eclipse_workplace/web_tensorflow/model_14/vocab.pkl' # you shuold change this
path3 = 'D:/Eclipse_workplace/web_tensorflow/samples/samples.csv'# you shuold change this

def read_dictionary(vocab_path):
    '''
    get the dictionary for word to index
    :param vocab_path: the pkl file path
    :return: the dictionary
    '''
    # vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def transform(vocab, d):
    '''
    transform the chinese character into index
    :param vocab: corpus
    :param d: senteces
    :return: indexs
    '''
    d = list(d)
    new_d = []
    for word in d:
        if word in vocab.keys():
            new_d.append(vocab[word])
        else:
            new_d.append(0)
    if len(new_d) >= 700:
        new_d = new_d[:700]
    else:
        new_d = new_d + [0] * (700 - len(new_d))
    return new_d


class Wordlist(object):
    def __init__(self, filename):
        lines = [x.split() for x in open(filename, 'r', encoding='utf-8').readlines()]
        self.size = len(lines)
        self.voc = [(item[0][0], item[1]) for item in zip(lines, range(self.size))]
        self.voc = dict(self.voc)

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return 0

def load_vec(vocab,model):
    word_vecs = {}
    D = 50
    for i in vocab:
        try:
            word_vecs[i] = model[i]
        except:
            word_vecs[i]=np.random.uniform(-0.0005,0.0005,D)
    return word_vecs


def get_W(word_vecs, D):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, D), dtype='float32')
    W[0] = np.random.uniform(-0.0005, 0.0005, D)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def cut_line(string):
    string=','.join(re.findall(u'[\u4e00-\u9fff]+', string))
    scut=jieba.posseg.cut(string)
    wordSum=[]
    oneLine=[]
    for word in scut:
        wordSum.append((word.word,word.flag))
    for cuttedword in wordSum:
        if cuttedword[1] in retrive[0:13]:
            tmp=cuttedword[0]+''
            #print(a)
            oneLine.append(tmp)
    return oneLine

vocab = Wordlist('D:/Eclipse_workplace/web_tensorflow/books0416.txt')
w2v = load_vec(vocab.voc, model)
_, word_idx_map = get_W(w2v, 50)

def getInputId(inputsentence):

    ids = np.zeros((1, maxVacubNum))
    one_sample_list = cut_line(inputsentence)
    for i in range(len(one_sample_list)):
        try:
            if one_sample_list[i] not in stopwords:
                ids[0][i] = word_idx_map[one_sample_list[i]]
        except:
            pass
        if i > maxVacubNum:
            break
    return ids.astype(int)
tag_List = ['appendix', 'cancer', 'obstruction', 'inflammation', 'liver', 'pancrease', 'perforation',
                'stomach', 'Bleeding', 'colon', 'gallbladerr', 'hernia', 'intestine', 'peritoneum']



def record_sample(inputsentence):
    '''
    save the samples into a csv file
    :param inputsentence: all the data from web side
    :return:
    '''
    with open(path3,'a',encoding='utf-8') as f:
        f.write(inputsentence + '\n')
        f.close()
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools import inspect_checkpoint
def predict_one(input_sentence):
    '''
    restore the 14 calssfiers and start the prediction , now this process takes too much time
    :param input_sentence: from web side inputs
    :return: results
    '''
    result = {}
    a = datetime.datetime.now()
    print('start:', a)
    ids = np.zeros((1,700))
    vocab = read_dictionary(path2)
    ids[0][:700] = transform(vocab, input_sentence)
    ids = ids.astype(int)
    tag_List = ['appendix', 'cancer', 'obstruction', 'inflammation', 'liver', 'pancrease', 'perforation',
                'stomach', 'Bleeding', 'colon', 'gallbladerr', 'hernia', 'intestine','peritoneum']
    g1 = tf.Graph()
    g2 = tf.Graph()
    g3 = tf.Graph()
    g4 = tf.Graph()
    g5 = tf.Graph()
    g6 = tf.Graph()
    g7 = tf.Graph()
    g8 = tf.Graph()
    g9 = tf.Graph()
    g10 = tf.Graph()
    g11 = tf.Graph()
    g12 = tf.Graph()
    g13 = tf.Graph()
    g14 = tf.Graph()
    for i in range(len(tag_List)):
        tag = tag_List[i]
        if i == 0:
            curr_sees = tf.Session(graph = g1)
            curr_g = g1
        elif i == 1:
            curr_sees = tf.Session(graph= g2)
            curr_g = g2
        elif i == 2:
            curr_sees = tf.Session(graph= g3)
            curr_g = g3
        elif i == 3:
            curr_sees = tf.Session(graph= g4)
            curr_g = g4
        elif i == 4:
            curr_sees = tf.Session(graph= g5)
            curr_g = g5
        elif i == 5:
            curr_sees = tf.Session(graph= g6)
            curr_g = g6
        elif i == 6:
            curr_sees = tf.Session(graph= g7)
            curr_g = g7
        elif i == 7:
            curr_sees = tf.Session(graph= g8)
            curr_g = g8
        elif i == 8:
            curr_sees = tf.Session(graph= g9)
            curr_g = g9
        elif i == 9:
            curr_sees = tf.Session(graph= g10)
            curr_g = g10
        elif i == 10:
            curr_sees = tf.Session(graph= g11)
            curr_g = g11
        elif i == 11:
            curr_sees = tf.Session(graph= g12)
            curr_g = g12
        elif i == 12:
            curr_sees = tf.Session(graph= g13)
            curr_g = g13
        elif i == 13:
            curr_sees = tf.Session(graph= g14)
            curr_g = g14
        with curr_sees.as_default():
            with curr_g.as_default():
                start_time = time.time()
                # print('开始加载时间：', start_time)
                curr_sees.run(tf.global_variables_initializer())
                with tf.gfile.GFile('D:/Eclipse_workplace/web_tensorflow/checkpoints/' + tag + "_frozen.pb", "rb") as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name='',
                    op_dict=None,
                    producer_op_list=None
                )

                inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
                y = tf.get_default_graph().get_tensor_by_name('softmaxLayer/y:0')
                print("restoring time:", time.time() - start_time)
    #             # print('calculating time：', start_time)
                start_time1 = time.time()
                prediction = y.eval(feed_dict={inputs: ids})
                print("calculating time:", time.time() - start_time1)
                max_probability_index = np.argmax(prediction)
                if max_probability_index == 1:
                    probability = prediction[0][1]
                    probability = ("%.2f" % float(probability))
                    print('According to the System advice,{} probability is {}'.format(tag , probability))
                    result[tag] = probability

                # prediction = sess.run(y, {inputs: ids})
                prediction = np.argmax(prediction, 1)  # this is the results
                # print('Accoridng to the System advice,{} label is {}'.format(tag , prediction))
                curr_sees.close()
    print(result)
    current_time = datetime.datetime.now()
    print("耗时： {}".format((current_time - a).seconds))
    result_ = {'cancer': '1.00', 'inflammation': '0.98', 'stomach': '1.00', 'hernia': '1.00'} #for test
    return  result


def freeze_graph(model_path):
    model_path = model_path
    base_dir='D:/Eclipse_workplace/web_tensorflow/model_14/'
    tf.reset_default_graph()
    checkpoint_path = tf.train.latest_checkpoint(base_dir + '/' + model_path)
    saver = tf.train.import_meta_graph(checkpoint_path + '.meta', import_scope=None)
    output_node_names = "softmaxLayer/y"
    with tf.Session() as sess:
        # Restore the variable values
        saver.restore(sess, checkpoint_path)
        # Get the graph def from our current graph
        graph_def = tf.get_default_graph().as_graph_def()
        # Turn all variables into constants
        frozen_graph_def = convert_variables_to_constants(sess, graph_def, output_node_names.split(','))
        # Save our new graph def
        with tf.gfile.GFile('D:/Eclipse_workplace/web_tensorflow/checkpoints/' + model_path + '_frozen.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

def cut_sentence(inputsentence):
    '''
    :param inputsentence: get from web side inputs
    :return: the split chinese sentence results
    '''
    words_list = cut_non(inputsentence)
    # print(words_list)
    return  words_list

def first_action(inputsentence):
    '''
    from the flask framework
    :param inputsentence:
    :return:
    '''
    data= {}
    result1 = cut_sentence(inputsentence)
    result2 = predict_one(inputsentence)
    data['cut'] = result1
    data['predict'] = result2

    return data







if __name__ == '__main__':
    # tag_List = ['appendix', 'cancer', 'obstruction', 'inflammation', 'liver', 'pancrease', 'perforation',
    #             'stomach', 'Bleeding', 'colon', 'gallbladerr', 'hernia', 'intestine', 'peritoneum']
    # for tag in tag_List:
    #     freeze_graph(tag)



    sentenceis = '反复腹痛腹胀半年加重10+天,17年前因患胃溃疡行了胃大部切除术1+年前因外伤于我' \
                 '院行脾破裂切除术否认肝炎。结核史等传染性疾病史；否认高血压。糖尿病。心脏病等' \
                 '慢性病史；有输血史；否认药物过敏史；预防接种史不详,半余前患者常于不当饮食后出' \
                 '现腹痛腹胀伴烧心。恶心。呕吐。嗳气等症状呕吐物为少许酸味胃内容物上述症状复发患' \
                 '者未引起重视10+天前患者感腹痛腹胀加重遂于当地医院输液治疗至今未见好转遂来我院' \
                 '就诊门诊以不全性肠梗阻收入我科患者本次发病以来进少量流质饮食小便正常大便不成形自' \
                 '患病以来体重下降5kg'
    ids = np.zeros((1, 700))
    vocab = read_dictionary(path2)
    ids[0][:700] = transform(vocab, sentenceis)
    ids = ids.astype(int)
    predict_one(sentenceis)

    # with open('./1046.csv','r',encoding='utf-8') as f_w:
    #     data = f_w.readlines()
    #
    # with open('./aad.csv','w',encoding='utf-8') as f:
    #     for line in data:
    #         line = line.split(',')
    #         number = line[0].strip()
    #         result = line[6].strip()
    #         localtion = 'UNK'
    #         describe = str(cut_sentence(line[3].strip()+line[4].strip()+line[5].strip()))
    #         one_sample = number + ',' + result + ',' + localtion + ',' + describe + '\n'
    #         f.write(one_sample)
    # f.close()





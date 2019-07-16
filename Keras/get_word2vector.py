
import gensim
import numpy as np
np.set_printoptions(suppress=True)
import re
import csv
import jieba.posseg
from  NLP_part.word_embedding.deal_body_part_label import read_label
from sklearn.model_selection import train_test_split
import pandas as pd

#
# model = gensim.models.KeyedVectors.load_word2vec_format('./corpus/corpus.model.bin', binary=True)
# print(model['发热'])


numSamples = 1046
Chief_complaint_num = 16
Current_medical_history_num = 90
Past_history_num = 90

stopwords = [line.rstrip() for line in open('E:\\PythonProjects\\new_aad\\NLP_part\\word_embedding\\corpus\\stopwords.txt', 'r', encoding='utf-8')]

def cut_line(string):
    string = ','.join(re.findall(u'[\u4e00-\u9fff]+', string))
    scut = jieba.posseg.cut(string)
    wordSum = []
    oneLine = []
    for word in scut:
        wordSum.append((word.word, word.flag))
    for cuttedword in wordSum:
        tmp = cuttedword[0] + ''
        oneLine.append(tmp)
    return oneLine


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


def load_vec(vocab, model,D):
    word_vecs = {}
    for i in vocab:
        try:
            word_vecs[i] = model[i]
        except:
            word_vecs[i] = np.random.uniform(-0.0005, 0.0005, D)
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

def read_row_from_csv(filename, row_number):
    csvfile = open(filename, 'r',encoding='utf-8')
    result = csv.reader(csvfile)
    result_list = []
    row = []
    for line in result:
        result_list.append(line)
    for i in result_list:
        row.append(i[row_number - 1])
    return row

def getSamplsList(filename):
    Chief_complaint = read_row_from_csv(filename, 1)
    Current_medical_history = read_row_from_csv(filename, 2)
    Past_history = read_row_from_csv(filename, 3)
    data = []
    for a,b,c in zip(Chief_complaint,Current_medical_history,Past_history):
        data.append([a + ',' + b + ',' +  c])
    return data

def getIDs(filename,path_model,path_corpus,body_part_filename,tag,numSamples,D):

    model = gensim.models.KeyedVectors.load_word2vec_format(path_model, binary=True)
    vocab = Wordlist(path_corpus)
    w2v = load_vec(vocab.voc, model,D)
    newword2vector, word_idx_map = get_W(w2v, D)
    # add label into feature inputs
    liver, gallbladerr, pancrease, stomach, intestine, colon, appendix, peritoneum = read_label(body_part_filename,tag,numSamples)
    ids = np.zeros((numSamples, Chief_complaint_num + Current_medical_history_num + Past_history_num+1)).astype(int)
    data = getSamplsList(filename)
    for index, value in enumerate(data):

        value = value[0].split(',')
        Chief_complaint = value[0]
        Current_medical_history = value[1]
        Past_history = value[2]
        #deal Chief_complaint
        Chief_complaint_list = cut_line(Chief_complaint)
        Current_medical_history_list = cut_line(Current_medical_history)
        Past_history_list = cut_line(Past_history)
        for i in range(len(Chief_complaint_list)):
            try:
                if Chief_complaint_list[i] not in stopwords:
                    ids[index][i] = word_idx_map[Chief_complaint_list[i]]
            except:
                continue
            if i > Chief_complaint_num:
                break
        # deal Current_medical_history
        for j in range(len(Current_medical_history_list)):
            try:
                if Current_medical_history_list[j] not in stopwords:
                    ids[index][j + Chief_complaint_num] = word_idx_map[Current_medical_history_list[j]]
            except:
                pass
            if j > Current_medical_history_num  - 1:
                break
        # deal Past_history
        for k in range(len(Past_history_list)):
            try:
                if Past_history_list[k] not in stopwords:
                    ids[index][k + Chief_complaint_num + Current_medical_history_num] = word_idx_map[Past_history_list[k]]
            except:
                pass
            if k > Past_history_num:
                break
        if tag == 'liver':
            ids[index][-1] = liver[index]
            #这里需要重新设置一个效率比较高的循环
    ids = ids.astype(int)
    np.savetxt('./samples/train_256.csv', ids, delimiter=',',fmt="%.d")
    print('get idsMatrix successful')

    # for m in range(len(ids)):
    #     print(newword2vector[ids[m][0:150]])


def splitsample(filename):
    data = pd.read_csv(filename)
    columns = data.columns
    features_columns = columns.delete(len(columns) - 1)
    features = data[features_columns]
    labels = data['label']
    # print(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)
    print('training set number is : ',len(features_train))
    print('testing set number is : ',len(features_test))
    k = 0
    l = 0
    for e1 in labels_train:
        if e1 == 1:
            k +=1
    for e2 in labels_test:
        if e2 == 1:
            l +=1
    print('negative samples in training set: ',k)
    print('negative samples in testing set: ',l)
    return features_train, features_test, labels_train, labels_test


def getword2vec_to_network(D):
    model = gensim.models.KeyedVectors.load_word2vec_format("E:\\PythonProjects\\new_aad\\NLP_part\\word_embedding\\corpus\\corpus_256.model.bin",binary=True)
    vocab = Wordlist('E:\\PythonProjects\\new_aad\\NLP_part\\word_embedding\\corpus\\words.txt')
    w2v = load_vec(vocab.voc, model,D)
    newword2vector, _ = get_W(w2v, D)
    return  newword2vector
def getword2vec_to_network_mycnn(D):
    model = gensim.models.KeyedVectors.load_word2vec_format("E:\\PythonProjects\\new_aad\\NLP_part\\word_embedding\\corpus\\corpus.model.bin",binary=True)
    vocab = Wordlist('E:\\PythonProjects\\new_aad\\NLP_part\\word_embedding\\corpus\\words.txt')
    w2v = load_vec(vocab.voc, model,D)
    newword2vector, _ = get_W(w2v, D)
    return  newword2vector

def test():
    model = gensim.models.KeyedVectors.load_word2vec_format("./corpus/corpus_vgg.model.bin",binary=True)
    vocab = Wordlist('./corpus/words.txt')
    w2v = load_vec(vocab.voc, model)
    newword2vector, word_idx_map = get_W(w2v, D)

    print(word_idx_map['发热'])
    print(newword2vector[[55,56,57]])
    with open('./samples/train_vgg.csv','r',encoding='utf-8') as f:
        all = f.readlines()
    for index,onesample in enumerate(all):
        if index> 1 and index <3:
            continue
            # feature = onesample[0:224]
            # feature = np.array(feature).astype(int)
            # print()
            # print(onesample[0:224])
            # print(np.array(newword2vector[feature]))

        # onesample_vector =
        #




if __name__ == "__main__":
    # test()
    # splitsample('./samples/train_vgg.csv')
    # sample_filename = './corpus/corpus.csv'
    # body_part_filename = './samples/body_part.csv'
    # path_model = './corpus/corpus_256.model.bin'
    # path_corpus = './corpus/words.txt'
    #
    # tag_List = ['liver', 'gallbladerr', 'pancrease', 'stomach', 'intestine', 'colon', 'appendix', 'peritoneum']
    # tag = 'liver'
    #
    # getIDs(sample_filename,path_model,path_corpus,body_part_filename,tag,numSamples,256)
    filename = './samples/train_256.csv'
    splitsample(filename)



    # data = getSamplsList(filename)
    # for i in range(len(data)):
    #     if i<4:
    #         print(data[i])
    # gettestIDs()





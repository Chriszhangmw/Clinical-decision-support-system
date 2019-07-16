'''
Created on 2018年4月16日

@author: ZhangMeiwei
'''
import numpy as np
import  csv
import get_input_label
import pickle as pickle


m_seq_length = 700

   
def read_from_csv(filename):
    csvfile = open(filename,'r',encoding='utf-8')
    result = csv.reader(csvfile)
    result_list= []
    for line in result:
        result_list.append(line)
    return result_list

def transform(vocab, d):
    d = list(d)
    new_d = []
    for word in d:
        if word in vocab.keys():
            new_d.append(vocab[word])
        else:
            new_d.append(0)
    # new_d = list(map(vocab.get, d[:m_seq_length]))
    if len(new_d) >= m_seq_length:
        new_d = new_d[:m_seq_length]
    else:
        new_d = new_d + [0] * (m_seq_length - len(new_d))
    return new_d

# if __name__ == '__main__':
#     with open('./data/covab.pkl', 'br') as f:
#         vocab = pickle.load(f)
#     vocab = vocab
#     print(vocab.get)
#     words = '代号为按开发和燃放？'
#     words = list(words)
#     for word in words:
#         if word in vocab.keys():
#             print(vocab[word])
#         else:
#             print(0)
#     a  = transform(vocab, words)
#     print(a)

#for training
def GetIds(samplefilename,labelPath,tag):
    numSamples = 924
    with open('./data/covab.pkl', 'br') as f:
        vocab = pickle.load(f)
    vocab = vocab
    ids = np.zeros((numSamples, m_seq_length + 1))
    data = read_from_csv(samplefilename)
    liver, gallbladerr, pancrease, stomach, intestine, colon,lanwei_label,fumo_label = get_input_label.read_label(labelPath,tag)
    if tag == 'liver':
        input_label = liver
    elif tag == 'gallbladerr':
        input_label = gallbladerr
    elif tag == 'pancrease':
        input_label = pancrease
    elif tag == 'stomach':
        input_label = stomach
    elif tag == 'intestine':
        input_label = intestine
    elif tag == 'colon':
        input_label = colon
    elif tag == 'appendix':
        input_label = lanwei_label
    elif tag == 'peritoneum':
        input_label = fumo_label
    numWords = []
    for i in range(len(data)):
        value = list(data[i][0] + data[i][1]+ data[i][2])
        words = []
        for word in value:
            words.append(word)
        counter = len(words)
        numWords.append(counter)
        # print(words)
        ids[i][:m_seq_length] = transform(vocab, words)
        ids[i][-1] = input_label[i][0]
    ids = ids.astype(int)
    # np.save('idsMatrix_'+ tag + '_train',ids)
    np.savetxt('./data/' + tag + '/' + tag + '_train_forxupeng.csv', ids, delimiter=',', fmt="%d")
    print('get ids file successful')

    return  numWords

import matplotlib.pyplot as plt
if __name__ == "__main__":
    tag = 'peritoneum'
    # samplefilename = 'data/' + tag + '/' + tag + '_test_data.csv'
    # labelPath = 'data/' +tag + '/' + tag + '_test_label.csv'
    samplefilename = 'data/' + tag + '/' + tag + '_train_data.csv'
    labelPath = 'data/' + tag + '/' + tag + '_train_label.csv'
    numwords = GetIds(samplefilename,labelPath,tag)





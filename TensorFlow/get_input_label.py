'''
Created on 2018年3月20日

@author: ZhangMeiwei
'''
import numpy as np
import csv
import re


numSamples = 924

def read_label(labelPath,tag):
    with open(labelPath,'r',encoding = 'utf-8') as f:
        label_data = f.readlines()
    liver_label = np.zeros([numSamples,1])
    gallbladerr_label = np.zeros([numSamples,1])
    pancrease_label = np.zeros([numSamples,1])
    stomach_label = np.zeros([numSamples,1])
    intestine_label = np.zeros([numSamples,1])
    colon_label = np.zeros([numSamples,1])
    lanwei_label = np.zeros([numSamples, 1])
    fumo_label = np.zeros([numSamples, 1])
    for i in range(len(label_data)):
        line = label_data[i]
        line = line.split('\t')
        if tag == 'liver':
            tmp1 = int(line[0])
            if tmp1 > 0:
                liver_label[i][0] = 1
        elif tag == 'gallbladerr':
            tmp2 = int(line[1])
            if tmp2 > 0:
                gallbladerr_label[i][0] = 1
        elif tag == 'pancrease':
            tmp3 = int(line[2])
            if tmp3 > 0:
                pancrease_label[i][0] = 1
        elif tag == 'stomach':
            tmp4 = int(line[3])
            if tmp4 > 0:
                stomach_label[i][0] = 1
        elif tag == 'intestine':
            tmp5 = int(line[4])
            if tmp5 > 0:
                intestine_label[i][0] = 1
        elif tag == 'colon':
            tmp6 = int(line[5])
            if tmp6 > 0:
                colon_label[i][0] = 1
        elif tag == 'appendix':
            tmp7 = int(line[6])
            if tmp7 > 0:
                lanwei_label[i][0] = 1
        elif tag == 'peritoneum':
            tmp8 = int(line[7])
            if tmp8 > 0:
                fumo_label[i][0] = 1
    return liver_label,gallbladerr_label, pancrease_label, stomach_label,intestine_label,colon_label,lanwei_label,fumo_label

if __name__ == '__main__':
    tag = 'liver'
    labelPath =  'data/' +tag + '/' + tag + '_test_label.csv'
    # labelPath = 'data/' + tag + '/' + tag + '_label_after_changeorder.csv'
    a,b,c,d,e,f ,g,h= read_label(labelPath,tag)
    # print(a)
    k = 0
    for i in e :
        if i[0] == 1:
            k +=1
    print(k)






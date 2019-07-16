'''
Created on 2018年3月20日

@author: ZhangMeiwei
'''
import numpy as np
import csv
import re


numSamples = 4620

def read_label(labelPath,tag):
    with open(labelPath,'r',encoding = 'utf-8') as f:
        label_data = f.readlines()
    inflammation_label = np.zeros([numSamples,1])
    Bleeding_label = np.zeros([numSamples,1])
    obstruction_label = np.zeros([numSamples,1])
    perforation_label = np.zeros([numSamples,1])
    hernia_label = np.zeros([numSamples,1])
    cancer_label = np.zeros([numSamples,1])
    for i in range(len(label_data)):
        line = label_data[i]
        line = line.split(',')
        if tag == 'inflammation':
            tmp1 = int(line[0])
            if tmp1 > 0:
                inflammation_label[i][0] = 1
        elif tag == 'Bleeding':
            tmp2 = int(line[1])
            if tmp2 > 0:
                Bleeding_label[i][0] = 1
        elif tag == 'obstruction':
            tmp3 = int(line[2])
            if tmp3 > 0:
                obstruction_label[i][0] = 1
        elif tag == 'perforation':
            tmp4 = int(line[3])
            if tmp4 > 0:
                perforation_label[i][0] = 1
        elif tag == 'hernia':
            tmp5 = int(line[4])
            if tmp5 > 0:
                hernia_label[i][0] = 1
        elif tag == 'cancer':
            tmp6 = int(line[5])
            if tmp6 > 0:
                cancer_label[i][0] = 1

    return inflammation_label,Bleeding_label, obstruction_label, perforation_label,hernia_label,cancer_label

if __name__ == '__main__':
    tag = 'inflammation'
    # labelPath =  'data/' +tag + '/' + tag + '_test_label.csv'
    labelPath = 'data/' + tag + '/' + tag + '_label_after_changeorder.csv'
    a,b,c,d,e,f = read_label(labelPath,tag)
    # print(a)
    k = 0
    for i in a:
        if i[0] == 1:
            k +=1
    print(k)






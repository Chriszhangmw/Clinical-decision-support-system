
import re
import sys
import jieba
import jieba.posseg
import numpy as np
# Add user dict

all_v=[]
path1 = 'D:/Eclipse_workplace/web_tensorflow/cut_sentence_corpus.txt'
with open(path1,'r',encoding='utf-8') as v:
    vocabulary=v.read()
    volist=vocabulary.split()
    all_v+=volist
retrive=('n','v','l','ng','nl','nz','f','vn','a','i','vg','tg','t')   
for i in all_v:
  jieba.add_word(i,5,'n')

def forin(m):
    L=[]
    i=-1
    if m==0:
      L=[0]
    else:
      while i<m-1:
        i=i+1
        L.append(i)
    return L

def cut_non(x):
    # Input x is a sentence expected to be cut, model are A or B for different part of speech.
    # Output is a list containing segments like ['xx','dd']
    xx=','.join(re.findall(u'[\u4e00-\u9fff]+', x))
    scut=jieba.posseg.cut(xx)
    ll=[]
    A=[]
    #po=position[model][0]
    for i in scut:
        ll.append((i.word,i.flag))
    for ii in ll:
        #if ii[1] in retrive[0:po]:
        if ii[1] in retrive[0:13]:
            a=ii[0]+''
            #print(a)
            A.append(a)
    return A   


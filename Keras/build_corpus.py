
import re

import jieba.posseg


stopwords = [line.rstrip() for line in open('./corpus/stopwords.txt', 'r', encoding='utf-8')]

def cut_line(string):
    string=','.join(re.findall(u'[\u4e00-\u9fff]+', string))
    scut=jieba.posseg.cut(string)
    wordSum=[]
    oneLine=[]
    for word in scut:
        wordSum.append((word.word,word.flag))
    for cuttedword in wordSum:
        tmp=cuttedword[0]+''
        #print(a)
        oneLine.append(tmp)
    return oneLine

with open('./corpus/corpus.csv','r',encoding='utf-8') as f:
    data = f.readlines()

with open('./corpus/words.txt','w',encoding='utf-8') as f_in:
    all_word = []
    for line in data:
        line = cut_line(line)
        for word in line:
            if word not in stopwords:
                all_word.append(word.strip())
                f_in.write(word.strip() + '\n')

    # all_word = list(set(all_word))
    # for e in all_word:
    #     f_in.write(e + '\n')
f_in.close()
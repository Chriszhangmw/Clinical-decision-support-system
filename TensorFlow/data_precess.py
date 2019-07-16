
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import jieba.posseg
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import re



#
#
#
#
# datapath = './data/training_data_922_6.xlsx'
# label_file = './data/training_label_922_6.csv'
#
stopwords = [line.rstrip() for line in open('stopwords.txt','r',encoding = 'utf-8')]
#
#
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
#
# def loadfile(datapath,label_file):
#     raw = pd.read_excel(datapath, encoding='utf-8', header=None)
#     raw['all'] = None
#     for index in range(len(raw[0])):  # combine data from 3 con
#         raw['all'][index] = raw[0][index] + raw[1][index] + raw[2][index]
#         # medical segmentation
#     raw['words'] = raw['all'].apply(lambda s: jieba.cut_non(s))  # using jieba segmentation
#     label = pd.read_table(label_file)
#     re = 0
#     raw['label'] = None
#     for i, n in enumerate(label[self.tag]):
#         if n != re:
#             raw['label'][i] = 1
#         else:
#             raw['label'][i] = 0
#     y = np.array(list(raw['label']))
#     combined = np.array((list(raw['all'])))
#     return combined, y
#
if __name__ == "__main__":
    datapath = './data/training_data_922_6.csv'
    # csvfile = open(datapath,'r',encoding='utf-8')#这部分代码的作用是将样本的数据写入到TXT中
    # result = csv.reader(csvfile)
    # result_list = []
    # row = []
    # with open('samplevocab.txt','w',encoding='utf-8') as f:
    #     for line in result:
    #         result_list.append(line)
    #         # print(line)
    #         one_sample_list = cut_line(str(line[0]+line[1]+line[2]))
    #         # print(one_sample_list)
    #         for i in range(len(one_sample_list)):
    #             try:
    #                 if one_sample_list[i] not in stopwords:
    #                      f.write(one_sample_list[i])
    #             except:
    #                 continue
    # print(len(result_list))
    with open('samplevocab.txt', 'r',encoding='utf-8') as f:
        corpus = f.readlines()
        print(len(corpus))
        corpus = set(''.join([i.strip() for i in corpus]))
        print('去掉停用词之前长度：',len(corpus))
        finalcorpus = []
        for word in corpus:
            if word not in stopwords:
                print(word)
                finalcorpus.append(word)
        print('去掉停用词之后的词长度：',len(finalcorpus))

    #生成pkl文件



    #
    # try:
    #     corpus = corpus.decode('utf8')
    # except Exception as e:
    #     # print e
    #     pass










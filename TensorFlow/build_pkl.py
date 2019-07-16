
import os, sys
import collections
import pickle as pickle



with open('../cleaned_samples/corpus.csv', 'r',encoding='utf-8') as f:
    corpus = f.readlines()
    corpus = set(''.join([i.strip() for i in corpus]))
    finalcorpus = []
    for word in corpus:
        finalcorpus.append(word)
    finalcorpus = ''.join([i.strip() for i in finalcorpus])
# print(len(finalcorpus))
counter = collections.Counter(finalcorpus)
count_pairs = sorted(list(counter.items()), key=lambda i: -i[1])
chars, _ = list(zip(*count_pairs))
print('chars',chars)

word2id = {}
word2id['unknow'] = -1
for char in chars:
    if char not in word2id:
        word2id[char] = [len(word2id) + 1]
    else:
        word2id[char] += 1
new_id = 0
for word in word2id.keys():
    word2id[word] = new_id
    new_id += 1

# print(len(word2id))
with open('./covab.pkl', 'wb') as fw:
    pickle.dump(word2id, fw)

# vocab = dict(list(zip(chars, list(range(1, len(chars) + 1)))))
# print(vocab)
# print(len(vocab))
# print(vocab['日'])

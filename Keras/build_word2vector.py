

from gensim.models import word2vec

model_file_name = './corpus/words.txt'
sentences = word2vec.Text8Corpus(model_file_name)
model = word2vec.Word2Vec(sentences, size=256)
model.wv.save_word2vec_format( './corpus/corpus_256.model.bin', binary=True)



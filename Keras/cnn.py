
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
import tensorflow as tf
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D,BatchNormalization,Conv2D,MaxPooling2D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import numpy as np
import jieba
import  re
from string import digits
from NLP_part.word_embedding.get_word2vector import getword2vec_to_network_mycnn
import matplotlib.pyplot as plt
import  pandas  as pd
from sklearn.cross_validation import train_test_split
#config of model
num_classes = 2
seq_length = 150
VERBOSE = 1
VALIDATION_SPLIT = 0
num_epochs = 1000
learning_rate = 0.0001
decay_rate = 0.9
batch_size = 2




word_embedding = getword2vec_to_network_mycnn(50)
data_path = 'D:\\Eclipse_workplace\\new_aad\\NLP_part\\word_embedding\\samples\\train.csv'
data = pd.read_csv(data_path)

X = data.ix[:, data.columns != 'label']
Y = data.ix[:, data.columns == 'label']
tensor_x = np.array(X)
tensor_y = np.array(Y)

train_x, test_x, train_y, test_y = train_test_split(tensor_x, tensor_y, test_size=0.2, random_state=0)

x_train = []
x_test = []

y_train = np.zeros([len(train_x),2])
y_test = np.zeros([len(test_x),2])
for i in range(len(train_y)):
    if train_y[i] == 1:
        y_train[i][1] = 1
for i in range(len(test_y)):
    if test_y[i] == 1:
        y_test[i][1] = 1
# print(y_train)
# print(y_test)
y_test = y_train
from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()

for i in range(len(train_x)):
    x = train_x[i]
    temp = word_embedding[x]
    # sc.fit(temp)
    # temp = sc.transform(temp)
    vector = []
    vector.append(temp)
    vector = np.array(vector)
    # print(vector.shape)
    # print(vector.shape)
    vector = vector.reshape(150,50,1)
    x_train.append(vector)
x_train = np.array(x_train)

for i in range(len(test_x)):
    x = test_x[i]
    temp = word_embedding[x]
    # sc.fit(temp)
    # temp = sc.transform(temp)
    vector = []
    vector.append(temp)
    vector = np.array(vector)
    vector = vector.reshape(150,50,1)
    x_test.append(vector)
# x_test = np.array(x_test)
x_test =np.array(x_train)
main_input = Input(shape=(150,50,1))
# fc1 = Dense()
x = Conv2D(filters=16, kernel_size=5, padding="valid", activation="relu", name='block1_conv1')(main_input)
x = MaxPooling2D(pool_size=(5, 5), strides=(4, 4), name='block1_pool')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=32, kernel_size=5, padding="valid", activation="relu", name='block1_conv2')(x)
x = MaxPooling2D(pool_size=(5, 5), strides=(4, 4), name='block1_poo2')(x)
x = BatchNormalization()(x)
x = Flatten(name='flatten')(x)
x = Reshape((224,1))(x)
x = LSTM(224)(x)
x = Dense(units=2, activation="softmax", name="prediction")(x)
# print(x)

model = Model(inputs=main_input,outputs=x)
model.summary()
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0001),
              metrics=["accuracy"])
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=True,validation_data=(x_test,y_test))
# score = model.evaluate(x=x_test, y=y_test, verbose=VERBOSE)
# print("test loss:", score[0])
# print("test acc:", score[1])

# 列出历史数据
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# 汇总损失函数历史数据
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()





































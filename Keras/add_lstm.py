
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.applications.vgg16 import VGG16
import pandas as pd
from sklearn.cross_validation import train_test_split


from sklearn.preprocessing import StandardScaler
from NLP_part.word_embedding.get_word2vector import getword2vec_to_network_mycnn
from NLP_part.word_embedding.util import TextLoader_vgg
import itertools
import numpy as np
import matplotlib.pyplot as plt


num_classes = 2
seq_length = 150
VERBOSE = 1
VALIDATION_SPLIT = 0.2
num_epochs = 2
learning_rate = 0.001
decay_rate = 0.9
batch_size = 4

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


sc = StandardScaler()
for i in range(len(train_x)):
    x = train_x[i]
    vector = word_embedding[x]
    sc.fit(vector)
    vector = sc.transform(vector)
    vector = vector.reshape(150,50,1)
    x_train.append(vector)
x_train = np.array(x_train)

for i in range(len(test_x)):
    x = test_x[i]
    vector = word_embedding[x]
    sc.fit(vector)
    vector = sc.transform(vector)
    vector = vector.reshape(150,50,1)
    x_test.append(vector)
x_test = np.array(x_test)
print(x_test.shape)

class LeNet(Model):
    def __init__(self, input_shape=(150, 50, 1), num_classes=2):
        # super(LeNet, self).__init__(name="LeNet")
        self.num_classes = num_classes
        ''' 定义要用到的层 layers '''
        # 输入层
        img_input = Input(shape=input_shape)

        # Conv => ReLu => Pool
        x = Conv2D(filters=20, kernel_size=5, padding="same", activation="relu" ,name='block1_conv1')(img_input)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)
        # Conv => ReLu => Pool
        x = Conv2D(filters=50, kernel_size=5, padding="same", activation="relu", name='block1_conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_poo2')(x)
        # 压成一维
        x = Flatten(name='flatten')(x)
        # 全连接层
        x = Dense(units=500, activation="relu", name="f1")(x)
        # softmax分类器
        x = Dense(units=num_classes, activation="softmax", name="prediction")(x)

        # 调用Model类的Model(input, output, name="***")构造方法
        super(LeNet, self).__init__(img_input, x, name="LeNet")

    def call(self, inputs):
        # 前向传播计算
        # 使用在__init__方法中定义的层
        return self.output(inputs)
INPUT_SHAPE = (150,50,1)
NB_CLASSES = 2
model = LeNet(INPUT_SHAPE, NB_CLASSES)
model.summary()

model.compile(loss="categorical_crossentropy", optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
              metrics=["accuracy"])
history = model.fit(x=x_train, y=y_train, batch_size=4, epochs=20, verbose=VERBOSE,validation_split=VALIDATION_SPLIT)
score = model.evaluate(x=x_test, y=y_test, verbose=VERBOSE)














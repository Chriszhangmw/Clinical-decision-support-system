import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.applications.vgg16 import VGG16



import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from NLP_part.word_embedding.get_word2vector import getword2vec_to_network
import pandas as pd
from sklearn.model_selection import train_test_split


# tf.enable_eager_execution()

input_shape = 224 * 224 * 3
classes = 2

NB_EPOCH = 20
BATCH_SIZE = 4
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 224, 224
NB_CLASSES = 2
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3) # 注意在TF中的数据格式 NHWC

word_embedding = getword2vec_to_network(672)
data_path = 'D:\\Eclipse_workplace\\new_aad\\NLP_part\\word_embedding\\samples\\train_vgg.csv'
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



sc = StandardScaler()

for i in range(len(train_x)):
    x = train_x[i]
    vector = word_embedding[x]
    sc.fit(vector)
    vector = sc.transform(vector)
    vector = vector.reshape(224,224,3)
    x_train.append(vector)
x_train = np.array(x_train)

for i in range(len(test_x)):
    x = test_x[i]
    vector = word_embedding[x]
    sc.fit(vector)
    vector = sc.transform(vector)
    vector = vector.reshape(224,224,3)
    x_test.append(vector)
x_test = np.array(x_test)
print(x_test.shape)

class vggNet(Model):
    def __init__(self, num_classes=2):
        # super(LeNet, self).__init__(name="LeNet")
        self.num_classes = num_classes
        ''' 定义要用到的层 layers '''
        # 输入层
        # img_input = input_shape

        base_model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False
        last = base_model.output
        x = Dense(256, activation='relu')(last)
        x = Dropout(0.5)(x)
        x = Dense(num_classes, activation='softmax')(x)


        # 调用Model类的Model(input, output, name="***")构造方法
        super(vggNet, self).__init__(x, name="vggNet")

    def call(self, inputs):
        # 前向传播计算
        # 使用在__init__方法中定义的层
        return self.output(inputs)

model = vggNet(NB_CLASSES)
model.summary()

model.compile(loss="categorical_crossentropy", optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
              metrics=["accuracy"])
history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,validation_split=VALIDATION_SPLIT)
score = model.evaluate(x=x_test, y=y_test, verbose=VERBOSE)
print("test loss:", score[0])
print("test acc:", score[1])

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

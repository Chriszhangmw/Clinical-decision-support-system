# import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense,Dropout,LSTM
from tensorflow.keras.optimizers import Adam,SGD

from tensorflow.python.keras.applications.vgg16 import VGG16
# import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from NLP_part.word_embedding.get_word2vector import getword2vec_to_network
from NLP_part.word_embedding.util import TextLoader_vgg
import itertools
import numpy as np
import matplotlib.pyplot as plt
import  pandas  as pd
from sklearn.cross_validation import train_test_split


def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=None)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_true, y_pred)
    class_names = ['positive', 'negative']
    plot_confusion_matrix(confusion_matrix
                          , classes=class_names
                          , title='Confusion matrix')


num_classes = 2
seq_length = 224
VERBOSE = 1
VALIDATION_SPLIT = 0.2
num_epochs = 100
learning_rate = 0.0001
decay_rate = 0.9
batch_size = 4


# class network():
#     def __init__(self,word_embedding):
#         self.word_embedding = word_embedding
#         self.input_data = tf.placeholder(tf.int32, [None, seq_length], name='inputs')
#         self.targets = tf.placeholder(tf.float64, [None,2])  # target is class label
#         self.data = tf.nn.embedding_lookup(self.word_embedding, self.input_data)
#         # self.out_x = self.data
#         self.out_x = tf.expand_dims(self.data, 3)
#         # print(self.output.get_shape)
#         # self.out_x = tf.reshape(self.output, [-1, 30, 25, 1])

word_embedding = getword2vec_to_network(256)
# net1 = network(word_embedding)

''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''
data_path = 'D:\\Eclipse_workplace\\new_aad\\NLP_part\\word_embedding\\samples\\train_256.csv'
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
    temp = word_embedding[x]
    sc.fit(temp)
    temp = sc.transform(temp)
    vector = []
    vector.append(temp)
    vector.append(temp)
    vector.append(temp)
    vector = np.array(vector)
    # print(vector.shape)
    vector = vector.reshape(224,224,3)
    x_train.append(vector)
x_train = np.array(x_train)

for i in range(len(test_x)):
    x = test_x[i]
    temp = word_embedding[x]
    sc.fit(temp)
    temp = sc.transform(temp)
    vector = []
    vector.append(temp)
    vector.append(temp)
    vector.append(temp)
    vector = np.array(vector)
    vector = vector.reshape(224,224,3)
    x_test.append(vector)
x_test = np.array(x_test)

def vgg16_model(input_shape = (224,224,3)):
    base_model = VGG16(weights='imagenet', include_top=True,input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    for i ,layer_ in enumerate(base_model.layers):
        print(i,layer_.name)
    last = base_model.output
    # tf.reshape(last,[None,1000,1])
    print(last.get_shape)
    # x = Flatten(name='flatten')(last)
    # x = Dense(40000, activation='relu')(last)
    # x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(last)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input,outputs=x)
    return model

model_vgg16 = vgg16_model(input_shape = (224,224,3))
model_vgg16.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='categorical_crossentropy')
model_vgg16.summary()
model_vgg16.compile(loss='categorical_crossentropy',optimizer = sgd, metrics = ['accuracy'])

history = model_vgg16.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=True,validation_split=VALIDATION_SPLIT)
y = model_vgg16.predict(x=x_test)

# print('*'*20)
# print(len(yy))
# print('yy:',yy)
# print(len(y))
# print('y:',y)
y_pre = []
for pre in y:
    pre = np.array(pre)
    y_pre.append(np.argmax(pre))
print('*'*20)
plot_matrix(y_test,y_pre)



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






























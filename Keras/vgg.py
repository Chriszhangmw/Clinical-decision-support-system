


from keras.applications.vgg16 import VGG16
from keras.models import  Model
from keras.layers import Dense, Dropout,Flatten
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import TensorBoard

#tf.keras  version
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense
# from tensorflow.keras.optimizers import Adam


import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from NLP_part.word_embedding.get_word2vector import getword2vec_to_network
from NLP_part.word_embedding.util import TextLoader_vgg
import itertools
import numpy as np
import matplotlib.pyplot as plt



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
VALIDATION_SPLIT = 0
num_epochs = 10
learning_rate = 0.0001
decay_rate = 0.9
batch_size = 4

class network():
    def __init__(self,word_embedding):
        self.word_embedding = word_embedding
        self.input_data = tf.placeholder(tf.int32, [None, seq_length], name='inputs')
        self.targets = tf.placeholder(tf.float64, [None,2])  # target is class label
        self.data = tf.nn.embedding_lookup(self.word_embedding, self.input_data)
        # self.out_x = self.data
        self.out_x = tf.expand_dims(self.data, 3)
        # print(self.output.get_shape)
        # self.out_x = tf.reshape(self.output, [-1, 30, 25, 1])

word_embedding = getword2vec_to_network(672)
net1 = network(word_embedding)

loader = TextLoader_vgg(1)
x_temp = []
y_train = []
test_x_temp = []
test_y_train = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for b in range(loader.num_batches):
        x, y = loader.next_batch(b)
        feed = {net1.input_data: x, net1.targets: y}
        out_x,out_y = sess.run([net1.out_x,net1.targets], feed_dict=feed)
        # print(out_x.shape)
        # print(out_y.shape)
        x_temp.append(out_x)
        y_train.append(out_y)
    for c in range(loader.test_num_batches):
        x_,y_ = loader.next_test_batch(c)
        feed_ = {net1.input_data: x_, net1.targets: y_}
        test_out_x, test_out_y = sess.run([net1.out_x, net1.targets], feed_dict=feed_)
        test_x_temp.append(test_out_x)
        test_y_train.append(test_out_y)


x_temp = np.array(x_temp)
test_x_temp = np.array(test_x_temp)
x_temp = np.squeeze(x_temp)
test_x_temp = np.squeeze(test_x_temp)

x_train = []
x_test = []
sc = StandardScaler()

for x_ in test_x_temp:
    sc.fit(x_)
    x_ = sc.transform(x_)
    x_ = x_.reshape(224,224,3)
    x_test.append(x_)

for x in x_temp:
    sc.fit(x)
    x = sc.transform(x)
    x = x.reshape(224,224,3)
    x_train.append(x)
x_train = np.array(x_train)
x_test = np.array(x_test)
print('x_right[0].shape:',x_train[0].shape)
print('x_train shape:',x_train.shape)
y_train = np.array(y_train)
test_y_train = np.array(test_y_train)

y_train = np.squeeze(y_train)
test_y_train = np.squeeze(test_y_train)
yy = []
k = 0
for lable in test_y_train:
    if lable[1] == 1:
        yy.append(1)
        k+=1
    else:
        yy.append(0)

# print('*'*20)
# print(test_y_train)
# print(len(test_y_train))
# print(test_y_train[0])
# print('k:',k)
# print(yy)
# print('*'*20)



def vgg16_model(input_shape = (224,224,3)):
    base_model = VGG16(weights='imagenet', include_top=True,input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    # base_model.layers.pop(-1)
    # base_model.layers.pop(-1)
    # base_model.layers.pop(-1)
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
model_vgg16.compile(loss='categorical_crossentropy',optimizer =tf.train.RMSPropOptimizer(0.0001), metrics = ['accuracy'])

# history = model_vgg16.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=True,
#                           callbacks=[TensorBoard(log_dir='./mytensorboard')])
history = model_vgg16.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=True)
y = model_vgg16.predict(x=x_test)
print('*'*20)
print(len(yy))
print('yy:',yy)
print(len(y))
print('y:',y)
y_pre = []
for pre in y:
    pre = np.array(pre)
    y_pre.append(np.argmax(pre))
print('*'*20)
plot_matrix(yy,y_pre)
# score = model_vgg16.evaluate(x=x_test, y=test_y_train, verbose=VERBOSE)
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


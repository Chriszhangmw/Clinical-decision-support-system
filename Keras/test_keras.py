import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.datasets import mnist
from  tensorflow.examples.tutorials.mnist   import   input_data
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

# tf.enable_eager_execution()

input_shape = 28 * 28
classes = 10

NB_EPOCH = 20
BATCH_SIZE = 200
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0
IMG_ROWS, IMG_COLS = 28, 28
NB_CLASSES = 10
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1) # 注意在TF中的数据格式 NHWC

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255
print(x_train[0].shape)
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255

print(y_train.shape)
print(y_train[0].shape)

class LeNet(Model):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
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

model = LeNet(INPUT_SHAPE, NB_CLASSES)
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

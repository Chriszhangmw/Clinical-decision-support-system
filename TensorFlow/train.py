#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import tensorflow as tf

from utils import TextLoader,TextLoader_test
from model import Model
import matplotlib.pyplot as plt
import itertools
num_epochs = 30
learning_rate = 0.001
decay_rate = 0.9
import  numpy as np


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



def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id
def random_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab)+1, embedding_dim))
    # embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def train(embeddings,Ture,tag):
    data_loader = TextLoader(4,tag)
    vocab_size = data_loader.vocab_size
    print('vocab_size',vocab_size)
    print('downloading model.......')
    model = Model(embeddings,Ture)
    print('finished downloading model.......')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./tf_log/' + tag + '/', sess.graph)
        print('start loop .......')
        for e in range(num_epochs):
            sess.run(tf.assign(model.lr, learning_rate * (decay_rate ** e)))
            data_loader.shuff()
            svm_x = []
            svm_y = []
            for b in range(data_loader.num_batches):
                x, y = data_loader.next_batch(b)
                # print(x.shape)
                # print('trouble shooting tag')
                feed = {model.input_data: x, model.targets: y}
                train_loss, state, _, accuracy,summary_,output3 = sess.run([model.cost, model.final_state, model.optimizer, model.accuracy,model.merged,model.output3], feed_dict=feed)
                writer.add_summary(summary_, global_step=b)
                print('第{}批次，第{}次循环时,train_loss ={},accuracy ={},process tag is {}'.format(e,b,train_loss,accuracy,tag))
                if e == num_epochs-1:
                    svm_x.append(np.array(output3))
                    svm_y.append(y)
                if e == num_epochs-1 and b == data_loader.num_batches-1:
                    saver.save(sess,'model_14/'+ tag + '/' + tag + '.model')
    a, b, c, d = np.array(svm_x).shape
    svm_x = np.array(svm_x).reshape((a * b, c * d))
    e, f = np.array(svm_y).shape
    svm_y = np.array(svm_y).reshape((e * f, 1))
    print(np.array(svm_x).shape)
    print(np.array(svm_y).shape)
    np.save('./Peng_second/'+ tag + '_x.npy', svm_x)
    np.save('./Peng_second/'+ tag + '_y.npy', svm_y)


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
def LR_model(x_test,true_label):
    x = np.load("./liver_lr_x.npy")
    y = np.load("./liver_lr_y.npy")
    lr = LogisticRegression(C=9000, penalty='l1')
    # # lr = LogisticRegression(C=9000.0, random_state=0)
    print(x.shape)
    print(y.shape)
    x_test = np.array(x_test)
    print('x_test shape:', x_test.shape)
    a1, a2, a3 = x_test.shape
    x_test = x_test.reshape((a1, a2 * a3))
    # print(y)
    a,b,c,d = x.shape
    x = x.reshape((a*b,c*d))
    e,f = y.shape
    y = y.reshape((e*f,1))
    print(y)
    mm = 0
    for ee in y:
        if ee == 1:
            mm +=1
    print('mm 的值为：',mm)

    sc = StandardScaler()  # 初始化一个对象sc去对数据集作变换
    sc.fit(x)
    X_train_std = sc.transform(x)
    X_test_std = sc.transform(x_test)
    lr.fit(X_train_std, y)
    y_pred = lr.predict_proba(X_test_std)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    j = 1
    for i in thresholds:
        y_test_predictions_high_recall = y_pred[:, 1] > i
        # plt.subplot(3, 3, j)
        plot_matrix(true_label, y_test_predictions_high_recall)
        j += 1





    # clf = svm.SVC(C=8000, kernel='rbf')#目前是最佳poly
    # clf = svm.SVC(C=8000, kernel='poly',probability=True,random_state = 0)
    # clf.fit(X_train_std,y)
    # y_pred = clf.predict(X_test_std)

    return  y_pred


def test_lr(tag):
    data_loader = TextLoader_test(2, tag)
    saver = tf.train.import_meta_graph('model_14/' + tag + '/' + tag + '.model.meta')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('model_14/' + tag + '/'))
        inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
        y = tf.get_default_graph().get_tensor_by_name('lr_input2:0')
        true_label = []
        x_test = []
        for i in range(data_loader.num_batches):
            x, label = data_loader.next_batch(i)
            prediction = sess.run(y, {inputs: x})
            print('prediction shape is :',prediction.shape)
            print('label shape is :',label.shape)
            true_label.append(label)
            x_test.append(np.array(prediction))
        a, b, c, d = np.array(x_test).shape
        x_test = np.array(x_test).reshape((a * b, c * d))
        e, f = np.array(true_label).shape
        svm_y = np.array(true_label).reshape((e * f, 1))
        print(np.array(true_label).shape)
        print(np.array(svm_y).shape)
        np.save('./Peng_second/' + tag + '_x_test.npy', x_test)
        np.save('./Peng_second/' + tag + '_y_test.npy', svm_y)


    #         for i in label:
    #             true_label.append(i)
    #         for j in prediction:
    #             pre_label.append(j)
    # y_pred = LR_model(pre_label,true_label)
    # plot_matrix(true_label, y_pred)

##########################################################
#    form  xupeng gragh
###########################################################

from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, classification_report, roc_curve, precision_recall_curve, auc

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
##########################################################
#    form  xupeng gragh
###########################################################


if __name__ == '__main__':
    tag_List =['stomach','pancrease','intestine','colon','appendix','gallbladerr','peritoneum']
    for tag in tag_List:
        test_lr(tag)



    # tag = 'peritoneum'
    # data_loader = TextLoader_test(2,tag)
    # saver = tf.train.import_meta_graph('model_14/' + tag + '/' + tag + '.model.meta')
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     saver.restore(sess, tf.train.latest_checkpoint('model_14/' + tag + '/'))
    #     inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    #     y = tf.get_default_graph().get_tensor_by_name('softmaxLayer/y:0')
    #     true_label = []
    #     pre_label = []
    #     model_p = []
    #     # print(data_loader.num_batches)
    #     # jj = 0
    #     for i in range(data_loader.num_batches):
    #         # print(i)
    #         x, label = data_loader.next_batch(i)
    #         prediction = sess.run(y,{inputs:x})
    #         # print(prediction)
    #         for pre in prediction:
    #             # jj +=1
    #             model_p.append(pre[1])
    #         prediction = np.argmax(prediction,1)
    #         for i in label:
    #             true_label.append(i)
    #         for j in prediction:
    #             pre_label.append(j)
    #     print(confusion_matrix(true_label, pre_label))
    #     print(classification_report(true_label, pre_label))
    #     p, r, thresholds = precision_recall_curve(true_label, model_p)
    #     plot_precision_recall_vs_threshold(p, r, thresholds)
    #     fpr, tpr, auc_thresholds = roc_curve(true_label, model_p)
    #     print(auc(fpr, tpr))  # AUC of ROC
    #     plot_roc_curve(fpr, tpr, 'recall_optimized')

    # plot_matrix(true_label, pre_label)


    # word2id = read_dictionary('./data/covab.pkl')
    # print('*'*100,word2id)
    # embeddings = random_embedding(word2id, 128)
    # tag_List =['stomach']
    # for tag in tag_List:
    #     train(embeddings, True,tag)
    # tag_List = ['pancrease','intestine','colon','appendix','inflammation','obstruction','perforation', 'hernia']
    # 'liver', 'gallbladerr', 'pancrease', 'stomach', 'intestine', 'colon',
    # 'appendix', 'peritoneum', 'inflammation', 'Bleeding', 'obstruction',
    # 'perforation', 'hernia', 'cancer'
    #



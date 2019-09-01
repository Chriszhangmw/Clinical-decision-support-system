
import os
import tensorflow as tf

import bert.modeling as modeling
from bert import tokenization
from model import Model
from data_processor import  fffffuck
from text_loader import TextLoader




# 模型操作辅助类
# modelpp = Model()

lr = 0.0006  # 学习率
# 配置文件
data_root = './bert_model_chinese'
bert_config_file = os.path.join(data_root, 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = os.path.join(data_root, 'bert_model.ckpt')
bert_vocab_file = os.path.join(data_root, 'vocab.txt')
token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)

input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
input_y = tf.placeholder(tf.float32, shape=[None, 1], name="input_y")
weights = {
    'out': tf.Variable(tf.random_normal([768, 1]))
}
biases = {
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}

model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False)

tvars = tf.trainable_variables()
(assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment)
output_layer_pooled = model.get_pooled_output()  # 这个获取句子的output
output_layer_pooled = tf.nn.dropout(output_layer_pooled, keep_prob=0.9)

w_out = weights['out']
b_out = biases['out']
pred = tf.add(tf.matmul(output_layer_pooled, w_out), b_out, name="pre1")
pred = tf.reshape(pred, shape=[-1, 1], name="pre")

loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(input_y, [-1])))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)



EPOCHS = 5
max_sentence_length = 512
batch_size = 8
data_path = './data'
train_input,predict_input =fffffuck(data_path,bert_vocab_file,True,True,
                                               './temp',max_sentence_length,batch_size,batch_size,batch_size)
data_loader = TextLoader(train_input,batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        data_loader.shuff()
        for j in range(data_loader.num_batches):
            x_train, y_train = data_loader.next_batch(j)
            # print(y_train)
            # print(y_train.shape)
            x_input_ids = x_train[:,0]
            x_input_mask = x_train[:,1]
            x_segment_ids = x_train[:,2]
            loss_, _ = sess.run([loss, train_op],
                                feed_dict={input_ids: x_input_ids, input_mask: x_input_mask, segment_ids: x_segment_ids,
                                           input_y: y_train})
            print('loss:', loss_)
    # modelpp.save_model(sess, './model_save', overwrite=True)
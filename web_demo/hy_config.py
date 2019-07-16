
class train_config(object):
    def __init__(self):
        self.epochs = 1
        self.learning_rate = 0.001
        self.decay_rate = 0.9
        self.vocab_path = './data/covab.pkl'
        self.embeddings_size = 128
        self.seq_length = 700

    def import_meta_graph(self,tag):
        # 'D:/Eclipse_workplace/Practice/Test_flask/samples/samples.csv'
        return 'D:/Eclipse_workplace/web_tensorflow/model_14/' + tag + '/' + tag + '.model.meta'

    def latest_checkpoint(self,tag):
        return 'D:/Eclipse_workplace/web_tensorflow/model_14/' + tag + '/'

class Config_LSTM(object):
    def __init__(self):
        self.rnn_size = 128
        self.num_layers = 2
        self.label_size = 2
        self.seq_length = 700
        self.vocab_path = './data/covab.pkl'

    #for  training data in utils
    def get_train_data_path(self,tag):
        return  './data/' + tag + '/' + tag + '_train.csv'

     # for testing data in utils
    def get_test_data_path(self, tag):
        return './data/' + tag + '/' + tag + '_test.csv'



class Config_CNN(object):
    def __init__(self):
        #for maxpooling layer
        self.max_pooling_ksize = [1, 2, 2, 1]
        self.max_pooling_strides = [1, 2, 2, 1]
        #for normlaziton layer
        self.norm_bias = 1.0
        self.norm_alpha = 0.001 / 9.0
        self.norm_beta = 0.75
        #for conculation layer
        self.kernel_shape = [11, 11, 1, 32]
        self.strides1 = [1,2,2,1]
        self.biases1_shape = 32
        self.lsize1=4

        self.kerne2_shape = [11, 11, 32, 64]
        self.kerne2_shape_small = [3, 3, 32, 64]
        self.strides2 = [1, 2, 2, 1]
        self.biases2_shape = 64
        self.lsize2 = 4

        self.kerne3_shape = [5, 5, 64, 128]
        self.strides3 = [1, 2, 2, 1]
        self.biases3_shape = 128
        self.lsize3 = 4



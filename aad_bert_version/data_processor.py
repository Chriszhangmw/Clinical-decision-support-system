
import os
import tensorflow as tf
import csv
import bert.tokenization as tokenization
import collections
import numpy as np
from keras.utils import to_categorical






class InputExample(object):
  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

class MyProcessor():
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return  ["0", "1"]

  def _read_tsv(self, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    print(len(lines))
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(line[0])
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example_simple( max_seq_length,
                           tokenizer,text_a,text_b=None):
  tokens_a = tokenizer.tokenize(text_a)
  tokens_b = None
  if text_b:
    tokens_b = tokenizer.tokenize(text_b)# 这里主要是将中文分字
  if tokens_b:
    # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
    # 因为要为句子补上[CLS], [SEP], [SEP]
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
  # (a) 两个句子:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) 单个句子:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # 这里 "type_ids" 主要用于区分第一个第二个句子。
  # 第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
  # 因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
  # 创建mask
  input_mask = [1] * len(input_ids)
  # 对于输入进行补0
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return input_ids,input_mask,segment_ids # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  # print(example)

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature



def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  features = []
  # print(label_list)
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      print("Writing example %d of %d" % (ex_index, len(examples)))
    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)
    temp = []
    # print(feature.input_ids)
    temp.append(feature.input_ids)
    temp.append(feature.input_mask)
    temp.append(feature.segment_ids)
    temp.append(feature.label_id)
    # print(temp)
    features.append(temp)

  return features

def fffffuck(data_dir,vocab_file_path,do_train,do_predict,output_dir,
             max_seq_length,train_batch_size,eval_batch_size,predict_batch_size):
    train_input = None
    eval_input = None
    predict_input = None
    processor = MyProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file_path, do_lower_case=True)
    if do_train:
        train_examples = processor.get_train_examples(data_dir)
        train_file = os.path.join(output_dir, "train.tf_record")
        train_input = file_based_convert_examples_to_features(
            train_examples, label_list, max_seq_length, tokenizer, train_file)
        print('***start to training**')
        print('  Number training examples  %d',len(train_examples))
        print('   Batch size %d',train_batch_size)
    # if do_eval:
    #     eval_examples = processor.get_dev_examples(data_dir)
    #     eval_file = os.path.join(output_dir, "eval.tf_record")
    #     eval_input = file_based_convert_examples_to_features(
    #         eval_examples, label_list, max_seq_length, tokenizer, eval_file)
    #     print('***start to validation**')
    #     print('  Number validate examples  %d', len(eval_examples))
    #     print('   Batch size %d', eval_batch_size)
    if do_predict:
        predict_examples = processor.get_test_examples(data_dir)
        predict_file = os.path.join(output_dir, "predict.tf_record")
        predict_input = file_based_convert_examples_to_features(predict_examples, label_list,
                                                max_seq_length, tokenizer,
                                                predict_file)
        print('***start to predict**')
        print('  Number predict examples %d', len(predict_examples))
        print('   Batch size  %d', predict_batch_size)

    return train_input,predict_input














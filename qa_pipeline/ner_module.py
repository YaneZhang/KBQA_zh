import tensorflow as tf
import os
import codecs
import pickle
import pandas as pd
import numpy as np

import sys 
sys.path.append("..")
from bert import tokenization
from bert import modeling

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

is_training=False
use_one_hot_embeddings=False
batch_size=1

vocab_file="../chinese_L-12_H-768_A-12/vocab.txt"
bert_config_file="../chinese_L-12_H-768_A-12/bert_config.json"
init_checkpoint="../chinese_L-12_H-768_A-12/bert_model.ckpt"
max_seq_length=48
output_dir="./output/ner_dir"
do_lower_case=True

graph = tf.Graph()
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config, graph=graph)
model=None

input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None
print(output_dir)
print('checkpoint path:{}'.format(os.path.join(output_dir, "checkpoint")))
if not os.path.exists(os.path.join(output_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")


with codecs.open(os.path.join(output_dir, 'nerlabel2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}

num_labels = len(label2id)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id=None,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_sequence_output()    # [batch_size, seq_length, hidden_size]

    transition = tf.get_variable(
        "transition", [num_labels, num_labels],
        initializer=tf.contrib.layers.xavier_initializer()
    )

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        linear = tf.keras.layers.Dense(num_labels,activation=None)
        logits = linear(output_layer)
        logits = tf.reshape(logits,[-1,max_seq_length,num_labels])

        mask2len = tf.reduce_sum(input_mask,axis=1)
        predict,viterbi_score = tf.contrib.crf.crf_decode(logits, transition, mask2len)

        return (logits, predict)


# 加载checkpoint
with graph.as_default():
    print("going to restore checkpoint")
    input_ids_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_mask")
    segment_ids_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="segment_ids")

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    (logits, pred_ids) = create_model(
        bert_config, is_training, input_ids_p, input_mask_p, segment_ids_p,
        num_labels, use_one_hot_embeddings)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(output_dir))


tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_id_to_entity(sentence, pred_ids_result, idx2label):
    """
    :param sentence: 命名实体识别的句子
    :param pred_ids_result: 预测序列ID
    :param idx2label: ID到label的映射
    :return: 命名实体的list
    """
    curr_seq, entities = [], []
    curr_entity = ""
    for ids in pred_ids_result[0]:
        if ids == 0:
            break
        curr_label = idx2label[ids]
        if curr_label in ['[CLS]', '[SEP]']:
            continue
        curr_seq.append(curr_label)

    assert len(curr_seq) == len(sentence)
    
    for i in range(len(curr_seq)-1):                # 之所以再遍历一次是因为担心有多个实体的情况
        if curr_seq[i] == "B" and curr_seq[i+1] != "I":
            entities.append(sentence[i])
        if curr_seq[i] == "B" and curr_seq[i+1] == "I":
            curr_entity += sentence[i]
        if curr_seq[i] == "I" and curr_seq[i+1] == "I":
            curr_entity += sentence[i]
        if curr_seq[i] == "I" and curr_seq[i+1] != "I":
            curr_entity += sentence[i]
            entities.append(curr_entity)
    if curr_seq[-1] == "I":
        curr_entity += sentence[i]
        entities.append(curr_entity)
    return entities


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    """
    Converts a single `InputExample` into a single `InputFeatures`.
    """

    tokens = example

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]

    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for i in range(len(tokens)):
        ntokens.append(tokens[i])
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        ntokens.append("[PAD]")


    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(ntokens) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_real_example=True)
    return feature


def ner_predict(sentence):
    '''predict阶段应该没有label_ids'''

    def convert(line):
        feature = convert_single_example(0, line, max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids],(batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, max_seq_length))
        return input_ids, input_mask, segment_ids

    # global graph
    with graph.as_default():
        if len(sentence) < 2:
            return sentence
        
        sentence = tokenizer.tokenize(sentence)
        input_ids, input_mask, segment_ids = convert(sentence)

        feed_dict = {input_ids_p: input_ids,
                    input_mask_p: input_mask,
                    segment_ids_p: segment_ids}
        
        pred_ids_result = sess.run([pred_ids], feed_dict)
        print(f"pred_ids_result: {pred_ids_result}")
        pred_entity = convert_id_to_entity(sentence, pred_ids_result[0].tolist(), id2label)

        return pred_entity

if __name__ == "__main__":
    while 1:
        print("please input test sentence:\n")
        sent = str(input())
        if sent == "end":
            break
        entity = ner_predict(sent)
        if not entity:
            print("this is so hard for me.")
        else:
            print(f"the answer is: {entity}")
    





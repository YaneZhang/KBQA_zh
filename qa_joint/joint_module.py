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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

is_training=False
use_one_hot_embeddings=False
batch_size=1

vocab_file="../chinese_L-12_H-768_A-12/vocab.txt"
bert_config_file="../chinese_L-12_H-768_A-12/bert_config.json"
init_checkpoint="../chinese_L-12_H-768_A-12/bert_model.ckpt"
max_seq_length=64
output_dir="./output/joint_dir"
do_lower_case=True

graph = tf.Graph()
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=gpu_config)

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

with codecs.open(os.path.join(output_dir, "intentlabel2id.pkl"), 'rb') as rf:
    intentlabel2id = pickle.load(rf)
    id2intentlabel = {value: key for key, value in intentlabel2id.items()}

with codecs.open(os.path.join(output_dir, "nerlabel2id.pkl"), 'rb') as rf:
    nerlabel2id = pickle.load(rf)
    id2nerlabel = {value: key for key, value in nerlabel2id.items()}

num_intent_labels = len(intentlabel2id)
num_slot_labels = len(nerlabel2id)
print(f"num_intent_labels: {num_intent_labels}, num_slot_labels: {num_slot_labels}")

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 intent_label_ids=None,
                 slot_label_ids=None,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.intent_label_ids = intent_label_ids
        self.slot_label_ids = slot_label_ids
        self.is_real_example = is_real_example


def intent_classification(model, labels, num_labels, is_training):

    output_layer = model.get_pooled_output()    # [batch_size, hidden_size]

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels],
        initializer=tf.zeros_initializer())

    with tf.variable_scope("intent_loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, num_labels])
        predict = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
        
        return (logits, predict)


def slot_classification(model, num_labels, input_mask, labels, is_training):

    output_layer = model.get_sequence_output()       # float Tensor of shape [batch_size, seq_length, hidden_size]
    
    transition = tf.get_variable(
        "transition", [num_labels, num_labels],
        initializer=tf.contrib.layers.xavier_initializer()
    )

    with tf.variable_scope("slot_loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        linear = tf.keras.layers.Dense(num_labels,activation=None)
        logits = linear(output_layer)
        logits = tf.reshape(logits,[-1,max_seq_length,num_labels])

        
        mask2len = tf.reduce_sum(input_mask,axis=1)

        predict,viterbi_score = tf.contrib.crf.crf_decode(logits, transition, mask2len)

        return (logits, predict)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, num_intent_labels, num_slot_labels, 
                 use_one_hot_embeddings, intent_label_ids=None, slot_label_ids=None):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    intent_probabilities = intent_classification(model=model, labels=intent_label_ids, num_labels=num_intent_labels, is_training=is_training)
    slot_predict = slot_classification(model=model, num_labels=num_slot_labels, input_mask=input_mask, labels=slot_label_ids, is_training=is_training)

    return (intent_probabilities, slot_predict)


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    tokens = tokenizer.tokenize(example)
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]

    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
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

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        is_real_example=True)
    return feature


with graph.as_default():
    input_ids = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length], name="input_ids")
    input_mask = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length], name="input_mask")
    segment_ids = tf.placeholder(tf.int32, shape=[batch_size, max_seq_length], name="segment_ids")

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    

    intent_predict, slot_predict = create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                                                     num_intent_labels, num_slot_labels, use_one_hot_embeddings)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(output_dir))


def joint_predict(sentence):

    def convert(line):
        feature = convert_single_example(0, line, max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids],(batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, max_seq_length))
        return input_ids, input_mask, segment_ids

    with graph.as_default():
        input_ids_p, input_mask_p, segment_ids_p = convert(sentence)
        feed_dict = {
            input_ids: input_ids_p,
            input_mask: input_mask_p,
            segment_ids: segment_ids_p,
        }
        intent_pred, slot_pred = sess.run([intent_predict, slot_predict], feed_dict=feed_dict)
        print(f"intent_pred: {intent_pred[1][0]}, slot_pred: {slot_pred[1][0].tolist()}")
        # intent_predict_id = tf.argmax(intent_prob, axis=-1, output_type=tf.int32)
        intent = id2intentlabel[intent_pred[1][0]]
        entities = convert_id_to_entity(sentence, slot_pred[1][0].tolist(), id2nerlabel)

        return intent, entities

def convert_id_to_entity(sentence, pred_ids_result, idx2label):
    """
    :param sentence: 命名实体识别的句子
    :param pred_ids_result: 预测序列ID
    :param idx2label: ID到label的映射
    :return: 命名实体的list
    """
    curr_seq, entities = [], []
    curr_entity = ""
    for ids in pred_ids_result:
        if ids == 0:   # [pad]
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


if __name__ == "__main__":
    sentence = input()
    intent, entities = joint_predict(sentence)   
    print(f"intent: {intent}")
    print(f"entities: {entities}")     

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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

is_training=False
use_one_hot_embeddings=False
batch_size=1

vocab_file="../chinese_L-12_H-768_A-12/vocab.txt"
bert_config_file="../chinese_L-12_H-768_A-12/bert_config.json"
init_checkpoint="../chinese_L-12_H-768_A-12/bert_model.ckpt"
max_seq_length=64
output_dir="./output/qa_dir"
do_lower_case=True

graph = tf.Graph()
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess=tf.Session(config=gpu_config, graph=graph)
model=None

# global graph

input_ids_p, input_mask_p, label_ids_p, segment_ids_p = None, None, None, None
print(output_dir)
print('checkpoint path:{}'.format(os.path.join(output_dir, "checkpoint")))
if not os.path.exists(os.path.join(output_dir, "checkpoint")):
    raise Exception("failed to get checkpoint. going to return ")

num_labels = 2


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

    output_layer = model.get_pooled_output()    # [batch_size, hidden_size]

    hidden_size = output_layer.shape[-1].value

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        linear = tf.keras.layers.Dense(num_labels,activation=None)
        logits = linear(output_layer)
        logits = tf.reshape(logits,[-1,num_labels])
        probabilities = tf.nn.softmax(logits, axis=-1)

        return probabilities

with graph.as_default():
    print("going to restore checkpoint")
    input_ids_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_ids")
    input_mask_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="input_mask")
    segment_ids_p = tf.placeholder(tf.int32, [batch_size, max_seq_length], name="segment_ids")

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    pred_ids = create_model(
        bert_config, is_training, input_ids_p, input_mask_p, segment_ids_p, num_labels, use_one_hot_embeddings)

    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(output_dir))


tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(ex_index, example1, example2, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    tokens_a = example1
    tokens_b = example2
    
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

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        # label_id=label_id,
        is_real_example=True)
    return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()




def qa_predict(sentence, predicate):

    def convert(line1, line2):
        feature = convert_single_example(0, line1, line2, max_seq_length, tokenizer)
        input_ids = np.reshape([feature.input_ids],(batch_size, max_seq_length))
        input_mask = np.reshape([feature.input_mask],(batch_size, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids],(batch_size, max_seq_length))
        return input_ids, input_mask, segment_ids

    global graph
    with graph.as_default():
        if len(sentence) < 2:
            return sentence
        
        sentence = tokenizer.tokenize(sentence)
        predicate = tokenizer.tokenize(predicate)
        input_ids, input_mask, segment_ids = convert(sentence, predicate)

        feed_dict = {input_ids_p: input_ids,
                    input_mask_p: input_mask,
                    segment_ids_p: segment_ids}
        
        pred_ids_result = sess.run([pred_ids], feed_dict)

        return pred_ids_result[0][0][1]

if __name__ == "__main__":
    while 1:
        print("please input the sentence:")
        sentence = str(input())
        if sentence == "end":
            break
        print("\nplease input the entity:")
        entity = str(input())
        probability = qa_predict(sentence, entity)
        print(f"\nthe similarity between sentence and entity is: {probability}")



import pandas as pd
import numpy as np

import json

import tensorflow as tf
from tensorflow.keras import preprocessing

from kobert_transformers import get_tokenizer

tokenizer = get_tokenizer()

with open('./kr_abstract_data/001.문서요약텍스트_sample/원시데이터/신문기사/sample_original.json', 'r') as json_file:
    json_data = json.load(json_file)

train_df = pd.DataFrame(json_data)
train_df.head()
train_df.columns

train_df['extractive'] # ground_truth

sum(train_df['extractive'], [])

ext_label = np.array(train_df['extractive'].tolist())
    
def get_input_ids(token):
    return sum(token['input_ids'], [])
    
def get_token_type_ids(token):
    return sum([(np.array(y) + 1).tolist() if (x % 2 == 1) else y for x, y in enumerate(token['token_type_ids'])], [])

def get_attention_mask(token):
    return sum(token['attention_mask'], [])

train_df['token'] = train_df['text'].map(lambda x: tokenizer([z['sentence'] for z in  x]))

def get_ids(row):
    input_ids = get_input_ids(row['token'])
    token_type_ids = get_token_type_ids(row['token'])
    attention_mask = get_attention_mask(row['token'])
    return pd.Series([input_ids, token_type_ids, attention_mask])

train_df[['input_ids', 'token_type_ids', 'attention_mask']] = train_df.apply(get_ids, axis=1)

def get_cls_index(input_ids):
    return np.where(np.array(input_ids) == 2)[0].tolist()

train_df['cls_index'] = train_df['input_ids'].map(lambda x: get_cls_index(x))

MAX_PAD_LENGTH = 512

ii_list = train_df['input_ids'].tolist()

idx = np.where(np.array([len(x) for x in ii_list]) < 512)[0]

padded_ii = preprocessing.sequence.pad_sequences(ii_list,
                                     maxlen=MAX_PAD_LENGTH,
                                     padding='post')

tti_list = train_df['token_type_ids'].tolist()

padded_tti = preprocessing.sequence.pad_sequences(tti_list,
                                     maxlen=MAX_PAD_LENGTH,
                                     padding='post')

am_list = train_df['attention_mask'].tolist()

padded_am = preprocessing.sequence.pad_sequences(am_list,
                                     maxlen=MAX_PAD_LENGTH,
                                     padding='post')

ci_list = [x for x, y in zip(train_df['cls_index'].tolist(), ii_list) if (len(y) < 512)]

padded_ci = preprocessing.sequence.pad_sequences(ci_list,
                                     maxlen=MAX_PAD_LENGTH,
                                     padding='post',
                                     value=512)

ci_one_hot = tf.one_hot(padded_ci, MAX_PAD_LENGTH)

ci_arr = tf.reduce_sum(ci_one_hot, axis=1)

MAX_CLS_PAD = max(len(x) for x in ci_list)

li_list = train_df['extractive'].tolist()

MAX_LABEL_LENGTH = max(sum(li_list, [])) + 1

ground_truth_idx = [[ 1 if (x in y) else 0 for x in z] for y,z in zip(ext_label[idx].tolist(), [ np.arange(len(x)).tolist() for x in ci_list])]

sample_data = tf.data.Dataset.from_tensor_slices((np.array(padded_ii[idx]), np.array(padded_tti[idx]), np.array(padded_am[idx]), ci_arr, tf.ragged.constant(ci_list), tf.ragged.constant(ground_truth_idx)))
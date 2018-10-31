from os.path import join as pjoin
import numpy as np
import json
import tensorflow as tf
import logging

PAD_ID = 0


##def get_indicies_sorted_by_context_length(data):
def reindex_by_length(data):
    context_lengths = np.array([len(context) for context in data["context"]])
    return np.argsort(context_lengths)


def dictionary_indexes(data, indices):
    for k, v in data.items():
        data[k] = v[indices]
    return data


def preprocess(data_dir):
    # Load the training data
    train = {}

    ids_context = [list(map(int, line.strip().split()))
                     for line in open(pjoin(data_dir, "train.ids.context"), encoding="utf8")]
	
    train["context"] = np.array(ids_context)

    ids_question = [list(map(int, line.strip().split()))
                      for line in open(pjoin(data_dir, "train.ids.question"), encoding="utf8")]
					  
    train["question"] = np.array(ids_question)

    context = [line for line in open(pjoin(data_dir, "train.context"), encoding="utf8")]
	
    train["word_context"] = np.array(context)
    
    span = [list(map(int, line.strip().split()))
                         for line in open(pjoin(data_dir, "train.span"), encoding="utf8")]
  
    train["answer_span_start"] = np.array(span)[:, 0]
    train["answer_span_end"] = np.array(span)[:, 1]

    train_indicies = reindex_by_length(train)
    train = dictionary_indexes(train, train_indicies)

    # Load the val data
    validate = {}

    ids_context = [list(map(int, line.strip().split()))
                   for line in open(pjoin(data_dir, "val.ids.context"), encoding="utf8")]
    
    validate["context"] = np.array(ids_context)
	
    ids_question = [list(map(int, line.strip().split()))
                    for line in open(pjoin(data_dir, "val.ids.question"), encoding="utf8")]
    
    validate["question"] = np.array(ids_question)
	
    context = [line for line in open(pjoin(data_dir, "val.context"), encoding="utf8")]
	
    validate["word_context"] = np.array(context)
	
    span = [list(map(int, line.strip().split()))
                       for line in open(pjoin(data_dir, "val.span"), encoding="utf8")]
    
    validate["answer_span_start"] = np.array(span)[:, 0]
    validate["answer_span_end"] = np.array(span)[:, 1]
	
    answer = [line for line in open(pjoin(data_dir, "val.answer"), encoding="utf8")]

    validate["word_answer"] = np.array(answer)

    indicies = reindex_by_length(validate)
    validate = dictionary_indexes(validate, indicies)

    return train, validate


def glove_vectors(data_dir):
    return np.load(pjoin(data_dir, "glove.trimmed.100.npz"))["glove"].astype(np.float32)

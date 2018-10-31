import time
import sys
import numpy as np 
import logging
import matplotlib.pyplot as plt
import matplotlib
from os.path import join as pjoin
matplotlib.style.use('ggplot')

logging.basicConfig(level = logging.INFO)
PAD_ID = 0

def save_graphs(data, path):

    # First plot the losses
    losses = data["losses"]
    fig = plt.figure()
    plt.plot([i for i in range(len(losses))], losses)
    plt.title("Batch sized used: {}".format(data["batch_size"]))
    plt.xlabel('batch number', fontsize=18)
    plt.ylabel('average loss', fontsize=16)
    fig.savefig(pjoin(path, 'loss.png'))
    plt.close(fig)

    batch_indices = data["batch_indices"]

    # Now plot the f1, EM for the training and validation sets
    f1_train, f1_val = data["f1_train"], data["f1_val"]

    fig = plt.figure()
    plt.plot(batch_indices, f1_train, 'b', batch_indices, f1_val, 'r')
    plt.title("Batch sized used: {}".format(data["batch_size"]))
    plt.xlabel('batch number', fontsize=18)
    plt.ylabel('F1 Score', fontsize = 16)
    fig.savefig(pjoin(path, "f1_scores.png"))
    plt.close(fig)

    EM_train, EM_val = data["EM_train"], data["EM_val"]

    fig = plt.figure()
    plt.plot(batch_indices, EM_train, 'b', batch_indices, EM_val, 'r')
    plt.title("Batch sized used: {}".format(data["batch_size"]))
    plt.xlabel('batch number', fontsize=18)
    plt.ylabel('EM Score', fontsize = 16)
    fig.savefig(pjoin(path, "EM_scores.png"))
    plt.close(fig)

def find_best_span(start, end):
    """
    start: (BS, MCL) tensor
    end: (BS, MCL) tensor
    """
    batch_size = start.shape[0]
    start_index = []
    end_index = []
    start = softmax(start)
    end = softmax(end)
    for i in range(batch_size):
        best_prob = 0.
        best_span = (0, 0)
        start_logit = start[i]
        end_logit = end[i]
        for j, start_prob in enumerate(start_logit):
            for k, end_prob in enumerate(end_logit[j:]):
                span_prob = start_prob * end_prob
                if span_prob > best_prob:
                    # print("previous best_prob: {}".format(best_prob))
                    # print("current best_prob: {}".format(span_prob))
                    best_span = (j, j + k)
                    best_prob = span_prob
        start_index.append(best_span[0])
        end_index.append(best_span[1])

    return(start_index, end_index)
                


def softmax(x):
    if len(x.shape) > 1:
        # Matrix
        max_vals = np.expand_dims(np.max(x, axis = 1), 0).T
        x = x - max_vals
        x = np.exp(x)
        x_col_sums = np.expand_dims(np.sum(x, axis = 1), 0).T
        x = x/x_col_sums
        
    else:
        # Vector
        max_val = np.max(x)
        x = x - max_val
        x = np.exp(x)
        x_sum = np.sum(x)
        x = x/x_sum
    return(x)


def pad_sequences(sequences, max_sequence_length):

    if max_sequence_length is None:
        max_sequence_length = max([len(sequence) for sequence in sequences])

    padded_sequences = []
    sequences_mask = []

    for s in sequences:
        padded_sequence = s[:max_sequence_length] 
        sequence_mask = [True for _ in padded_sequence]
        
        while len(padded_sequence) < max_sequence_length: 
            padded_sequence.append(PAD_ID)
            sequence_mask.append(False)

        padded_sequences.append(padded_sequence)
        sequences_mask.append(sequence_mask)
    return(padded_sequences, sequences_mask, max_sequence_length)

def batches(data, is_train = True, batch_size = 24, window_size = 3, shuffle = True):
    
    n_samples = len(data["context"])
    n_buckets = n_samples//batch_size + 1
    logging.debug("Number of samples: {}".format(n_samples))
    logging.debug("Number of buckets: {}".format(n_buckets))
    # If it is training then we simply get a window of batch_size * 3 and randomly sample 
    # Since the dataset is already sorted WRT context length, this will make the training a lot faster

    if is_train:
        batch_indices = np.arange(n_buckets)

        # Create batches that are of the same length
        if shuffle:
            np.random.shuffle(batch_indices)


        for i in batch_indices[::-1]:

            start = i * batch_size
            end = min(start + batch_size * window_size, n_samples)
            # logging.debug("start: {}".format(start))
            # logging.debug("end: {}".format(end))
            window = [i for i in range(start, end)]
            # logging.debug("window size: {}".format(len(window)))
            # logging.debug("batch_size: {}".format(batch_size))
            indices = np.random.choice(window, min(len(window), batch_size), replace=False)
            # logging.debug("selected indices size: {}".format(len(indices)))
            # logging.debug(indices)
            ret = {}
            for k, v in data.items():
                ret[k] = v[indices]
            yield ret
    else:
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, n_buckets * batch_size, batch_size):
            ret = {}
            start = i
            end = i + batch_size
            for k, v in data.items():
                ret[k] = v[indices[start:end]]
            yield ret

def get_random_samples(data, num_samples):  
    total_sample_num = len(data["context"])
    indices = np.random.choice(np.arange(total_sample_num), num_samples, replace = False)
    # logging.debug("Number of indices selected: {}".format(len(indices)))
    ret = {}
    for k, v in data.items():
        ret[k] = v[indices]
    return(ret)
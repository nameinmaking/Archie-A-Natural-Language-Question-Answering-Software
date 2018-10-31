#dependency tensorflow, matplotlib, google, protobuf 3.6.0, numpy

import os
from os.path import join as pjoin
import json
import datetime
import tensorflow as tf
from utils.data_loader import preprocess, glove_vectors
from utils.Save_Result import Save_Result
import logging
#from models.BiDAF import BiDAF
from models.encoderdecoder import encoderdecoder
#from models.Attention import LuongAttention


logging.basicConfig(level=logging.INFO)

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 96, "batch size")
flags.DEFINE_integer("eval_num", 250, "Number of Evaluations")
flags.DEFINE_integer("embedding_size", 100, "embedding size")
flags.DEFINE_integer("window_size", 4, "Training window")
flags.DEFINE_integer("hidden_size", 100, "RNNs size")
flags.DEFINE_integer("samples_used_for_evaluation", 500,"Samples to be used at evaluation for every eval_num batches trained")
flags.DEFINE_integer("num_epochs", 1, "Number of Epochs")
flags.DEFINE_integer("max_context_length", None, "Maximum length for the context")
flags.DEFINE_integer("max_question_length", None, "Maximum length for the question")

flags.DEFINE_string("data_dir", "data/squad", "Data directory")
flags.DEFINE_string("train_dir", "train/chkpoint/", "Saved training parameters directory")
flags.DEFINE_string("retrain_embeddings", False, "Whether or not to retrain the embeddings")
flags.DEFINE_string("share_encoder_weights", False, "Whether or not to share the encoder weights")
flags.DEFINE_string("learning_rate_annealing", False, "Whether or not to anneal the learning rate")
flags.DEFINE_string("ema_for_weights", False, "Whether or not to use EMA for weights")
flags.DEFINE_string("log", True, "Whether or not to log the metrics during training")
flags.DEFINE_string("optimizer", "adam", "The optimizer to be used ")
flags.DEFINE_string("model", "BiDAF", "Model type")
flags.DEFINE_string("find_best_span", True, "Whether find the span with the highest probability")


flags.DEFINE_float("learning_rate", 0.048, "Learning rate")
flags.DEFINE_float("keep_prob", 0.75, "The probably that a node is kept after the affine transform")
flags.DEFINE_float("max_grad_norm", 5.,"The maximum grad norm during backpropagation, anything greater than max_grad_norm is truncated to be max_grad_norm")

flags = flags.FLAGS


def initialize_model(session, train_path):
    #if not os.path.exists(train_path):
        session.run(tf.global_variables_initializer())
        os.makedirs(train_path, exist_ok=True)
        print(train_path)
        
        #open a file and save the configuration
        with open(pjoin(train_path, "configuration.txt"), "w") as f:
            output = ""
            for k, v in flags.__flags.items():
              output += "{} : {}\n".format(k, v)
            f.write(output)
    #else:
     #   saver = tf.train.Saver()
     #   checkpoint = tf.train.get_checkpoint_state(train_path)
        #saver.restore(session, checkpoint.model_checkpoint_path)

def main(_):
    #Call the pre process function to get data
    training, validation = preprocess(flags.data_dir)

    #Use the precomputed Glove vectors
    embeddings = glove_vectors(flags.data_dir)

    #Save the result to a file
    results = Save_Result(flags.train_dir)

    with tf.device("/cpu:0"):
        model = encoderdecoder(results,embeddings,flags)

    logging.info("This is the Beginning of the Training step")

    #print(vars(flags)["__flags"])

    with tf.Session() as sess:
        initialize_model(sess, flags.train_dir)
        model.train(sess, training, validation)


if __name__ == "__main__":
    tf.app.run()

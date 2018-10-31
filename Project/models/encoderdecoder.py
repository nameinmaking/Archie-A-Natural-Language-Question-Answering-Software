import logging
from utils.general import pad_sequences
from utils.model import processing_logit_softmax, processing_logit, BiLSTM, get_optimizer
from models.model import Model
import tensorflow as tf



logging.basicConfig(level=logging.INFO)


class Encoder(object):
    def __init__(self, size):
        self.size = size

    def encode(self, inputs, masks, initial_state_fw=None, initial_state_bw=None, dropout=1.0, reuse=False):
        # The character level embeddings
        # if self.config.char_level_embeddings:
        #     char_level_embeddings = self._character_level_embeddings(inputs, filter_sizes, num_filters, dropout)


        # The contextual level embeddings
        output_concat, (final_state_fw, final_state_bw) = BiLSTM(inputs, masks, self.size, initial_state_fw,
                                                                 initial_state_bw, dropout, reuse)
        logging.debug("output shape: {}".format(output_concat.get_shape()))

        return (output_concat, (final_state_fw, final_state_bw))



class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, inputs, mask, max_input_length, dropout):

        with tf.variable_scope("start"):
            start = processing_logit(inputs, max_input_length)
            start = processing_logit_softmax(start, mask)

        with tf.variable_scope("end"):
            end = processing_logit(inputs, max_input_length)
            end = processing_logit_softmax(end, mask)

        return (start, end)


class encoderdecoder(Model):
    def __init__(self, results, embeddings, config):
        self.embeddings = embeddings
        self.config = config
        self.encoder = Encoder(config.hidden_size)
        self.decoder = Decoder(config.hidden_size)
        self.add_placeholders()

        with tf.variable_scope("encoderdecoder", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.question_embeddings, self.context_embeddings = self.setup_embeddings()
            self.build(config = config, results = results)

    def add_placeholders(self):
        self.context_ph = tf.placeholder(tf.int32, shape=(None, None))
        self.context_mask_ph = tf.placeholder(tf.bool, shape=(None, None))
        self.question_ph = tf.placeholder(tf.int32, shape=(None, None))
        self.question_mask_ph = tf.placeholder(tf.bool, shape=(None, None))
        self.answer_start_ph = tf.placeholder(tf.int32,shape=(None))
        self.answer_nd_ph = tf.placeholder(tf.int32,shape=(None))
        self.max_context_length_ph = tf.placeholder(tf.int32)
        self.max_question_length_ph = tf.placeholder(tf.int32)
        self.dropout_ph = tf.placeholder(tf.float32)

    def setup_embeddings(self):
        with tf.variable_scope("embeddings"):
            if self.config.retrain_embeddings:
                embeddings = tf.get_variable("embeddings", initializer=self.embeddings)
            else:
                embeddings = tf.cast(self.embeddings, dtype=tf.float32)

            question_embeddings = self._embedding_lookup(embeddings, self.question_ph,
                                                         self.max_question_length_ph)
            context_embeddings = self._embedding_lookup(embeddings, self.context_ph,
                                                        self.max_context_length_ph)
        return question_embeddings, context_embeddings

    def _embedding_lookup(self, embeddings, indicies, max_length):
        embeddings = tf.nn.embedding_lookup(embeddings, indicies)
        embeddings = tf.reshape(embeddings, shape=[-1, max_length, self.config.embedding_size])
        return embeddings

    def add_preds_op(self):

        # First we set up the encoding 

        logging.info(("-" * 10, "ENCODING ", "-" * 10))
        with tf.variable_scope("q"):
            Hq, (q_final_state_fw, q_final_state_bw) = self.encoder.encode(self.question_embeddings,
                                                                           self.question_mask_ph,
                                                                           dropout=self.dropout_ph)

            if self.config.share_encoder_weights:
                Hc, (c_final_state_fw,  c_final_state_bw) = self.encoder.encode(self.context_embeddings,
                                                                               self.context_mask_ph,
                                                                               initial_state_fw=q_final_state_fw,
                                                                               initial_state_bw=q_final_state_bw,
                                                                               dropout=self.dropout_ph,
                                                                               reuse=True)
            else:
                with tf.variable_scope("c"):
                    Hc, (c_final_state_fw, c_final_state_bw) = self.encoder.encode(self.context_embeddings,
                                                                                   self.context_mask_ph,
                                                                                   initial_state_fw=q_final_state_fw,
                                                                                   initial_state_bw=q_final_state_bw,
                                                                                   dropout=self.dropout_ph)

        logging.info(("-" * 10, " DECODING ", "-" * 10))
        with tf.variable_scope("decoding"):
            start, end = self.decoder.decode(Hc, self.context_mask_ph,
                                             self.max_context_length_ph, self.dropout_ph)
        return start, end

    def add_loss_op(self, preds):
        with tf.variable_scope("loss"):
            answer_start_one_hot = tf.one_hot(self.answer_start_ph, self.max_context_length_ph)
            answer_end_one_hot = tf.one_hot(self.answer_nd_ph, self.max_context_length_ph)
            logging.info("answer_start_one_hot.get_shape() {}".format(answer_start_one_hot.get_shape()))
            logging.info("answer_end_one_hot.get_shape() {}".format(answer_end_one_hot.get_shape()))

            start, end = preds
            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=start, labels=answer_start_one_hot))
            loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=end, labels=answer_end_one_hot))
            loss = loss1 + loss2
        return loss

    def add_training_op(self, loss):
        variables = tf.trainable_variables()
        gradients = tf.gradients(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)

        if self.config.learning_rate_annealing:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.config.learning_rate, global_step, 1250, 0.96,
                                                       staircase=False)
            global_step = tf.add(1, global_step)
        else:
            learning_rate = self.config.learning_rate

        optimizer = get_optimizer(self.config.optimizer, learning_rate)
        train_op = optimizer.apply_gradients(zip(gradients, variables))

        # For applying EMA for trained parameters

        if self.config.ema_for_weights:
            ema = tf.train.ExponentialMovingAverage(0.999)
            ema_op = ema.apply(variables)

            with tf.control_dependencies([train_op]):
                train_op = tf.group(ema_op)

        return train_op

    def create_feed_dict(self, context, question, answer_start_batch=None, answer_end_batch=None,
                         is_train=True):


        context_batch, context_mask, max_context_length = pad_sequences(context,
                                                                        max_sequence_length=self.config.max_context_length)
        question_batch, question_mask, max_question_length = pad_sequences(question,
                                                                           max_sequence_length=self.config.max_question_length)

        feed_dict = {self.context_ph: context_batch,
                     self.context_mask_ph: context_mask,
                     self.question_ph: question_batch,
                     self.question_mask_ph: question_mask,
                     self.max_context_length_ph: max_context_length,
                     self.max_question_length_ph: max_question_length}

        if is_train:
            feed_dict[self.dropout_ph] = self.config.keep_prob
        else:
            feed_dict[self.dropout_ph] = 1.0

        if answer_start_batch is not None and answer_end_batch is not None:
            feed_dict[self.answer_start_ph] = answer_start_batch
            feed_dict[self.answer_nd_ph] = answer_end_batch

        return feed_dict

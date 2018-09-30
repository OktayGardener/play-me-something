import logging

import tensorflow as tf
from pms.graph.context_embeddings import ContextEmbedding

from pms.util import lazy_property

log = logging.getLogger(__name__)


class ContextRNNModel:
    COUNTRY_EMBEDDING_SIZE = 8
    CONTEXT_EMBEDDING_SIZE = 40
    USER_AGE_NORMALIZATION_COEF = 100.0
    CONTEXT_AGE_NORMALIZATION_COEF = 2 * 365.0 * 5.0
    TIME_NORMALIZATION_COEF = 100000000

    def __init__(
            self, example, country_vocabulary_size, context_vocabulary_size, history_size,
            embedding_size
    ):
        self.example = example
        self.country_embeddings = country_vocabulary_size
        self.history_embeddings = context_vocabulary_size
        self.history_size = history_size

        self.country_embeddings = tf.Variable(
            name='country_embeddings',
            initial_value=tf.random_uniform([country_vocabulary_size, self.COUNTRY_EMBEDDING_SIZE],
                                            -1.0, 1.0)
        )

        self.history_embeddings = tf.Variable(
            name='context_embeddings',
            initial_value=tf.random_uniform([context_vocabulary_size, self.CONTEXT_EMBEDDING_SIZE],
                                            -1.0, 1.0)
        )

        self.layer1_weights = tf.Variable(
            name='layer1_weights',
            initial_value=tf.random_uniform([
                self.stacked_query.get_shape().as_list()[0], embedding_size
            ], -1.0, 1.0)
        )

        self.layer1_biases = tf.Variable(
            name='layer1_biases', initial_value=tf.constant(0.1, shape=[embedding_size])
        )

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.average_loss = tf.placeholder(shape=[], dtype=tf.float64)
        self.playlists_per_sec = tf.placeholder(shape=[], dtype=tf.float64)

        self.context_embedding = ContextEmbedding(
            self.query_repr,
            example,
            context_vocabulary_size=context_vocabulary_size,
            embedding_size=embedding_size,
            history_size=history_size
        )

        self.loss = self.context_embedding.nce_loss_64

    @lazy_property
    def normalized_embeddings(self):
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.history_embeddings), 1, keep_dims=True))
        return self.history_embeddings / norm

    @lazy_property
    def history_repr(self):
        embed = tf.nn.embedding_lookup(self.history_embeddings, self.example.history)
        expanded_embed = tf.expand_dims(embed, 0)
        _output, state = tf.nn.dynamic_rnn(self.gru, expanded_embed, dtype=tf.float32)
        final_output = tf.gather(tf.squeeze(_output, 0), tf.shape(self.example.history)[0] - 1)
        return final_output

    # RNN Specific
    @lazy_property
    def gru(self):
        MEM_STATE_SIZE = 128
        return tf.contrib.rnn.GRUCell(MEM_STATE_SIZE)

    @lazy_property
    def time_of_day_vector(self):
        return tf.expand_dims(self.example.time_of_day, 0)

    @lazy_property
    def time_of_week_vector(self):
        return tf.expand_dims(self.example.time_of_week, 0)

    @lazy_property
    def stacked_query(self):
        country_embd = tf.nn.embedding_lookup(self.country_embeddings, self.example.country)
        return tf.concat([
            self.history_repr, country_embd, self.time_of_day_vector, self.time_of_week_vector
        ], 0)

    @lazy_property
    def query_repr(self):
        return tf.squeeze(
            tf.nn.relu(
                tf.matmul(tf.expand_dims(self.stacked_query, 0), self.layer1_weights) +
                self.layer1_biases
            )
        )

    @lazy_property
    def minimizer(self):
        """Construct the minimizer operation"""
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        return optimizer.minimize(self.loss, global_step=self.global_step)

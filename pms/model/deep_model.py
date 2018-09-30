import logging

import math
import tensorflow as tf
from pms.graph.context_embeddings import ContextEmbedding
from pms.util import lazy_property, monitor_tensor

log = logging.getLogger(__name__)


class ContextModel:
    COUNTRY_EMBEDDING_SIZE = 8
    CONTEXT_EMBEDDING_SIZE = 64
    USER_AGE_NORMALIZATION_COEF = 100.0
    CONTEXT_AGE_NORMALIZATION_COEF = 2 * 365.0 * 5.0
    TIME_NORMALIZATION_COEF = 100000000
    USER_AGE_NORMALISATION_COEF = 100.0

    def __init__(
            self, example, country_vocabulary_size, city_vocabulary_size, context_vocabulary_size,
            history_size, embedding_size
    ):
        self.example = example
        self.country_embeddings = country_vocabulary_size
        self.history_embeddings = context_vocabulary_size
        self.city_embeddings = city_vocabulary_size
        self.history_size = history_size
        self.embedding_size = embedding_size

        self.country_embeddings = tf.Variable(
            name='country_embeddings',
            initial_value=tf.random_uniform([country_vocabulary_size, self.COUNTRY_EMBEDDING_SIZE],
                                            -1.0, 1.0)
        )  # 60 x 8

        self.city_embeddings = tf.Variable(
            name='city_embeddings',
            initial_value=tf.random_uniform([city_vocabulary_size, self.COUNTRY_EMBEDDING_SIZE],
                                            -1.0, 1.0)
        )  # 60 x 8

        self.history_embeddings = tf.Variable(
            name='context_embeddings',
            initial_value=tf.random_uniform([context_vocabulary_size, self.CONTEXT_EMBEDDING_SIZE],
                                            -1.0, 1.0)
        )  # 100 000 x 64

        self.layer_weights = []
        self.layer_biases = []
        self.dimensions = []
        self.create_deep_layers(self.dimensions)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.average_loss = tf.placeholder(shape=[], dtype=tf.float64)
        self.examples_per_sec = tf.placeholder(shape=[], dtype=tf.float64)

        self.context_embedding = ContextEmbedding(
            self.query_repr,
            example,
            context_vocabulary_size=context_vocabulary_size,
            embedding_size=embedding_size,
            history_size=history_size
        )

        self.loss = self.context_embedding.sampled_softmax_loss_64

    def create_deep_layers(self, dimensions):
        """ Create deep layers with the given dimensions """
        input_dim = self.stacked_query.get_shape().as_list()[1]
        for dim in dimensions + [self.embedding_size]:
            self.layer_weights.append(
                tf.Variable(
                    initial_value=tf.truncated_normal([input_dim, dim],
                                                      stddev=1.0 / math.sqrt(input_dim))
                )
            )

            self.layer_biases.append(tf.Variable(initial_value=tf.constant(0.1, shape=[dim])))

            input_dim = dim

    @lazy_property
    def normalized_embeddings(self):
        norm = tf.sqrt(
            tf.reduce_sum(tf.square(self.context_embedding.context_vectors), 1, keep_dims=True)
        )
        return self.context_embedding.context_vectors / norm

    @lazy_property
    def time_of_day_vector(self):
        return tf.expand_dims(self.example.time_of_day, 1)

    @lazy_property
    def time_of_week_vector(self):
        return tf.expand_dims(self.example.time_of_week, 1)

    @lazy_property
    def user_gender_vector(self):
        user_gender = tf.cast(self.example.gender, tf.float32)
        return tf.expand_dims(user_gender / 2.0, 1)

    @lazy_property
    def user_platform_vector(self):
        user_platform = tf.cast(self.example.platform, tf.float32)
        return tf.expand_dims(user_platform, 1)

    @lazy_property
    def user_age_vector(self):
        user_age = tf.cast(self.example.age, tf.float32)
        return tf.expand_dims(user_age / self.USER_AGE_NORMALISATION_COEF, 1)

    @lazy_property
    def history_repr(self):
        embed = tf.nn.embedding_lookup(self.history_embeddings, self.example.history)
        return tf.reduce_mean(embed, reduction_indices=1)

    @lazy_property
    def stacked_query(self):
        country_embd = tf.nn.embedding_lookup(self.country_embeddings, self.example.country)
        city_embd = tf.nn.embedding_lookup(self.city_embeddings, self.example.city)

        return tf.concat([
            self.history_repr, country_embd, city_embd, self.user_age_vector,
            self.user_gender_vector, self.user_platform_vector, self.time_of_day_vector,
            self.time_of_week_vector
        ], 1)

    @lazy_property
    def query_repr(self):
        vec = self.stacked_query
        for i in range(len(self.dimensions) + 1):
            vec = tf.nn.relu(tf.matmul(vec, self.layer_weights[i]) + self.layer_biases[i])
            monitor_tensor('layer_%s' % i, vec)
        return vec

    @lazy_property
    def minimizer(self):
        """Construct the minimizer operation"""
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        return optimizer.minimize(self.loss, global_step=self.global_step)

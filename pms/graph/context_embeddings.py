from pms.util import lazy_property
import math
import tensorflow as tf


class ContextEmbedding:
    def __init__(self, query_repr, example, context_vocabulary_size, embedding_size, history_size):
        self.query_repr = query_repr
        self.example = example
        self.context_vocabulary_size = context_vocabulary_size
        self.embedding_size = embedding_size
        self.history_size = history_size

        self.context_vectors = tf.Variable(
            name='context_vectors',
            initial_value=tf.truncated_normal([context_vocabulary_size, embedding_size],
                                              stddev=1.0 / math.sqrt(embedding_size))
        )

        self.context_biases = tf.Variable(
            name='context_biases', initial_value=tf.constant(0.1, shape=[context_vocabulary_size])
        )

        # Initialize the graph
        _ = self.similarity

        # Add summaries
        tf.summary.scalar('nce_loss_64', self.nce_loss_64)
        tf.summary.scalar('sampled_softmax_64_loss', self.sampled_softmax_loss_64)
        tf.summary.scalar('sampled_softmax_1024_loss', self.sampled_softmax_loss_1024)

    @lazy_property
    def tiled_query_repr(self):
        tiled = tf.tile(self.query_repr, [tf.shape(self.example.history)[0]])
        return tf.reshape(tiled, [-1, self.embedding_size])

    @lazy_property
    def nce_loss_64(self):
        """Compute the average NCE loss for a playlist
            tf.nce_loss automatically draws a new sample of the negative labels each
            time we evaluate the loss.
        """
        return self.make_loss(64, tf.nn.nce_loss)

    @lazy_property
    def sampled_softmax_loss_64(self):
        return self.make_loss(64, tf.nn.sampled_softmax_loss)

    @lazy_property
    def sampled_softmax_loss_1024(self):
        return self.make_loss(1024, tf.nn.sampled_softmax_loss)

    @lazy_property
    def nce_loss_1024(self):
        return self.make_loss(1024, tf.nn.nce_loss)

    def make_loss(self, num_sampled, loss_fn, sampler=None):
        labels = tf.expand_dims(self.example.context, 1)
        num_classes = self.context_vectors.get_shape()[0].value

        return tf.reduce_mean(
            loss_fn(
                self.context_vectors,
                self.context_biases,
                labels,
                self.query_repr,
                num_sampled=num_sampled,
                num_classes=num_classes,
                sampled_values=sampler
            )
        )

    @lazy_property
    def similarity(self):
        return tf.add(
            tf.matmul(self.query_repr, self.context_vectors, transpose_b=True), self.context_biases
        )

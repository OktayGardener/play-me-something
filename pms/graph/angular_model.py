import tensorflow as tf

from pms.util import lazy_property


class AngularModel:
    def __init__(self, model):
        self.model = model

    @lazy_property
    def angular_track_vectors(self):
        """Track vectors mapped to a new space suitable for similarity computation using
            angular distance.
        """
        dim = self.model.track_biases.get_shape()[0].value
        track_biases_transposed = tf.reshape(self.model.track_biases, [dim, 1])
        track_vec_with_bias_dim = tf.concat([self.model.track_vectors, track_biases_transposed], 1)
        # Make sure no vector is longer than 1
        euclidian_norms = tf.sqrt(tf.reduce_sum(tf.square(track_vec_with_bias_dim), 1))
        max_norm = tf.reduce_max(euclidian_norms)
        scaled_track_vec = track_vec_with_bias_dim / max_norm

        # Add normalization dimension to ensure vectors are of length 1.
        norm_squared = tf.reduce_sum(tf.square(scaled_track_vec), 1, keep_dims=True)
        # Handle floating point errors
        norm_squared = tf.minimum(norm_squared, 1)
        normalization_factor = tf.sqrt(1 - norm_squared)
        return tf.concat([scaled_track_vec, normalization_factor], 1)

    @lazy_property
    def query_angular_repr(self):
        return tf.concat([self.model.query_repr, tf.ones([1]), tf.zeros([1])], 0)

    @lazy_property
    def angular_similarity(self):
        return tf.squeeze(
            tf.matmul(
                tf.expand_dims(self.query_angular_repr, 0),
                self.angular_track_vectors,
                transpose_b=True
            ), [0]
        )

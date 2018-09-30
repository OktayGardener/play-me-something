import os
import re
import subprocess

import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from tensorflow.python.lib.io.tf_record import TFRecordOptions

from pms.util import lazy_property, generic_glob

TFRECORD_OPTIONS = TFRecordOptions(compression_type=TFRecordCompressionType.ZLIB)


class TestExample:
    def __init__(self, records_dir, num_epochs, batch_size):
        self.records_dir = records_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    @staticmethod
    def _generate_examples(filename_queue):
        print "creating tfrecord reader to start reading"
        reader = tf.TFRecordReader(options=TFRECORD_OPTIONS)
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                "context": tf.FixedLenFeature([], tf.int64),
                "time_of_day": tf.FixedLenFeature([], tf.float32),
                "time_of_week": tf.FixedLenFeature([], tf.float32),
                "user_country": tf.FixedLenFeature([], tf.int64),
                "user_city": tf.FixedLenFeature([],
                                                tf.int64), "user_age": tf.FixedLenFeature([],
                                                                                          tf.int64),
                "user_gender": tf.FixedLenFeature([], tf.int64),
                "user_platform": tf.FixedLenFeature([], tf.int64),
                "user_id": tf.FixedLenFeature([], tf.string), "history": tf.VarLenFeature(tf.int64)
            }
        )

        return features['context'], features['time_of_day'], features['time_of_week'], \
               features['user_country'], features['user_city'], features['user_age'], \
               features['user_gender'], features['user_platform'], features['user_id'], \
               features['history'].values

    @lazy_property
    def all_fields(self):
        """A single playlist (title/description words and track ids) from input
        :return:
         A word tensor of shape [?] containing all words in the title/description of a playlist
         A track tensor of shape [?] containing all tracks from the corresponding playlist
         A country id scalar contains country id of user who owns playlist
         An age tensor of shape [1] contains age of user who owns playlist
        """

        if self.records_dir:
            print("Records dir found.")
            # Input data.
            filenames = generic_glob(os.path.join(self.records_dir, '*.tfrecords'))
            filename_queue = tf.train.string_input_producer(
                filenames or ['--empty--'], num_epochs=self.num_epochs
            )

            context, time_of_day, time_of_week, user_country, user_city, user_age, \
            user_gender, user_platform, user_id, history = self._generate_examples(filename_queue)

            return tf.train.batch([
                context, time_of_day, time_of_week, user_country, user_city, user_age, user_gender,
                user_platform, user_id, history
            ],
                                  batch_size=self.batch_size,
                                  capacity=64 * 100,
                                  dynamic_pad=True,
                                  allow_smaller_final_batch=True)
        else:
            print("No Records dir found")
            # Note: In prediction tracks is used to represent the candidate set to be scored
            return tf.train.batch([
                tf.placeholder(shape=[], dtype=tf.int32, name='context'),
                tf.placeholder(shape=[], dtype=tf.float32, name='time_of_day'),
                tf.placeholder(shape=[], dtype=tf.float32, name='time_of_week'),
                tf.placeholder(shape=[], dtype=tf.int32, name='user_country'),
                tf.placeholder(shape=[], dtype=tf.int32, name='user_city'),
                tf.placeholder(shape=[], dtype=tf.int32, name='user_age'),
                tf.placeholder(shape=[], dtype=tf.int32, name='user_gender'),
                tf.placeholder(shape=[], dtype=tf.int32, name='user_platform'),
                tf.placeholder(shape=[], dtype=tf.string, name='user_id'),
                tf.placeholder(shape=[None], dtype=tf.int32, name='history')
            ],
                                  batch_size=self.batch_size,
                                  capacity=64 * 100,
                                  dynamic_pad=True,
                                  allow_smaller_final_batch=True)

    @lazy_property
    def context(self):
        return self.all_fields[0]

    @lazy_property
    def time_of_day(self):
        return self.all_fields[1]

    @lazy_property
    def time_of_week(self):
        return self.all_fields[2]

    @lazy_property
    def country(self):
        return self.all_fields[3]

    @lazy_property
    def city(self):
        return self.all_fields[4]

    @lazy_property
    def age(self):
        return self.all_fields[5]

    @lazy_property
    def gender(self):
        return self.all_fields[6]

    @lazy_property
    def platform(self):
        return self.all_fields[7]

    @lazy_property
    def user_id(self):
        return self.all_fields[8]

    @lazy_property
    def history(self):
        return self.all_fields[9]

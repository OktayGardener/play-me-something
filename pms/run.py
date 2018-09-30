import logging
import yaml

import os
import sys
import threading
import time

import numpy as np
import tensorflow as tf
import pms.model
import subprocess
import re
import time

from tensorflow.contrib.tensorboard.plugins import projector

from pms.example import Example
from pms.test_example import TestExample
from pms.defaults import *
from pms.tsv_index import TSVIndex
from pms.util import cached, get_spotify_username

log = logging.getLogger(__name__)

log.info('Tensorflow version: %r', tf.__version__)

flags = tf.app.flags

# Runnable params
flags.DEFINE_boolean('examples', False, 'Print the examples that we are training on')
flags.DEFINE_boolean('train', False, 'Train the model')
flags.DEFINE_boolean('test', False, 'Use testdata as example for the model, for --eval and --demo.')
flags.DEFINE_boolean('eval', False, 'Evaluate the model')
flags.DEFINE_boolean('demo', False, 'Demo the model')

flags.DEFINE_boolean('cloud', False, 'Adapt to training in the cloud')
flags.DEFINE_boolean('stats', False, 'Print statistics')
flags.DEFINE_string('model-class', 'ContextModel', 'Model class name')

# Directories
flags.DEFINE_string('records-dir', None, 'TF Records output directory')
flags.DEFINE_string('test-records-dir', None, 'TF Records output directory')
flags.DEFINE_string('checkpoint-dir', None, 'TF Records output directory')

# Model specific flags
flags.DEFINE_integer('num-epochs', 1, 'Number of epochs')
flags.DEFINE_integer('country-count', DEFAULT_COUNTRY_COUNT,
                     'Maximum number of countries to represent')
flags.DEFINE_integer('context-count', DEFAULT_CONTEXT_COUNT,
                     'Maximum number of contexts to represent')
flags.DEFINE_integer('city-count', DEFAULT_CITY_COUNT,
                     'Maximum number of cities to represent')

flags.DEFINE_integer('embedding-size', 85, 'Context embedding size')
flags.DEFINE_integer('batch-size', 77, 'Batch size')

# TSV flags
flags.DEFINE_string('country-tsv', None, 'Country tsv file')
flags.DEFINE_string('city-tsv', None, 'City tsv file')
flags.DEFINE_string('platform-tsv', None, 'Country tsv file')
flags.DEFINE_string('context-tsv', None, 'Context tsv file')

# Training
flags.DEFINE_integer("concurrent-steps", 4, "Number of concurrent training steps.")

# TF
flags.DEFINE_integer("stat-interval", 10, "Statistics interval in seconds")

# Feature flags (for demo/eval)
flags.DEFINE_string('country', 'us', 'User country')
flags.DEFINE_integer('age', 30, 'User age (years)')
flags.DEFINE_string('gender', 'male', 'User gender (male/female)')
flags.DEFINE_integer('playlist-age', 0, 'Playlist age (days)')
flags.DEFINE_boolean('visualize', False, 'Visualize embeddings using TSNE')
GENDER_ENCODING = {'male': 2, 'female': 0}  # TODO: Use TSV
FEATURE_KEYS = ('country', 'age', 'gender')

FLAGS = flags.FLAGS


class UserExample:
    """ Class used for mapping mock data for evaluation of the model """

    def __init__(self, number, context, time_of_day, time_of_week, country, city, age,
                 gender, platform, history):
        self.number = number
        self.context = context
        self.time_of_day = time_of_day
        self.time_of_week = time_of_week
        self.country = country
        self.city = city
        self.age = age
        self.gender = gender
        self.platform = platform
        self.history = history

    def history_name(self):
        return [context_index().name(x) for x in self.history]

    def __str__(self):
        return 'User #%s, gender %d from country %s, city %s, with age %d listened to %s (%s) ' \
               'with platform %s at time %s at the day %s ' \
               'with history \n %s and history_indexes \n %s \n ------------' \
               % (
                   self.number,
                   GENDER_ENCODING.get(self.gender, 1),
                   country_index().name(self.country),
                   city_index().name(self.city),
                   self.age,
                   context_index().name(self.context),
                   self.context,
                   platform_index().name(self.platform),
                   self.time_of_day,
                   self.time_of_week,
                   self.history_name(),
                   self.history
                )


def generate_fake_test_users():
    """ Generate test users using numpy """
    users = []
    for number in xrange(0, 10):
        context = np.random.choice(1000, 1, replace=False)[0]
        time_of_day = np.random.random_sample()
        # applies to this day monday -> 0.3232738095238095, 0.42444940476190474
        time_of_week = np.random.uniform(0.0, 0.1, 1)[0]
        country = np.random.choice(60, 1, replace=False)[0]
        city = np.random.choice(2999, 1, replace=False)[0]
        age = np.random.choice(100, 1, replace=False)[0]
        gender = np.random.choice(2, 1, replace=False)[0]
        platform = np.random.choice(10, 1, replace=False)[0]
        hist_count = np.random.randint(1, 10, 10)
        for x in hist_count:
            history = [i for i in np.random.choice(1000, x, replace=False)]
            users.append(UserExample(number + 1, context, time_of_day, time_of_week, country,
                                     city, age, gender, platform, history))
    return users


def gen_testdata(model):
    print("Run evaluation on testset")
    with tf.Session() as session:
        init(session)
        restore(session, FLAGS.checkpoint_dir)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=session)
        cnt = 0
        eval_examples = []

        while coord.stop_on_exception:
            try:
                [contexts, time_of_days, time_of_weeks, countries, cities, age, gender,
                 platform, histories] = session.run(
                    [model.example.context,
                     model.example.time_of_day, model.example.time_of_week,
                     model.example.country, model.example.city, model.example.age,
                     model.example.gender, model.example.platform, model.example.history])
                eval_examples += generate_real_test_users(
                    cnt, contexts, time_of_days, time_of_weeks, countries, cities,
                    age, gender, platform, histories
                )
                cnt += len(contexts)
            except tf.errors.OutOfRangeError:
                print("Done parsing testdata from TFRecords.")
                session.close()
                return eval_examples


def generate_real_test_users(num, contexts, time_of_days, time_of_weeks, countries, cities, ages,
                             genders, platforms, histories):
    users = []
    for x in xrange(0, (len(contexts))):
        context = contexts[x]
        time_of_day = time_of_days[x]
        time_of_week = time_of_weeks[x]
        country = countries[x]
        city = cities[x]
        age = ages[x]
        platform = platforms[x]
        gender = genders[x]
        current_history = histories[x]
        added_history = [i for i in current_history if i != 0]
        history = added_history

        users.append(UserExample(num + 1, context, time_of_day, time_of_week, country,
                                 city, age, gender, platform, history))
    return users


class Metrics:
    def __init__(self):
        self.accuracy = 0.0
        self.mean_accuracy = 0.0
        self.reciprocal_rank = 0.0
        self.mean_reciprocal_rank = 0.0
        self.number_of_examples = 0
        self.num_batches = 0

        self.reset_time = time.time()

    def update_accuracy(self):
        self.accuracy += 1
        self.mean_accuracy = float(self.accuracy / self.number_of_examples)

    def update_rr(self, rr):
        self.reciprocal_rank += rr
        self.mean_reciprocal_rank = float(self.reciprocal_rank / self.number_of_examples)

    def update_num(self):
        self.number_of_examples += 1

    def get_mean_reciprocal_rank(self):
        return self.mean_reciprocal_rank

    def get_mean_accuracy(self):
        return self.mean_accuracy

    def get_number_of_examples(self):
        return self.number_of_examples

    def reset(self):
        self.reset_time = time.time()
        self.accuracy = 0.0
        self.mean_accuracy = 0.0
        self.reciprocal_rank = 0.0
        self.mean_reciprocal_rank = 0.0
        self.number_of_examples = 0
        self.num_batches = 0

    def total_time(self):
        return time.time() - self.reset_time

    def __str__(self):
        return '-----------------------------------------------------------' \
                '\n METRICS \nNumber of evaluated examples: %d, \n ' \
                'Current Accuracy: %.5f, \n ' \
                'Current RR: %.5f, \n' \
                'Current Mean Accuracy: %.5f, \n ' \
                'Current Mean RR: %.5f, \n ' \
                '----------------------------------------------------------- \n\n' % (
                    self.number_of_examples,
                    self.accuracy,
                    self.reciprocal_rank,
                    self.mean_accuracy,
                    self.mean_reciprocal_rank
                )


class TrainStats:
    def __init__(self):
        self.loss_sum = 0
        self.loss_cnt = 0
        self.reset_time = time.time()

    def update(self, loss_value):
        self.loss_sum += loss_value
        self.loss_cnt += 1

    def reset(self):
        self.loss_sum = 0
        self.loss_cnt = 0
        self.reset_time = time.time()

    def average_loss(self):
        if self.loss_cnt == 0:
            return 0
        return self.loss_sum / self.loss_cnt

    def examples_per_sec(self):
        return (self.loss_cnt / (time.time() - self.reset_time)) * 77


def feed_dict(example, context, time_of_day, time_of_week, country, city, age, gender,
              platform, history):
    """ Map example to the form of a feed_dict for model overrides """
    return {example.time_of_day: [time_of_day],
            example.time_of_week: [time_of_week],
            example.country: [country],
            example.city: [city],
            example.age: [age],
            example.gender: [gender],
            example.platform: [platform],
            example.history: [history]}


def evaluate_while_training(model, session, metrics):
    print "Start to evalueate %d examples." % len(model.eval_examples)
    random_index = np.random.randint(low=0, high=len(model.eval_examples), size=1000)
    for x in random_index:
        print_similarity(model.eval_examples[x], session, model, metrics, False)
        if metrics.number_of_examples % 200 == 0:
            print_similarity(model.eval_examples[x], session, model, metrics, True)
            print(metrics)


def print_similarity(test_user, session, model, metrics, log, compact=True):
    """ Print the similarity for a given test user """
    feeds = feed_dict(model.example, test_user.context, test_user.time_of_day,
                      test_user.time_of_week,
                      test_user.country, test_user.city, test_user.age, test_user.gender,
                      test_user.platform, test_user.history)

    sim, = session.run([model.context_embedding.similarity], feed_dict=feeds)

    top_k = 100  # if history else 0  # number of nearest neighbors
    nearest = (-sim)[0].argsort()[0:top_k]

    metrics.update_num()
    if test_user.context == nearest[0]:
        metrics.update_accuracy()

    try:
        rr = nearest.tolist().index(test_user.context) + 1
        metrics.update_rr(float(1.0 / rr))
    except ValueError:
        metrics.update_rr(0.0)

    if log:
        log_str = '-----------------------------------------------------------'
        log_str += '\n %s \n Prediction: ' % test_user

        if compact:
            separator = ', '
        else:
            separator = '\n  '
            log_str += '\n  '

        log_str += separator.join(context_index().name(id) for id in nearest)
        log_str += '\n Indexes: '
        log_str += str([id for id in nearest])
        print(log_str)
        print('-----------------------------------------------------------\n\n')

def train_thread(session, minimizer, loss, train_stats, coord):
    """ Main thread for running the training """
    try:
        while True:
            run_options = None
            run_metadata = None

            _, loss_value = session.run([minimizer, loss], options=run_options,
                                        run_metadata=run_metadata)

            train_stats.update(loss_value)
    except Exception as ex:
        log.exception('train_thread failed')
        coord.request_stop(ex)


def setup_tensorboard_embedding(config, path, tensor_name, parameter_name):
    """ Config method for setting up tensorboard embeddings """
    embedding = config.embeddings.add()
    embedding.tensor_name = tensor_name
    if path and tf.gfile.Exists(path):
        embedding.metadata_path = path
    else:
        print('Invalid path for %s: %s -tensorboard visualization will be miss labels' %
              (parameter_name, path))


def setup_tensorboard(config, model, summary_writer):
    """ Config method for creating the visual embeddings in TensorBoard """
    # Setup tensorboard embedding projector
    if hasattr(model, 'country_embeddings'):
        print("setup country embeddings")
        setup_tensorboard_embedding(config, FLAGS.country_tsv, model.country_embeddings.name,
                                    "country-tsv")

    if hasattr(model, 'history_embeddings'):
        print("setup history embeddings")
        print(FLAGS.context_tsv)
        setup_tensorboard_embedding(config, FLAGS.context_tsv, model.history_embeddings.name,
                                    "context-tsv")

    if hasattr(model, 'city_embeddings'):
        print("setup city embeddings")
        print(FLAGS.city_tsv)
        setup_tensorboard_embedding(config, FLAGS.city_tsv, model.city_embeddings.name,
                                    "city-tsv")

    projector.visualize_embeddings(summary_writer, config)


def logging_thread(model, session, train_stats, metrics, coord, summary_op, saver):
    """ Thread for logging desired values during training """
    with coord.stop_on_exception():

        summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir, session.graph, flush_secs=20)
        setup_tensorboard(projector.ProjectorConfig(), model, summary_writer)

        cnt = 0
        while not coord.wait_for_stop(FLAGS.stat_interval):
            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            loss_val, summary, step = session.run(
                [model.loss, summary_op, model.global_step],
                feed_dict={model.average_loss: train_stats.average_loss(),
                           model.examples_per_sec: train_stats.examples_per_sec(),
                           model.mean_reciprocal_rank: model.metrics.get_mean_reciprocal_rank(),
                           model.mean_accuracy: model.metrics.get_mean_accuracy(),
                           model.num_predicted_examples: model.metrics.get_number_of_examples()
                           })

            summary_writer.add_summary(summary, step)

            # The average loss is an estimate of the loss over the last stat_interval
            log.info("Step %s - loss: %f, %d examples/s" %
                     (step, train_stats.average_loss(), train_stats.examples_per_sec()))
            train_stats.reset()
            model.stats.reset()
            # Save model regularly
            cnt += 1
            if cnt % 4 == 0:
                if FLAGS.checkpoint_dir and step > 0:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(session, checkpoint_path, global_step=step)
                    log.info('Saved checkpoint to %s' % checkpoint_path)

                evaluate_while_training(model, session, metrics)


def restore(session, checkpoint_dir):
    """ Restore a model given a directory where the model is stored """
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        print('Restoring model from %s' % checkpoint_dir)
        # loader = tf.train.Saver({'Variable': self.word_embeddings,
        #                          'Variable_1': self.track_vectors,
        #                          'Variable_2': self.track_biases,
        #                          'global_step': self.global_step})
        loader = tf.train.Saver()
        loader.restore(session, checkpoint.model_checkpoint_path)
        return True

    return False


# Begin training
def train(model):
    """ Train the model and spin up logging, training threads """
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
    with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.cloud)) as session:
        tf.summary.scalar('Average_loss', model.average_loss)
        tf.summary.scalar('Playlists_per_second', model.examples_per_sec)
        tf.summary.scalar('Mean_Reciprocal_Rank', model.mean_reciprocal_rank)
        tf.summary.scalar('Mean_Accuracy', model.mean_accuracy)
        tf.summary.scalar('Number_of_Predictions', model.num_predicted_examples)

        summary_op = tf.summary.merge_all()
        init(session)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord)

        log.info("Initialized")

        if FLAGS.checkpoint_dir:
            if not os.path.exists(FLAGS.checkpoint_dir):
                os.makedirs(FLAGS.checkpoint_dir)
            restore(session, FLAGS.checkpoint_dir)

        threads = []
        train_stats = TrainStats()
        metrics = Metrics()
        for _ in xrange(FLAGS.concurrent_steps):
            threads.append(threading.Thread(target=train_thread,
                                            args=[session, model.minimizer, model.loss, train_stats,
                                                  coord]))

        threads.append(threading.Thread(target=logging_thread,
                                        args=[
                                            model, session, train_stats, model.metrics, coord,
                                            summary_op, saver
                                        ]))

        for t in threads:
            t.start()

        coord.join(threads, stop_grace_period_secs=5)


def stats(model):
    with tf.Session() as session:
        init(session)
        restore(session, FLAGS.checkpoint_dir)
        [e, w, b] = session.run([model.history_embeddings, model.context_embedding.context_vectors,
                                 model.context_embedding.context_biases])
        print('-- Top words --')
        for i in xrange(10):
            print('%s - norm:%f' % (context_index().name(i), np.linalg.norm(e[i])))
        print('-- Top tracks --')

        for i in xrange(100):
            print('%s - norm: %f, bias: %f' % (context_index().name(i),
                                               np.linalg.norm(w[i]),
                                               b[i]))


def evaluate(model):
    print("Run evaluation on testset")
    with tf.Session() as session:
        init(session)
        restore(session, FLAGS.checkpoint_dir)
        tf.train.start_queue_runners()
        print('Restored model trained %d steps' % model.global_step.eval())
        metrics = Metrics()

        while True:
            # get batch of users by reading from .tfreacords (TestExample)
            [contexts, time_of_days, time_of_weeks, countries, cities, age, gender,
             platform, histories] = session.run(
                [model.example.context,
                 model.example.time_of_day, model.example.time_of_week,
                 model.example.country, model.example.city, model.example.age,
                 model.example.gender, model.example.platform, model.example.history])
            # Create class from features for pretty prints
            test_users = generate_real_test_users(
                    contexts, time_of_days, time_of_weeks,
                    countries, cities, age, gender, platform, histories
                )
            # print("len contexts: "), len(contexts)
            # print("len time of days: "), len(time_of_days)
            # print(len(test_users))
            cnt = 0
            for x in test_users:
                cnt += 1
                print_similarity(x, session, model, metrics, False)
                if cnt % 50 == 0:
                    print_similarity(x, session, model, metrics, True)
            print(metrics)


def examples(model):
    """ Print examples for evaluation of .tfrecords and their correct format """
    with tf.Session() as session:
        print("Creating tf session")
        init(session)
        tf.train.start_queue_runners()
        ex = model.example
        print("start queue runners::")
        print("while true::")
        while True:
            context, time_of_day, time_of_week, country, city, age, gender, platform, \
            history = session.run([
                ex.context, ex.time_of_day, ex.time_of_week,
                ex.country, ex.city, ex.age, ex.gender,
                ex.platform, ex.history
            ])
            print('Example: \n -------')
            print('User context: %s, time_of_day: %s time_of_week: %s, country: %s,'
                  ' city: %s, age: %s, gender: %s, platform: %s') % (
                      context[0], time_of_day[0], time_of_week[0], country[0], city[0], age[0],
                      gender[0], platform[0]
                    )
            print('History: ')
            print('\t'.join(map(str, history[0])))


def test_examples(model):
    """ Print examples for evaluation of .tfrecords and their correct format """
    with tf.Session() as session:
        print("Creating tf session")
        init(session)
        tf.train.start_queue_runners()
        ex = model.example
        print("start queue runners::")
        print("while true::")
        while True:
            context, time_of_day, time_of_week, country, city, age, gender, platform, user_id, \
            history = session.run([
                ex.context, ex.time_of_day, ex.time_of_week, ex.country, ex.city,
                ex.age, ex.gender, ex.platform, ex.user_id, ex.history
            ])
            print('Example: \n -------')
            print('User context: %s, time_of_day: %s time_of_week: %s, country: %s,'
                  ' city: %s, age: %s, gender: %s, platform: %s, user_id: %s') % (
                      context[0], time_of_day[0], time_of_week[0], country[0], city[0], age[0],
                      gender[0], platform[0], user_id[0]
                    )
            print('History: ')
            print('\t'.join(map(str, history[0])))


def spotify_user_examples(model):
    """ Print examples for evaluation of .tfrecords and their correct format """
    with tf.Session() as session:
        print("Creating tf session")
        init(session)
        tf.train.start_queue_runners()
        ex = model.example
        print("start queue runners::")
        print("while true::")
        while True:
            context, time_of_day, time_of_week, country, city, age, gender, platform, user_id, \
            history = session.run([ex.context, ex.time_of_day, ex.time_of_week,
                                   ex.country, ex.city, ex.age, ex.gender, ex.platform,
                                   ex.user_id, ex.history])

            print('Example: \n -------')
            print('User context: %s, time_of_day: %s time_of_week: %s, country: %s,'
                  ' city: %s, age: %s, gender: %s, platform: %s, user_id: %s') \
                 % (context[0], time_of_day[0], time_of_week[0], country[0], city[0], age[0],
                    gender[0], platform[0], user_id[0])
            print('History: ')
            print('\t'.join(map(str, history[0])))




def demo(model):
    with tf.Session() as session:
        init(session)
        restore(session, FLAGS.checkpoint_dir)
        tf.train.start_queue_runners()
        print('Restored model trained %d steps' % model.global_step.eval())

        while True:
            query = raw_input('Enter Spotify Username: ').decode(sys.stdin.encoding)
            user_id = get_spotify_username(query)
            if user_id is not None:
                print "User found. User_id: %s" % user_id
            else:
                print("Username not found.")
                continue

            found = []
            metrics = Metrics()
            examples = 0
            while True:
                if metrics.num_batches % 1000 == 0:
                    print("Still looking...")
                # get batch of users by reading from .tfreacords (TestExample)
                [contexts, time_of_days, time_of_weeks, countries, cities, age, gender,
                 platform, user_ids, histories] = session.run(
                    [model.example.context,
                     model.example.time_of_day, model.example.time_of_week,
                     model.example.country, model.example.city, model.example.age,
                     model.example.gender, model.example.platform, model.example.user_id,
                     model.example.history])
                metrics.num_batches += 1
                try:
                    found.append(user_ids.tolist().index(user_id))
                except ValueError:
                    continue

                if len(found):
                    for x in found:
                        current = generate_real_test_users([contexts[x]], [time_of_days[x]],
                                                              [time_of_weeks[x]], [countries[x]],
                                                              [cities[x]], [age[x]], [gender[x]],
                                                              [platform[x]], [histories[x]])

                        print_similarity(current[0], session, model, metrics, True)
                        examples += 1
                        break
                    print(metrics)
                    break



def visualize(model):
    def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)

    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
        raise

    with tf.Session() as session:
        init(session)
        restore(session, FLAGS.checkpoint_dir)
        final_embeddings = model.normalized_embeddings.eval()

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [context_index().name(i) for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)


def init(session):
    """ Initialize the tf session """
    log.info('Tensorflow version: %r', tf.__version__)
    session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


@cached
def country_index():
    """ Lookup index for country embeddings """
    return TSVIndex('country', FLAGS.country_tsv)


@cached
def city_index():
    """ Lookup index for context embeddings """
    return TSVIndex('city', FLAGS.city_tsv)


@cached
def context_index():
    """ Lookup index for context embeddings """
    return TSVIndex('context', FLAGS.context_tsv)


@cached
def platform_index():
    """ Lookup index for context embeddings """
    return TSVIndex('platform', FLAGS.platform_tsv)


def main():
    if not FLAGS.cloud:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format='%(message)s')

    # Avoid using training examples by mistake
    records_dir = FLAGS.records_dir if (FLAGS.examples or FLAGS.train) else ''

    test_records_dir = FLAGS.test_records_dir
    test_example = Example(records_dir=test_records_dir, num_epochs=FLAGS.num_epochs,
                      batch_size=FLAGS.batch_size)

    metrics = Metrics()
    model_class = pms.model.get(FLAGS.model_class)
    if not model_class:
        raise Exception('Invalid model class: %s' % FLAGS.model_class)

    eval_model = model_class(example=test_example, metrics=metrics,
                             country_vocabulary_size=FLAGS.country_count,
                        city_vocabulary_size=FLAGS.city_count,
                        context_vocabulary_size=FLAGS.context_count,
                        history_size=100000,
                        embedding_size=64)

    eval_examples = gen_testdata(eval_model)
    eval_model = None

    metrics = Metrics()
    example = Example(records_dir=records_dir, num_epochs=FLAGS.num_epochs,
                          batch_size=FLAGS.batch_size)

    model = model_class(example=example, metrics=metrics,
                        country_vocabulary_size=FLAGS.country_count,
                        city_vocabulary_size=FLAGS.city_count,
                        context_vocabulary_size=FLAGS.context_count,
                        history_size=100000,
                        embedding_size=64)

    model.eval_examples = eval_examples
    model.metrics = metrics

    if FLAGS.examples:
        if FLAGS.test:
            test_examples(model)
        else:
            examples(model)
    elif FLAGS.train:
        if FLAGS.test:
            print("DONT TRAIN ON THE TEST DATA")
            sys.exit(-1)
        else:
            train(model)
    elif FLAGS.eval:
        evaluate(model)
    elif FLAGS.stats:
        stats(model)
    elif FLAGS.visualize:
        visualize(model)
    elif FLAGS.demo:
        demo(model)
    else:
        raise Exception('No command, try --demo or --train')


if __name__ == '__main__':
    filename = sys.argv[1]
    if filename.endswith('.yaml'):
        print('Loading: %s' % filename)
        params = yaml.load(open(filename))
        for key, value in params.items():
            setattr(FLAGS, key.replace('-', '_'), value)
            print('- %s: %s' % (key, value))
    main()

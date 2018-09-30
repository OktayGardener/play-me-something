import collections
import tensorflow as tf


class TSVIndex:
    def __init__(self, name, tsv_file):
        self.id_to_name = {}
        self.name_to_id = {}

        if not tsv_file:
            print('No %s TSV index file given. Features will be set to 0.' % name)
            self.id_to_name[0] = 'unknown'
            self.name_to_id = collections.defaultdict(int)
        else:
            print('Opening: ', tsv_file)
            with tf.gfile.GFile(tsv_file) as f:
                f.next()  # skip header
                for id, line in enumerate(f):
                    name = line.strip().split('\t')[0]
                    self.id_to_name[id] = name
                    self.name_to_id[name] = id

    def id(self, word):
        return self.name_to_id[word]

    def name(self, word_id):
        return self.id_to_name[word_id].replace("$", "S").decode("utf-8", "ignore")

    def max_id(self):
        return max(self.id_to_name.keys())

    def contains(self, name):
        return name in self.name_to_id

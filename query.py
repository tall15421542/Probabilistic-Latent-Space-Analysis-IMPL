import numpy as np

from utils import get_file_eof

def get_train_term_id(voc_pair, voc_id_to_train_id_dict, train_voc_pair_to_term_id_dict):
    first_voc_id, second_voc_id = voc_pair
    if first_voc_id in voc_id_to_train_id_dict:
        train_first_voc_id = voc_id_to_train_id_dict.get(first_voc_id)
    else:
        return -1
    if second_voc_id in voc_id_to_train_id_dict:
        train_second_voc_id = voc_id_to_train_id_dict.get(second_voc_id)
    elif second_voc_id == -1:
        train_second_voc_id == -1
    else:
        return -1

    if (train_first_voc_id, train_second_voc_id) in train_voc_pair_to_term_id_dict:
        return train_voc_pair_to_term_id_dict.get((train_first_voc_id, train_second_voc_id))
    else:
        return -1

class QueryTerm:
    def __init__(self, term_id, tf, tfidf):
        self.term_id = term_id
        self.tf = tf
        self.tfidf = tfidf

class Query:
    def __init__(self, num_of_topic, query_id, url):
        self.term_vec = []
        self.topic_distribution = np.random.dirichlet(np.ones(num_of_topic))
        self.query_id = query_id
        self.url = url

class QueryContainer:
    def __init__(self, num_of_topic, model_path, train_voc_to_id_dict, train_voc_pair_to_term_id_dict):
        self.query_vec = []
        # read doc to initilize query
        query_path = '{}/file-list'.format(model_path)
        with open(query_path) as query_file:
            urls = query_file.read().splitlines()
            for query_id, url in enumerate(urls):
                self.query_vec.append(Query(num_of_topic, query_id, url))

        # read vocab, remap vocab id
        vocab_path = '{}/vocab.all'.format(model_path)
        voc_id_to_train_id_dict = {}
        with open(vocab_path) as vocab_file:
            vocs = vocab_file.read().splitlines()
            for voc_id, voc in enumerate(vocs):
                if voc in train_voc_to_id_dict:
                    voc_id_to_train_id_dict[voc_id] = train_voc_to_id_dict.get(voc)

        # read inverted file/build query based on train term id
        inverted_file_path = '{}/inverted-file'.format(model_path)
        with open(inverted_file_path) as inverted_file:
            eof = get_file_eof(inverted_file)
            while inverted_file.tell() != eof:
                first_voc_id, second_voc_id, df = map(int, inverted_file.readline().split(" "))
                term_id = get_train_term_id((first_voc_id, second_voc_id), \
                        voc_id_to_train_id_dict, train_voc_pair_to_term_id_dict)

                for i in range(df):
                    query_id, tf, tfidf = inverted_file.readline().split(" ")
                    if term_id == -1:
                        continue
                    query_id = int(query_id)
                    tf = int(tf)
                    tfidf = float(tfidf)
                    query = self.query_vec[query_id]
                    query.term_vec.append(QueryTerm(term_id, tf, tfidf))


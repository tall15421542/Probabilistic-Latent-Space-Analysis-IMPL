import numpy as np
import math

def get_topk_idx_of_2d_arr(nd_arr, k):
    idx = np.argpartition(nd_arr, -k)
    idx = idx[:,-k:]
    topk_unsorted = np.take_along_axis(nd_arr, idx, axis = -1)
    topk_idx_of_idx = np.argsort(topk_unsorted, axis = -1)
    topk_idx = np.take_along_axis(idx, topk_idx_of_idx, axis = -1)
    return topk_idx

def get_topk_value_given_topk_idx(nd_arr, topk_idx):
    return np.take_along_axis(nd_arr, topk_idx, axis = -1)


class PLSA:
    def __init__(self, document_vec, inverted_file_term_vec, num_of_topic):
        self.document_vec = document_vec
        self.num_of_doc = len(document_vec)
        self.num_of_term = len(inverted_file_term_vec)
        self.num_of_topic = num_of_topic
        self.document_vec = document_vec
        self.inverted_file_term_vec = inverted_file_term_vec
        self.prob_topic_given_doc_and_term = np.ndarray( \
                shape = (self.num_of_topic, self.num_of_doc, self.num_of_term), dtype = float)
        self.prob_term_given_topic = np.random.dirichlet(np.ones(self.num_of_term), self.num_of_topic)
        self.prob_topic_given_doc_tran = np.random.dirichlet(np.ones(self.num_of_topic), self.num_of_doc).transpose()

    def E_step(self):
        for topic_id in range(self.num_of_topic):
            for doc_id in range(self.num_of_doc):
                for term_id in range(self.num_of_term):
                    self.prob_topic_given_doc_and_term[topic_id][doc_id][term_id] = \
                            self.prob_topic_given_doc_tran[topic_id][doc_id] * self.prob_term_given_topic[topic_id][term_id]
        normalizer = self.prob_topic_given_doc_and_term.sum(axis=0)
        normalizer[normalizer == 0] = 1
        self.prob_topic_given_doc_and_term /= normalizer
    
    def M_step(self):
        # update P(w|z)
        for topic_id in range(self.num_of_topic):
            for term_id, inverted_index_vec in enumerate(self.inverted_file_term_vec):
                prob = 0.
                for inverted_index in inverted_index_vec:
                    prob += inverted_index.tf * self.prob_topic_given_doc_and_term[topic_id][inverted_index.doc_id][term_id]    
                self.prob_term_given_topic[topic_id][term_id] = prob
        normalizer = self.prob_term_given_topic.sum(axis=1)
        np.testing.assert_array_equal
        self.prob_term_given_topic /= normalizer[:, np.newaxis]
        
        # update P(z|d)
        for topic_id in range(self.num_of_topic):
            for doc_id, document in enumerate(self.document_vec):
                prob = 0.
                term_vec = document.term_vec
                for document_term in term_vec:
                    prob += document_term.tf * self.prob_topic_given_doc_and_term[topic_id][doc_id][document_term.term_id]
                self.prob_topic_given_doc_tran[topic_id][doc_id] = prob
        normalizer = self.prob_topic_given_doc_tran.sum(axis=0)
        self.prob_topic_given_doc_tran /= normalizer
                
    def evaluate_likelihood(self):
        # f(d,w) * log P(w|d)
        # P(w|d) = P(z|d)P(w|z)
        likelihood = 0.
        for doc_id, document in enumerate(self.document_vec):
            document_term_vec = document.term_vec
            for document_term in document_term_vec:
                term_id = document_term.term_id
                prob = 0.
                for topic_id in range(self.num_of_topic):
                    prob += self.prob_topic_given_doc_tran[topic_id][doc_id] * self.prob_term_given_topic[topic_id][term_id]  
                likelihood += document_term.tf * math.log(prob)
        return -likelihood

    # Model Train EM / evaluate likelihood / early stopping
    def train(self):
        for i in range(20):
            print("#######################")
            print("E step", i)
            self.E_step()
            print("E step Complete")
            print("M step", i)
            self.M_step()
            print("M step Complete")
            likelihood = self.evaluate_likelihood()
            print(likelihood)
            print("#######################")

    def output_topk_term_given_topic(self, topk, term_id_voc_pair_vec, voc_id_to_voc_vec, model_path):
        topk_term_given_topic_path = '{}/topk_term_given_topic'.format(model_path)
        with open(topk_term_given_topic_path, "w") as topk_term_given_topic_file:
            topk_idx = get_topk_idx_of_2d_arr(self.prob_term_given_topic, topk)
            topk_idx = np.flip(topk_idx, axis = -1)
            for topic_id in range(self.num_of_topic):
                topk_term_given_topic_file.write("topic_id {}\n".format(topic_id))
                for idx in range(topk):
                    term_id = topk_idx[topic_id][idx]
                    first_voc_id, second_voc_id = term_id_voc_pair_vec[term_id]
                    topk_term_given_topic_file.write('{}{}\n'.format(voc_id_to_voc_vec[first_voc_id], \
                            voc_id_to_voc_vec[second_voc_id] if second_voc_id != -1 else ''))
                topk_term_given_topic_file.write('\n')

    def output_topk_doc_given_topic(self, topk, doc_id_to_url_vec, model_path):
        topk_doc_given_topic_path = '{}/topk_doc_given_topic_path'.format(model_path)
        with open(topk_doc_given_topic_path, "w") as topk_doc_given_topic_file:
            topk_idx = get_topk_idx_of_2d_arr(self.prob_topic_given_doc_tran, topk)
            topk_idx = np.flip(topk_idx, axis = -1)
            for topic_id in range(self.num_of_topic):
                topk_doc_given_topic_file.write("topic_id {}\n".format(topic_id))
                for idx in range(topk):
                    doc_id = topk_idx[topic_id][idx]
                    doc_url = doc_id_to_url_vec[doc_id]
                    topk_doc_given_topic_file.write('{}\n'.format(doc_url))
                topk_doc_given_topic_file.write('\n')

import numpy as np
import math

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

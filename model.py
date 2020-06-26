import numpy as np
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

        print(self.prob_topic_given_doc_and_term.shape)
        print(self.prob_term_given_topic.shape)
        print(self.prob_term_given_topic.sum(axis=1))
        print(self.prob_topic_given_doc_tran.shape)
        print(self.prob_topic_given_doc_tran)
        print(self.prob_topic_given_doc_tran.sum(axis=0))

    def E_step(self):
        for topic_id in range(self.num_of_topic):
            for doc_id in range(self.num_of_doc):
                for term_id in range(self.num_of_term):
                    self.prob_topic_given_doc_and_term[topic_id][doc_id][term_id] = \
                            self.prob_topic_given_doc_tran[topic_id][doc_id] * self.prob_term_given_topic[topic_id][term_id]
        normalizer = self.prob_topic_given_doc_and_term.sum(axis=0)
        self.prob_topic_given_doc_and_term /= normalizer
        print(self.prob_topic_given_doc_and_term.sum(axis=0))
        
    # Model Train EM / evaluate likelihood / early stopping
    def train(self):
        self.E_step()

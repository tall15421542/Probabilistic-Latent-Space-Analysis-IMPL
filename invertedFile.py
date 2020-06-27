class InvertedIndex:
    def __init__(self, doc_id, tf, tfidf):
        self.doc_id = doc_id
        self.tf = tf
        self.tfidf = tfidf

class InvertedFile:
    def __init__(self, document_vec, num_of_term):
        # (term_id, [InvertedIndex])
        self.term_vec = []
        for i in range(num_of_term):
            self.term_vec.append([])

        for doc_id, doc in enumerate(document_vec):
            for term in doc.term_vec:
                self.term_vec[term.term_id].append(InvertedIndex(doc_id, term.tf, term.tfidf))


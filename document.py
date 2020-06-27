class DocumentTerm:
    def __init__(self, term_id, tf, tfidf):
        self.term_id = term_id
        self.tf = tf
        self.tfidf = tfidf

class Document:
    def __init__(self, doc_id):
        self.term_vec = []
        self.doc_id = doc_id

class DocumentContainer:
    def __init__(self, num_of_doc):
        self.doc_vec = []
        for doc_id in range(num_of_doc):
            self.doc_vec.append(Document(doc_id))

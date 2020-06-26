import argparse 
import os
import random 

from model import PLSA

def get_file_eof(file_obj):
    file_obj.seek(0, os.SEEK_END)
    eof = file_obj.tell()
    file_obj.seek(0, 0)
    return eof

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

def get_doc_id_to_url_vec(file_list_path):
    doc_id_to_url_vec = []
    with open(file_list_path) as file_list:
        lines = file_list.read().splitlines()
        for url in lines:
            doc_id_to_url_vec.append(url)
    return doc_id_to_url_vec

def get_voc_id_to_voc_vec(voc_list_path):
    voc_id_to_voc_vec = []
    with open(voc_list_path) as voc_list:
        lines = voc_list.read().splitlines()
        for voc in lines:
            voc_id_to_voc_vec.append(voc)
    return voc_id_to_voc_vec

def read_inverted_file(inverted_file_path, train_document_container, term_id_to_vec_pair_vec):
    with open(inverted_file_path) as inverted_file:
        eof = get_file_eof(inverted_file)
        while(inverted_file.tell() != eof):
            first_voc_id, second_voc_id, df = map(int, inverted_file.readline().split(" "))
            next_term_id = len(term_id_to_vec_pair_vec)
            term_id_to_vec_pair_vec.append((first_voc_id, second_voc_id))
            for doc in range(df):
                doc_id, tf, tfidf = inverted_file.readline().split(" ")
                doc_id = int(doc_id)
                tf = int(tf)
                tfidf = float(tfidf)
                document = train_document_container.doc_vec[doc_id]
                document.term_vec.append(DocumentTerm(next_term_id, tf, tfidf))

def main():
    parser = argparse.ArgumentParser(description = "plsa")
    parser.add_argument('-m', action = 'store', dest = 'model_path', required = True)
    parser.add_argument('-r', action = 'store', dest = 'train_ratio', type=float, default = 0.9)
    parser.add_argument("-t", action = 'store', dest = 'num_of_topic', type = int, default = 8)
    args = parser.parse_args()

    inverted_file_path = '{}/inverted-file'.format(args.model_path)
    voc_list_path = '{}/vocab.all'.format(args.model_path)
    file_list_path = '{}/file-list'.format(args.model_path)
    print(inverted_file_path)
    print(voc_list_path)
    print(file_list_path)

    doc_id_to_url_vec = get_doc_id_to_url_vec(file_list_path)
    voc_id_to_voc_vec = get_voc_id_to_voc_vec(voc_list_path)

    train_document_container = DocumentContainer(len(doc_id_to_url_vec))

    # (first_voc_id, second_voc_id)
    term_id_to_vec_pair_vec = []
    
    # read inverted file / build document_container
    read_inverted_file(inverted_file_path, train_document_container, term_id_to_vec_pair_vec)
                
    # build inverted_file for train
    train_inverted_file = InvertedFile(train_document_container.doc_vec, len(term_id_to_vec_pair_vec))
    
    # Model initialize
    model = PLSA(train_document_container.doc_vec, train_inverted_file.term_vec, args.num_of_topic)

    # Model Train EM / evaluate likelihood / early stopping
    model.train()

    # output top k P(w|z) over z 

    # output top k P(z|d) for over d

if __name__ == '__main__':
    main()

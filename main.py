import argparse 
import os

from model import PLSA
from document import DocumentContainer
from invertedFile import InvertedFile
from utils import get_doc_id_to_url_vec
from utils import get_voc_id_to_voc_vec
from utils import read_inverted_file

def main():
    parser = argparse.ArgumentParser(description = "plsa")
    parser.add_argument('-m', action = 'store', dest = 'model_path', required = True)
    parser.add_argument('-r', action = 'store', dest = 'train_ratio', type=float, default = 0.9)
    parser.add_argument("-t", action = 'store', dest = 'num_of_topic', type = int, default = 16)
    parser.add_argument('-k', action = 'store', dest = 'topk', type = int, default = 10)
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
    term_id_to_voc_pair_vec = []
    
    # read inverted file / build document_container
    read_inverted_file(inverted_file_path, train_document_container, term_id_to_voc_pair_vec)
                
    # build inverted_file for train
    train_inverted_file = InvertedFile(train_document_container.doc_vec, len(term_id_to_voc_pair_vec))
    
    # Model initialize
    model = PLSA(train_document_container.doc_vec, train_inverted_file.term_vec, args.num_of_topic)

    # Model Train EM / evaluate likelihood / early stopping
    model.train()

    # output top k P(w|z) over z 
    model.output_topk_term_given_topic(args.topk, term_id_to_voc_pair_vec, voc_id_to_voc_vec, args.model_path)
    model.output_topk_doc_given_topic(args.topk, doc_id_to_url_vec, args.model_path)

    # output top k P(z|d) for over d

if __name__ == '__main__':
    main()

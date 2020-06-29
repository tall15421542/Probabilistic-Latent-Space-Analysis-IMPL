import argparse 
import os

from model import PLSA
from document import DocumentContainer
from invertedFile import InvertedFile
from utils import get_doc_id_to_url_vec
from utils import get_voc_id_to_voc_vec
from utils import read_inverted_file
from utils import make_dir_if_not_exist 
from utils import get_url_set
from utils import output_ranking_list
from query import QueryContainer
from MAP import MAP
from queryParser import QueryXMLParser

def main():
    parser = argparse.ArgumentParser(description = "plsa")
    parser.add_argument('-m', action = 'store', dest = 'model_path', required = True)
    parser.add_argument('-r', action = 'store', dest = 'train_ratio', type=float, default = 0.9)
    parser.add_argument("-t", action = 'store', dest = 'num_of_topic', type = int, default = 10)
    parser.add_argument('-k', action = 'store', dest = 'topk', type = int, default = 20)
    parser.add_argument('-q', action = 'store', dest = 'query_model_path')
    parser.add_argument('-v', action = 'store', dest = 'validation_model_path')
    parser.add_argument('-a', action = 'store', dest = 'ans_path', default = "queries/ans_train.csv")
    parser.add_argument('--ranking', action = 'store', dest = 'ranking_list_dir')
    parser.add_argument('--test', action = 'store', dest = 'test_model_path')
    args = parser.parse_args()

    inverted_file_path = '{}/inverted-file'.format(args.model_path)
    voc_list_path = '{}/vocab.all'.format(args.model_path)
    file_list_path = '{}/file-list'.format(args.model_path)
    print(inverted_file_path)
    print(voc_list_path)
    print(file_list_path)

    doc_id_to_url_vec = get_doc_id_to_url_vec(file_list_path)
    doc_url_set = get_url_set(file_list_path)
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

    voc_to_voc_id_dict = {}
    for voc_id, voc in enumerate(voc_id_to_voc_vec):
        voc_to_voc_id_dict[voc] = voc_id 
    
    voc_pair_to_term_id_dict = {}
    for term_id, voc_pair in enumerate(term_id_to_voc_pair_vec):
        voc_pair_to_term_id_dict[voc_pair] = term_id
    
    if(args.validation_model_path):
        print(args.validation_model_path)
        validation_container = QueryContainer(args.num_of_topic, args.validation_model_path, \
                voc_to_voc_id_dict, voc_pair_to_term_id_dict)
        model.set_validation(validation_container.doc_vec)

    # Model Train EM / evaluate likelihood / early stopping
    is_not_folding = False
    model.train(is_not_folding)

    # output top k P(w|z) over z 
    model.output_topk_term_given_topic(args.topk, term_id_to_voc_pair_vec, voc_id_to_voc_vec, args.model_path)

    # output top k P(z|d) for over d
    model.output_topk_doc_given_topic(args.topk, doc_id_to_url_vec, args.model_path)

    # output doc and topic mapping
    model.output_doc_and_topic_mapping(doc_id_to_url_vec, args.model_path)

    # read query model
    if(args.query_model_path):
        print(args.query_model_path)

        query_container = QueryContainer(args.num_of_topic, args.query_model_path, \
                voc_to_voc_id_dict, voc_pair_to_term_id_dict)

        query_folding_engine = PLSA(query_container.doc_vec, train_inverted_file.term_vec, \
                args.num_of_topic)
        query_folding_engine.set_prob_term_given_topic(model.prob_term_given_topic)
        query_folding_engine.folding()

        query_id_to_topk_doc_id_vec = model.retrieve_topk_doc_id(query_folding_engine.prob_topic_given_doc_tran.transpose())
        ranking_list_path = '{}/ranking_list_{}'.format(args.model_path, args.num_to_topic)
        output_ranking_list(args.topk, query_id_to_topk_doc_id_vec, doc_id_to_url_vec, args.ranking_list_dir)

#        query_folding_engine.output_topk_query_given_topic(args.topk, doc_id_to_url_vec, args.model_path)
#        query_folding_engine.output_query_status(doc_id_to_url_vec, args.model_path)

    if(args.test_model_path):
        print(args.test_model_path)
        test_container = QueryContainer(args.num_of_topic, args.test_model_path, \
                voc_to_voc_id_dict, voc_pair_to_term_id_dict)
        test_folding_engine = PLSA(test_container.doc_vec, train_inverted_file.term_vec, \
                args.num_of_topic)
        test_folding_engine.output_doc_and_topic_mapping(doc_id_to_url_vec, args.model_path)


if __name__ == '__main__':
    main()

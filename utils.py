import os
from document import DocumentTerm

def get_file_eof(file_obj):
    file_obj.seek(0, os.SEEK_END)
    eof = file_obj.tell()
    file_obj.seek(0, 0)
    return eof

def get_url_set(file_list_path):
    url_set = set()
    with open(file_list_path) as file_list:
        lines = file_list.read().splitlines()
        for url in lines:
            url = url.split("/")[-1].lower()
            url_set.add(url)
    return url_set

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

def make_dir_if_not_exist(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError:
            print ("Creation of the directory %s failed" % dir_path)
        else:
            print ("Successfully created the directory %s" % dir_path)

def output_ranking_list(topk, query_id_to_topk_doc_id_vec, doc_id_to_url_vec, ranking_list_path):
    with open(ranking_list_path, "w") as ranking_list_file:
        ranking_list_file.write("query_id,retrieved_docs\n")
        for query_id, topk_doc_id in enumerate(query_id_to_topk_doc_id_vec):
            ranking_list_file.write("{},".format(query_id + 1))
            for i in range(topk):
                doc_id = topk_doc_id[i]
                doc_url = doc_id_to_url_vec[doc_id]
                doc_url = doc_url.split("/")[-1].lower()
                ranking_list_file.write("{} ".format(doc_url))
            ranking_list_file.write("\n")
 
            

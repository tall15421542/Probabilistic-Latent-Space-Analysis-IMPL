import numpy as np
import csv
import xml.etree.ElementTree as ET

class MAP:
    def __init__(self, ans_path, doc_id_to_url_vec, prob_topic_given_doc, train_url_set):
        self.ans_train_vec = []
        with open(ans_path, "r") as ans_file:
            csv_file = csv.DictReader(ans_file)
            for row in csv_file:
                files = row['retrieved_docs'].split(" ")
                doc_url_set = set()
                for file_url in files:
                    if file_url in train_url_set:
                        doc_url_set.add(file_url)
                self.ans_train_vec.append(doc_url_set)

        self.url_map_to_query_dict = {}
        for query_id, doc_url_set in enumerate(self.ans_train_vec):
            for doc_url in doc_url_set:
                if doc_url not in self.url_map_to_query_dict:
                    self.url_map_to_query_dict[doc_url] = set()
                self.url_map_to_query_dict.get(doc_url).add(query_id)

        self.doc_id_to_url_vec = doc_id_to_url_vec
        self.topic_to_sorted_doc_id = []
        self.set_doc_id_to_topic_vec(prob_topic_given_doc)
        self.set_topic_id_to_query_id_vec(self.topic_id_to_doc_vec)

    def set_topic_id_to_query_id_vec(self, topic_id_to_doc_vec):
        num_of_topic = len(topic_id_to_doc_vec)
        self.topic_id_to_query_id_vec = [0] * num_of_topic
        for topic_id, doc_id_set in enumerate(topic_id_to_doc_vec):
            query_id_cnt_vec = [0] * num_of_topic
            for doc_id in doc_id_set:
                doc_url = self.doc_id_to_url_vec[doc_id]
                doc_url = doc_url.split("/")[-1].lower()
                for query_id in self.url_map_to_query_dict[doc_url]:
                    query_id_cnt_vec[query_id] += 1
            self.topic_id_to_query_id_vec[topic_id] = query_id_cnt_vec.index(max(query_id_cnt_vec))

    def set_doc_id_to_topic_vec(self, prob_topic_given_doc):
        num_of_doc, num_of_topic = prob_topic_given_doc.shape
        doc_id_to_topic_vec = np.argmax(prob_topic_given_doc, axis = -1)
        self.topic_id_to_doc_vec = [set() for i in range(num_of_topic)] 
        for doc_id, topic_id in enumerate(doc_id_to_topic_vec):
            self.topic_id_to_doc_vec[topic_id].add(doc_id)

    def evaluate(self, topk_doc_given_topic):
        map_score = 0.
        for topic_id, doc_id_vec in enumerate(topk_doc_given_topic):
            query_id = self.topic_id_to_query_id_vec[topic_id]
            map_score_query = 0.
            cnt = 0
            print("Topic", topic_id, "=>", query_id)
            for idx, doc_id in enumerate(doc_id_vec):
                rank = idx + 1
                doc_url = self.doc_id_to_url_vec[doc_id]
                doc_url = doc_url.split("/")[-1].lower()
                ans_query_id = self.url_map_to_query_dict[doc_url]
                if query_id in ans_query_id:
                    cnt += 1
                    map_score_query += float(cnt) / float(rank)
            map_score_query /= len(self.ans_train_vec[query_id])
            map_score += map_score_query
            print("map", map_score_query)
            precision = float(cnt) / float(len(doc_id_vec))
            print("percision", precision)
        map_score /= len(topk_doc_given_topic)
        with open("map_score_plsa", "a") as map_file:
            map_file.write('{}\n'.format(map_score))
        return map_score
                



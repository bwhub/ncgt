'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from calendar import c
import pickle

import sklearn
import pandas as pd

from code.base_class.method import method


class MethodGraphBatching(method):
    data = None
    k = 5

    def __init__(self, similarity=None):
        super(method, self).__init__()
        self.similarity = similarity
    
    def get_top_k_neighbor(self, S):
        # key: node index in adj matrix
        # value: node/document identifier
        index_id_dict = self.data['index_id_map']

        # key: node/document identifier
        # value: list of tuples of neighbors (node/document identifier, intimacy score)
        user_top_k_neighbor_intimacy_dict = {}
        
        for node_index in index_id_dict:
            node_id = index_id_dict[node_index]
            s = S[node_index]
            s[node_index] = -1000.0 # since we don't want a node to appear in its node context
            top_k_neighbor_index = s.argsort()[-self.k:][::-1]
            user_top_k_neighbor_intimacy_dict[node_id] = []
            for neighbor_index in top_k_neighbor_index:
                neighbor_id = index_id_dict[neighbor_index]
                user_top_k_neighbor_intimacy_dict[node_id].append((neighbor_id, s[neighbor_index]))

        return user_top_k_neighbor_intimacy_dict

    # def get_top_k_neighbor_from_dict(self, pickled_dict_path):
    #     with open(pickled_dict_path, 'rb') as f:
    #         result_dict = pickle.load(f)

    #     node_ind_2_paper_id_path = './data/ogbn-arxiv/dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv'
    #     id_2_ind = self.get_paper_id_2_node_ind_dict(node_ind_2_paper_id_path)
        
    #     user_top_k_neighbor_intimacy_dict = {}

    #     for (paper_id, neighbor_list) in result_dict.items():
    #         node_ind = id_2_ind[paper_id]
    #         user_top_k_neighbor_intimacy_dict[node_ind] = []

    #         for i in range(self.k):
    #             temp_tup = (id_2_ind[neighbor_list[i][0]], neighbor_list[i][1])
    #             user_top_k_neighbor_intimacy_dict[node_ind].append(temp_tup)

    #         # print(user_top_k_neighbor_intimacy_dict[node_ind])
    #         # assert False, 'Break Point'

    #     return user_top_k_neighbor_intimacy_dict


    def get_top_k_neighbor_from_dict(self, pickled_dict_path):
        with open(pickled_dict_path, 'rb') as f:
            result_dict = pickle.load(f)

        # node_ind_2_paper_id_path = './data/ogbn-arxiv/dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv'
        # id_2_ind = self.get_paper_id_2_node_ind_dict(node_ind_2_paper_id_path)
        
        user_top_k_neighbor_intimacy_dict = {}

        for (node_ind, neighbor_list) in result_dict.items():
            # node_ind = id_2_ind[paper_id]
            user_top_k_neighbor_intimacy_dict[node_ind] = []

            for i in range(self.k):
                temp_tup = (neighbor_list[i][0], neighbor_list[i][1])
                user_top_k_neighbor_intimacy_dict[node_ind].append(temp_tup)

            # print(user_top_k_neighbor_intimacy_dict[node_ind])
            # assert False, 'Break Point'

        return user_top_k_neighbor_intimacy_dict

    def get_top_k_neighbor_from_dict_two_level(self, pickled_dict_path, multiple):
        with open(pickled_dict_path, 'rb') as f:
            result_dict = pickle.load(f)
        
        user_top_k_neighbor_intimacy_dict = {}


        for (node_ind, neighbor_list) in result_dict.items():
            user_top_k_neighbor_intimacy_dict[node_ind] = []

            node_feature = self.data['X'][node_ind].cpu().detach().numpy().reshape(1, -1)

            # print(f"node_feauture is of shape {node_feature.shape}.")
            # print(f"node_ind = {node_ind}.")

            cosine_similarity_list = []

            for i in range(self.k*multiple):
                # temp_tup = (id_2_ind[neighbor_list[i][0]], neighbor_list[i][1])
                # user_top_k_neighbor_intimacy_dict[node_ind].append(temp_tup)
                neighbor_node_ind = neighbor_list[i][0]
                neighbor_node_feature = self.data['X'][neighbor_node_ind].cpu().detach().numpy().reshape(1, -1)
                cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(node_feature, neighbor_node_feature)[0][0]
                cosine_similarity_list.append((neighbor_node_ind, cosine_similarity))

            cosine_similarity_list.sort(key=lambda i:i[1],reverse=True)
            user_top_k_neighbor_intimacy_dict[node_ind] = cosine_similarity_list[:self.k]
            # print(f"cosine_similarity_list = {cosine_similarity_list}")
            # print(f"final_similarity_list = {user_top_k_neighbor_intimacy_dict[node_ind]}")
            # print(f"\n\n")


        return user_top_k_neighbor_intimacy_dict


    # def get_top_k_neighbor_from_dict_two_level(self, pickled_dict_path, multiple):
    #     with open(pickled_dict_path, 'rb') as f:
    #         result_dict = pickle.load(f)

    #     node_ind_2_paper_id_path = './data/ogbn-arxiv/dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv'
    #     id_2_ind = self.get_paper_id_2_node_ind_dict(node_ind_2_paper_id_path)
        
    #     user_top_k_neighbor_intimacy_dict = {}

    #     features = self.data['X'].cpu().detach().numpy()

    #     for (paper_id, neighbor_list) in result_dict.items():
    #         node_ind = id_2_ind[paper_id]
    #         user_top_k_neighbor_intimacy_dict[node_ind] = []

    #         node_feature = self.data['X'][node_ind].cpu().detach().numpy().reshape(1, -1)

    #         # print(f"node_feauture is of shape {node_feature.shape}.")
    #         # print(f"node_ind = {node_ind}.")

    #         cosine_similarity_list = []

    #         for i in range(self.k*multiple):
    #             # temp_tup = (id_2_ind[neighbor_list[i][0]], neighbor_list[i][1])
    #             # user_top_k_neighbor_intimacy_dict[node_ind].append(temp_tup)
    #             neighbor_node_ind = id_2_ind[neighbor_list[i][0]]
    #             neighbor_node_feature = self.data['X'][neighbor_node_ind].cpu().detach().numpy().reshape(1, -1)
    #             cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(node_feature, neighbor_node_feature)[0][0]
    #             cosine_similarity_list.append((neighbor_node_ind, cosine_similarity))

    #         cosine_similarity_list.sort(key=lambda i:i[1],reverse=True)
    #         user_top_k_neighbor_intimacy_dict[node_ind] = cosine_similarity_list[:self.k]
    #         # print(f"cosine_similarity_list = {cosine_similarity_list}")
    #         # print(f"final_similarity_list = {user_top_k_neighbor_intimacy_dict[node_ind]}")
    #         # print(f"\n\n")


    #     return user_top_k_neighbor_intimacy_dict


    def get_paper_id_2_node_ind_dict(self, node_ind_2_paper_id_path):
        # Load the mapping between node_ind and paper_id.
        node_ind_2_paper_id_pd = pd.read_csv(node_ind_2_paper_id_path)

        # A Python dictionary with key: node_ind and value: paper_id
        paper_id_dict = node_ind_2_paper_id_pd.to_dict()['paper id']

        return {val:key for key, val in paper_id_dict.items()}


    def get_top_k_neighbor_two_level(self, S1, S2, multiple=2):
        index_id_dict = self.data['index_id_map']

        user_top_k_neighbor_intimacy_dict = {}
        for node_index in index_id_dict:
            node_id = index_id_dict[node_index]
            s1 = S1[node_index]
            s1[node_index] = -1000.0 # since we don't want a node to appear in its node context
            s1_top_k_multiple_neighbor_index = s1.argsort()[-(multiple*self.k):][::-1]

            s2 = S2[node_index]
            s2[node_index] =  -1000.0
            s2_top_k_neighbor_index = [ind for ind in s2.argsort()[::-1] if ind in s1_top_k_multiple_neighbor_index][:self.k]

            user_top_k_neighbor_intimacy_dict[node_id] = []
            for neighbor_index in s2_top_k_neighbor_index:
                neighbor_id = index_id_dict[neighbor_index]
                user_top_k_neighbor_intimacy_dict[node_id].append((neighbor_id, s2[neighbor_index]))

        return user_top_k_neighbor_intimacy_dict

    def run(self):

        if self.similarity == 'random_walk':
            print('Using random walk for intimacy score.')
            return self.get_top_k_neighbor(self.data['S'])
        elif self.similarity == 'cosine':
            print('Using cosine similarity for intimacy score.')
            return self.get_top_k_neighbor(self.data['S_cosine'])
        elif self.similarity == 'jaccard':
            print('Using jaccard similarity for intimacy score.')
            return self.get_top_k_neighbor(self.data['S_jaccard'])
        elif self.similarity == 'random_walk_and_cosine':
            print('Using random walk first and then cosine similarity for intimacy score.')
            return self.get_top_k_neighbor_two_level(S1=self.data['S'], S2=self.data['S_cosine'])
        elif self.similarity == 'random_walk_and_jaccard':
            print('Using random walk first and then jaccard similarity for intimacy score.')
            return self.get_top_k_neighbor_two_level(S1=self.data['S'], S2=self.data['S_jaccard'], multiple=2)
        elif self.similarity == 'random_walk_dict':
            print('Using random walk (precomputed pickled file) for intimacy score.')
            pickled_dict_path = './data/ogbn-arxiv/220609_142021_top_1000_neighbor_intimacy_dict.pkl'
            return self.get_top_k_neighbor_from_dict(pickled_dict_path)
        elif self.similarity == 'random_walk_dict_and_cosine':
            print('Using random walk (precomputed pickled file) fisrt and then cosine similarity for intimacy score.')
            pickled_dict_path = './data/ogbn-arxiv/220609_142021_top_1000_neighbor_intimacy_dict.pkl'
            return self.get_top_k_neighbor_from_dict_two_level(pickled_dict_path, multiple=2)
        elif self.similarity == 'cosine_dict':
            print('Using cosine_similarity (precomputed pickled file) for intimacy score.')
            pickled_dict_path = './data/ogbn-arxiv/220628_135400_top_100_neighbor_cosine_intimacy_dict.pkl'
            return self.get_top_k_neighbor_from_dict(pickled_dict_path)
        else:
            assert False, 'No given options for similarity measurements'


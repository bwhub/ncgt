'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD
import sklearn
import pickle
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
from ogb.nodeproppred import NodePropPredDataset
from code.base_class.dataset import dataset
# from sklearn import preprocessing


class DatasetLoader(dataset):
    c = 0.15
    k = 5
    data = None
    batch_size = None

    dataset_source_folder_path = None
    dataset_name = None

    load_all_tag = False
    compute_s = False

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DatasetLoader, self).__init__(dName, dDescription)
        self.preprocessing = False

    def load_hop_wl_batch(self):
        print('Load WL Dictionary')
        f = open('./result/WL/' + self.dataset_name, 'rb')
        wl_dict = pickle.load(f)
        f.close()

        print('Load Hop Distance Dictionary')
        f = open('./result/Hop/hop_' + self.dataset_name + '_' + str(self.k), 'rb')
        hop_dict = pickle.load(f)
        f.close()

        print('Load Subgraph Batches')
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        return hop_dict, wl_dict, batch_dict

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot
    
    def compute_cosine_similarity(self, feature_matrix):
        return sklearn.metrics.pairwise.cosine_similarity(feature_matrix, feature_matrix)
    
    def compute_jaccard_similarity(self, feature_matrix):
        # https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
        # https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/spatial/distance.py#L735
        # print(f"Feature matrix before {feature_matrix[-2]}")
        # feature_matrix[feature_matrix > 0] = 1
        # print(f"Feature matrix is of type {type(feature_matrix)} and dtype {feature_matrix.dtype}")
        feature_matrix = np.ceil(feature_matrix.astype(float))
        # print(f"Feature matrix after {feature_matrix[-2]}")
        # assert False, 'Break Point'
        return 1 - sklearn.metrics.pairwise_distances(feature_matrix, metric='jaccard')

    def load(self):
        """Load citation network dataset (cora only for now)"""
        print('Loading {} dataset...'.format(self.dataset_name))
        print(f"Loading dataset from {self.dataset_source_folder_path}/node")
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        print(f"idx_features_labels is of shape {idx_features_labels.shape}.")
        # An example for idx_features_labels when reading from cora dataset
        # array([['31336', '0', '0', ..., '0', '0', 'Neural_Networks'],
        #        ['1061127', '0', '0', ..., '0', '0', 'Rule_Learning'],
        #        ['1106406', '0', '0', ..., '0', '0', 'Reinforcement_Learning'],
        #        ...,
        #        ['1128978', '0', '0', ..., '0', '0', 'Genetic_Algorithms'],
        #        ['117328', '0', '0', ..., '0', '0', 'Case_Based'],
        #        ['24043', '0', '0', ..., '0', '0', 'Neural_Networks']],
        #       dtype='<U22')

        # print('Computing cosine similarities.')
        self.S_cosine = None
        # self.S_cosine = self.compute_cosine_similarity(idx_features_labels[:, 1:-1])
        # self.S_cosine = self.compute_cosine_similarity(idx_features_labels[:, 1:-1].astype(np.float32))
        # print('Computing  Jaccard similarities.')
        self.S_jaccard = None
        # self.S_jaccard = self.compute_jaccard_similarity(idx_features_labels[:, 1:-1])
        # print(f"Cosine similarity matrix is of shape {self.S_cosine.shape}.")
        
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        one_hot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        # A dictionary that represents a map from 
        #   key (node/document identifier, e.g., 31336, 1061127, ...)
        #   to 
        #   value (indices of the adjency matrix of graph that represents that dataset. e.g., 0, 1, ...)
        idx_map = {j: i for i, j in enumerate(idx)}

        print(f"idx_map has {len(idx_map)} key value pairs.")


        # The inverse of idx_map
        index_id_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        print(f"one_hot_labels.shape[0] = {one_hot_labels.shape[0]}")
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
                            dtype=np.float32)

        # print(f"1: adj is symetric {np.allclose(adj.toarray(), adj.T.toarray())}")

        print('Computing random walk similarities.')
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # print(f"2: adj is symetric {np.allclose(adj.toarray(), adj.T.toarray())}")

        eigen_adj = None
        if self.compute_s:
            eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())
        

        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        if self.dataset_name == 'cora':
            idx_train = range(140)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
        elif self.dataset_name == 'citeseer':
            idx_train = range(120)
            idx_test = range(200, 1200)
            idx_val = range(1200, 1500)
            #features = self.normalize(features)
        elif self.dataset_name == 'pubmed':
            idx_train = range(60)
            idx_test = range(6300, 7300)
            idx_val = range(6000, 6300)
        elif self.dataset_name == 'cora-small':
            idx_train = range(5)
            idx_val = range(5, 10)
            idx_test = range(5, 10)
        elif self.dataset_name == 'ogbn-arxiv-small':
            # WARNING: The following split is for development purpose.
            # Do not use for scientific experiments.
            idx_train = range(9000)
            idx_val = range(9000, 16500)
            idx_test = range(16500, 21126)
        elif self.dataset_name == 'ogbn-arxiv':
            ogbn_arxiv_split_idx_path = './data/ogbn-arxiv/ogbn_arxiv_split_idx.pkl'
            print(f"Loading split_idx from {ogbn_arxiv_split_idx_path}.")
            with open(ogbn_arxiv_split_idx_path, 'rb') as f:
                ogbn_arxiv_split_idx = pickle.load(f)
            
            idx_train = ogbn_arxiv_split_idx["train"]
            idx_val = ogbn_arxiv_split_idx["valid"]
            idx_test = ogbn_arxiv_split_idx["test"]
        

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(one_hot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        if self.load_all_tag:
            hop_dict, wl_dict, batch_dict = self.load_hop_wl_batch()
            # print(f"wl_dict.keys() is of type {type(list(wl_dict.keys())[0])}")
            raw_feature_list = []
            role_ids_list = []
            position_ids_list = []
            hop_ids_list = []
            for node in idx:
                node_index = idx_map[node]
                neighbors_list = batch_dict[node]
                # print(f'Node {node} neighbors_list is {neighbors_list}')

                raw_feature = [features[node_index].tolist()]
                role_ids = [wl_dict[node]]
                position_ids = range(len(neighbors_list) + 1)
                hop_ids = [0]
                for neighbor, intimacy_score in neighbors_list:
                    # print(f"neighbor is of type {type(neighbor)}, value {neighbor}.")
                    neighbor_index = idx_map[neighbor]
                    raw_feature.append(features[neighbor_index].tolist())
                    role_ids.append(wl_dict[neighbor])
                    if neighbor in hop_dict[node]:
                        hop_ids.append(hop_dict[node][neighbor])
                    else:
                        hop_ids.append(99)
                
                raw_feature_list.append(raw_feature)
                role_ids_list.append(role_ids)
                position_ids_list.append(position_ids)
                hop_ids_list.append(hop_ids)
            
            raw_embeddings = torch.FloatTensor(raw_feature_list)
            wl_embedding = torch.LongTensor(role_ids_list)
            hop_embeddings = torch.LongTensor(hop_ids_list)
            int_embeddings = torch.LongTensor(position_ids_list)
        else:
            raw_embeddings, wl_embedding, hop_embeddings, int_embeddings = None, None, None, None
        
        # device for training, use GPU (cuda) if accessible.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # The result dictionary to be returned after loading the dataset
        result_dict = {}
        result_dict['S'] = eigen_adj
        result_dict['S_cosine'] = self.S_cosine
        result_dict['S_jaccard'] = self.S_jaccard
        result_dict['index_id_map'] = index_id_map
        result_dict['edges'] = edges_unordered
        result_dict['idx'] = idx
        result_dict['idx_train'] = idx_train
        result_dict['idx_test'] = idx_test
        result_dict['idx_val'] = idx_val
        if self.preprocessing:
            result_dict['X'] = features
            result_dict['A'] = adj
            result_dict['y'] = labels
            result_dict['raw_embeddings'] = raw_embeddings
            result_dict['wl_embedding'] = wl_embedding
            result_dict['hop_embeddings'] = hop_embeddings
            result_dict['int_embeddings'] = int_embeddings
        else:
            result_dict['X'] = features.to(device)
            result_dict['A'] = adj.to(device)
            result_dict['y'] = labels.to(device)
            result_dict['raw_embeddings'] = raw_embeddings.to(device)
            result_dict['wl_embedding'] = wl_embedding.to(device)
            result_dict['hop_embeddings'] = hop_embeddings.to(device)
            result_dict['int_embeddings'] = int_embeddings.to(device)

        return result_dict

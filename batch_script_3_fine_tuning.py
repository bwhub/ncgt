# batch_script_3_fine_tuning.py
#!/usr/bin/env python3

"""Script for running fine tuning experiments of GraphBert from terminals.

Fine tuning experiments. Currently supports:
- Graph Bert Node Classification.

Avaiable functions:
- run_fine_tuning_on_node_classification: Fine-Tuning Task 1: Graph Bert Node Classification (Cora, Citeseer, and Pubmed)
"""


import argparse
from tokenize import group
import torch

from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings

def run_fine_tuning_on_node_classification(dataset: str, lr: float, k: int, max_epoch: int, hidden_size: int, attention_head: int, hidden_layer: int, residual_type: str, device: str, group_name: str, run_index: int):
    """Fine-Tuning Task 1: Graph-Bert Node Classification (Cora, Citeseer, and Pubmed)
    
    Run Graph-Bert fine tuning on Node Classifcation with the given configuration.

    Parameters:
    -        dataset (str): the dataset to run the experiment on.
    -           lr (float): optimizer learning rate
    -              k (int): the size of surrounding context
    -      max_epoch (int): training budget in epochs
    -    hidden_size (int): feature space size for node embeddings
    - attention_head (int): number of attention heads
    -   hidden_layer (int): number of hidden layers in Graph-Bert
    -  residual_type (str): graph residual type
    -         device (str): device for training Graph-Bert
    -     group_name (str): group name for a group of experiments
    -      run_index (int): index for repeated runs of the same experiment

    Returns:
    - Absolutely nothing!

    """

    if dataset == 'cora':
        NUM_CLASS = 7
        NUM_FEATURE = 1433
        GRAPH_SIZE = 2708
    elif dataset == 'citeseer':
        NUM_CLASS = 6
        NUM_FEATURE = 3703
        GRAPH_SIZE = 3312
    elif dataset == 'pubmed':
        NUM_CLASS = 3
        NUM_FEATURE = 500
        GRAPH_SIZE = 19717
    elif dataset == 'ogbn-arxiv':
        NUM_CLASS = 40
        NUM_FEATURE = 128
        GRAPH_SIZE = 169343
    else:
        assert False, f"Input dataset {dataset} not supported."
    
    print(f"************ Start ************")
    print(f"'GrapBert fine tuning experiment {group_name} on dataset: {dataset}, k: {k}, lr: {lr}, max_train_epoch {max_epoch}, hidden dimension: {hidden_size}, hidden layer: {hidden_layer}, attention head: {attention_head}, residual: {residual_type}, device: {device}, run_index {run_index}.")
    
    # ---- objection initialization setction ---------------
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset + '/'
    data_obj.dataset_name = dataset
    data_obj.k = k
    data_obj.load_all_tag = True
    data_obj.preprocessing = False

    bert_config = GraphBertConfig(
                                  residual_type=residual_type,
                                  k=k,
                                  x_size=NUM_FEATURE,
                                  y_size=NUM_CLASS,
                                  hidden_size=hidden_size,
                                  intermediate_size=hidden_size,
                                  num_attention_heads=attention_head,
                                  num_hidden_layers=hidden_layer,
                                  train_device=device
                                 )
                                 
    method_obj = MethodGraphBertNodeClassification(bert_config)
    #---- set to false to run faster ----
    method_obj.spy_tag = True
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr

    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = f"./result/GraphBert/{group_name}_learning_record/"
    result_obj.result_destination_file_name = f"{dataset}_k_{k}_run_{run_index}.pkl"

    setting_obj = Settings()

    evaluate_obj = None
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
    # ------------------------------------------------------

    method_obj.save_pretrained(f"./result/PreTrained_GraphBert/{dataset}/{group_name}/node_classification_complete_model_run_{run_index}/")
    print('************ Finish ************')


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True, help='The dataset to run the experiment on.')
    parser.add_argument('--lr', required=False, default=0.001, help='Optimizer learning rate.')
    parser.add_argument('--k', required=True, help='The size of surrounding context.')
    parser.add_argument('--max_epoch', required=True, help='Training budget in epochs.')
    parser.add_argument('--hidden_size', required=False, default=32, help='Feature space size for node embeddings.')
    parser.add_argument('--attention_head', required=False, default=2, help='Number of attention heads.')
    parser.add_argument('--hidden_layer', required=False, default=2, help='Number of hidden layers in Graph-Bert.')
    parser.add_argument('--residual_type', required=False, default='graph_raw', help='Graph residual type.')
    parser.add_argument('--group_name', required=True, help='Group name for experiment. Experiments results of the same group will be saved at the same place.')
    parser.add_argument('--run_index', required=True, help='Index for repeated runs of the same experiment.')

    args = parser.parse_args()

    # Dataset for the experiment
    DATASET = str(args.dataset)
    # Optimizer learning rate
    LR = float(args.lr)
    # Size of surrounding context
    K = int(args.k)
    # Traning epochs
    MAX_EPOCH = int(args.max_epoch)
    # Feature space size for node embeddings
    HIDDEN_SIZE = int(args.hidden_size)
    # Number of attention heads
    NUM_ATTENTION_HEAD = int(args.attention_head)
    # Number of hidden layers in Graph-Bert
    NUM_HIDDEN_LAYER = int(args.hidden_layer)
    # Graph residual_type
    RESIDUAL_TYPE = str(args.residual_type)
    # Experiment group name
    GROUP_NAME = str(args.group_name)
    # Index for repeated runs of the same experiment config.
    RUN_IND = int(args.run_index)
    # DEVICE to use for training
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test the argparse with the following line
    # 'python3 batch_script_3_fine_tuning.py --dataset cora --lr 0.01 --k 7 --max_epoch 150 --group_name lillpop --run_index 0'
  

    run_fine_tuning_on_node_classification(
                                           dataset=DATASET,
                                           lr=LR,
                                           k=K,
                                           max_epoch=MAX_EPOCH,
                                           hidden_size=HIDDEN_SIZE,
                                           attention_head=NUM_ATTENTION_HEAD,
                                           hidden_layer=NUM_HIDDEN_LAYER,
                                           residual_type=RESIDUAL_TYPE,
                                           device=DEVICE,
                                           group_name=GROUP_NAME,
                                           run_index=RUN_IND
                                        )


if __name__ == "__main__":
    main()
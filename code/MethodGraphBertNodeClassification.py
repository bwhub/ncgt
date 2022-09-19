import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time
import numpy as np
import sklearn
import scipy

from code.EvaluateAcc import EvaluateAcc    


BertLayerNorm = torch.nn.LayerNorm

class MethodGraphBertNodeClassification(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True

    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config):
        super(MethodGraphBertNodeClassification, self).__init__(config)
        self.config = config
        # device used for training models
        self.train_device = self.config.train_device
        self.bert = MethodGraphBert(config).to(self.train_device)
        self.res_h = torch.nn.Linear(config.x_size, config.hidden_size).to(self.train_device)
        self.res_y = torch.nn.Linear(config.x_size, config.y_size).to(self.train_device)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.y_size).to(self.train_device)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):
        residual_h, residual_y = self.residual_term()
        if idx is not None:
            if residual_h is None:
                outputs = self.bert(raw_features[idx], wl_role_ids[idx], init_pos_ids[idx], hop_dis_ids[idx], residual_h=None)
            else:
                outputs = self.bert(raw_features[idx], wl_role_ids[idx], init_pos_ids[idx], hop_dis_ids[idx], residual_h=residual_h[idx])
                residual_y = residual_y[idx]
        else:
            if residual_h is None:
                outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=None)
            else:
                outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=residual_h)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        labels = self.cls_y(sequence_output)

        if residual_y is not None:
            labels += residual_y

        return F.log_softmax(labels, dim=1)

    def residual_term(self):
        if self.config.residual_type == 'none':
            return None, None
        elif self.config.residual_type == 'raw':
            return self.res_h(self.data['X']), self.res_y(self.data['X'])
        elif self.config.residual_type == 'graph_raw':
            return torch.spmm(self.data['A'], self.res_h(self.data['X'])), torch.spmm(self.data['A'], self.res_y(self.data['X']))

    def train_model(self, max_epoch):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        accuracy = EvaluateAcc('', '')

        max_score = 0.0
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # -------------------------

            self.train()
            optimizer.zero_grad()

            output = self.forward(
                                    self.data['raw_embeddings'].to(self.train_device),
                                    self.data['wl_embedding'].to(self.train_device),
                                    self.data['int_embeddings'].to(self.train_device),
                                    self.data['hop_embeddings'].to(self.train_device),
                                    self.data['idx_train'].to(self.train_device)
                                )
            
            loss_train = F.cross_entropy(output, self.data['y'][self.data['idx_train']])
            # TODO: remove
            # print(f'self.data["idx_train"] is of type {type(self.data["idx_train"])}, and shape {self.data["idx_train"].shape}')
            # print(f'Content {self.data["idx_train"]}.')
            # assert False

            accuracy.data = {'true_y': self.data['y'][self.data['idx_train']], 'pred_y': output.max(1)[1]}
            # print(f"The shape of true_y is {self.data['y'][self.data['idx_train']].shape}, the shape of pred_y is {output.detach().cpu().shape}.")
            # assert False
            # print(f"ture_y[:3] is {self.data['y'][self.data['idx_train']][:3]}.")
            # print(f"pred_y[:3] is {output[:1]}")
            # print(f"pred_y[:3].max(1)[1] is {output.max(1)}")

            # assert False

            acc_train = accuracy.evaluate()
            f1_train = sklearn.metrics.f1_score(accuracy.data['true_y'].cpu(), accuracy.data['pred_y'].cpu(), average='weighted')
            auc_train = sklearn.metrics.roc_auc_score(
                                                        y_true=self.data['y'][self.data['idx_train']].cpu(),
                                                        y_score=scipy.special.softmax(output.detach().cpu(), axis=1),
                                                        multi_class='ovr',
                                                        average='weighted'
                                                    )
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'], self.data['hop_embeddings'], self.data['idx_val'])
            # ## DEBUG
            # print(f"output is of type {type(output)} and shape {output.shape}.")
            # print(f"y_true is of type {type(self.data['y'][self.data['idx_val']])} and shape {len(set(self.data['y'][self.data['idx_val']].numpy()))}.")
            # assert False, "Debug"
            # ## DEBUG
            loss_val = F.cross_entropy(output, self.data['y'][self.data['idx_val']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_val']],
                             'pred_y': output.max(1)[1]}
            acc_val = accuracy.evaluate()
            f1_val = sklearn.metrics.f1_score(accuracy.data['true_y'].cpu(), accuracy.data['pred_y'].cpu(), average='weighted')
            auc_val = sklearn.metrics.roc_auc_score(
                                            y_true=self.data['y'][self.data['idx_val']].cpu(),
                                            y_score=scipy.special.softmax(output.detach().cpu(), axis=1),
                                            multi_class='ovr',
                                            average='weighted'
                                        )

            #-------------------------
            #---- keep records for drawing convergence plots ----
            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'], self.data['hop_embeddings'], self.data['idx_test'])
            loss_test = F.cross_entropy(output, self.data['y'][self.data['idx_test']])
            accuracy.data = {'true_y': self.data['y'][self.data['idx_test']],
                             'pred_y': output.max(1)[1]}
            acc_test = accuracy.evaluate()
            f1_test = sklearn.metrics.f1_score(accuracy.data['true_y'].cpu(), accuracy.data['pred_y'].cpu(), average='weighted')
            auc_test = sklearn.metrics.roc_auc_score(
                                            y_true=self.data['y'][self.data['idx_test']].cpu(),
                                            y_score=scipy.special.softmax(output.detach().cpu(), axis=1),
                                            multi_class='ovr',
                                            average='weighted'
                                        )

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(),
                                                'acc_train': acc_train.item(),
                                                'f1_train': f1_train,
                                                'auc_train': auc_train,
                                                'loss_val': loss_val.item(),
                                                'acc_val': acc_val.item(),
                                                'f1_val': f1_val,
                                                'auc_val': auc_val,
                                                'loss_test': loss_test.item(),
                                                'acc_test': acc_test.item(),
                                                'f1_test': f1_test,
                                                'auc_test': auc_test,
                                                'time': time.time() - t_epoch_begin
                                                }

            # -------------------------
            if epoch % 10 == 0:
                # print('Epoch: {:04d}'.format(epoch + 1),
                #       'loss_train: {:.4f}'.format(loss_train.item()),
                #       'acc_train: {:.4f}'.format(acc_train.item()),
                #       'loss_val: {:.4f}'.format(loss_val.item()),
                #       'acc_val: {:.4f}'.format(acc_val.item()),
                #       'loss_test: {:.4f}'.format(loss_test.item()),
                #       'acc_test: {:.4f}'.format(acc_test.item()),
                #       'time: {:.4f}s'.format(time.time() - t_epoch_begin))
                progress_str = (
                                f"Epoch: {epoch + 1:4d}"
                                f"\n\t loss_train: {loss_train.item():.4f}, acc_train: {acc_train.item():.4f}, f1_train: {f1_train:.4f}, auc_train: {auc_train:.4f}"
                                f"\n\t loss_val  : {loss_val.item():.4f}, acc_val  : {acc_val.item():.4f}, f1_val  : {f1_val:.4f}, auc_val  : {auc_val:.4f}"
                                f"\n\t loss_test : {loss_test.item():.4f}, acc_test : {acc_test.item():.4f}, f1_test : {f1_test:.4f}, auc_test : {auc_test:.4f}"
                            )
                print(progress_str)

        min_loss_epoch_index = np.argmin([self.learning_record_dict[epoch]['loss_test'] for epoch in self.learning_record_dict])
        self.summary_str = (
                       f"\n-------------- Optimization Finished! --------------"
                       f"\nTotal time elapsed: {(time.time() - t_begin):.4f}s\n"
                       f"Minimum test loss was found at epoch {min_loss_epoch_index}: \n\t"
                       f"loss_test {self.learning_record_dict[min_loss_epoch_index]['loss_test']:.4f}, "
                       f"acc_test {self.learning_record_dict[min_loss_epoch_index]['acc_test']:.4f}, "
                       f"f1_test {self.learning_record_dict[min_loss_epoch_index]['f1_test']:.4f}, "
                       f"auc_test {self.learning_record_dict[min_loss_epoch_index]['auc_test']:.4f}.\n"
                       f"Best seen metrics across different epochs: \n\t"
                       f"best_loss {np.min([self.learning_record_dict[epoch]['loss_test'] for epoch in self.learning_record_dict]):.4f}, "
                       f"best_acc {np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict]):.4f}, "
                       f"best_f1 {np.max([self.learning_record_dict[epoch]['f1_test'] for epoch in self.learning_record_dict]):.4f}, "
                       f"best_auc {np.max([self.learning_record_dict[epoch]['auc_test'] for epoch in self.learning_record_dict]):.4f}, "
                      )
        print(self.summary_str)

        # TODO: Is the return value used any way?
        return time.time() - t_begin, np.max([self.learning_record_dict[epoch]['acc_test'] for epoch in self.learning_record_dict])


    def run(self):

        self.train_model(self.max_epoch)

        result_dict = {}
        result_dict['learning_record_dict'] = self.learning_record_dict
        
        key_mask_out = ['data', 'spy_tag', 'name_or_path', 'config']
        experiment_config_key_list = [k for k in self.__dict__.keys() if not(k.startswith('_') or k in key_mask_out)]
        result_dict['experiment_config'] = {k: self.__dict__[k] for k in experiment_config_key_list}
        result_dict['experiment_config']['config'] = {k: self.config.__dict__[k] for k in self.config.__dict__.keys() if not k.startswith('_')}
        print(f"Saving configuration with keys {experiment_config_key_list}")
        
        return result_dict
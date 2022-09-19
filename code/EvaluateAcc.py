'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score


class EvaluateAcc(evaluate):
    data = None
    
    def evaluate(self):
        # We need to move data to cpu since sklearn.metrics.accuracy_score does not support cuda.
        return accuracy_score(self.data['true_y'].cpu(), self.data['pred_y'].cpu())

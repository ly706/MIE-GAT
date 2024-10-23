import torch
from sklearn.metrics import roc_auc_score,accuracy_score
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = float(self.sum) / self.count

class Metric(object):
    def __init__(self,threshold=0.5):
        self.reset()
        self.threshold=threshold
        self.epsilon=1e-5

    def reset(self):
        self.true = []
        self.pre_round= []
        self.id=[]

    def update(self,t,p):
        self.true=self.true+t
        self.pre_round=self.pre_round+p

    def get_acc(self):
        acc=accuracy_score(self.true,self.pre_round)
        # print("acc:",acc)
        return acc

    def get_true(self):
        return self.true;

    def get_confusion_matrix(self):
        tp=0
        fp=0
        fn=0
        tn=0
        for i in range(len(self.pre_round)):
            if self.pre_round[i] == 1 and self.true[i] == 1:
                tp = tp + 1
            elif self.pre_round[i] == 1 and self.true[i] == 0:
                fp = fp + 1
            elif self.pre_round[i] == 0 and self.true[i] == 1:
                fn = fn + 1
            elif self.pre_round[i] == 0 and self.true[i] == 0:
                tn = tn + 1
        return [[tp,fp],[fn,tn]]

    def get_metric(self):
        [[tp,fp],[fn,tn]]=self.get_confusion_matrix()
        result = {}
        result['recall'] = (float(tp)+self.epsilon) / (tp + fn+self.epsilon)
        result['spec'] = (float(tn)+self.epsilon) / (tn + fp+self.epsilon)
        result['pre'] = (float(tp)+self.epsilon) / (tp + fp+self.epsilon)
        result['f1'] = (2 * result['recall'] * result['pre']+self.epsilon) / (result['recall'] + result['pre']+self.epsilon)
        # print("metric:",result)
        return [result['recall'],result['spec'],result['pre'],result['f1']]

class Metric_auc(object):
    def __init__(self,threshold=0.5,dataset=None):
        self.reset()
        self.threshold=threshold
        self.dataset = dataset
        self.epsilon=1e-5

    def reset(self):
        self.pre = []
        self.true = []
        self.pre_round= []
        self.id_index=[]
        self.id=[]

        self.g_pre = []
        self.g_true = []
        self.g_pre_round = []
        self.rank_pre_round = []
        self.g_id = []
        # self.pre_score = [{0:, 1:}]

    def update(self,t,p):
        # for i in range(len(p)):
        #     p[i] = round(p[i], 4)
        self.pre=self.pre+p
        self.true=self.true+t
        self.round_pre(p)

    def update_id(self,id_index):
        self.id_index=self.id_index+id_index
        # self.id = self.id_index
        self.id = [self.dataset[img_idx]['id'] for img_idx in self.id_index]
        self.g_id = list(set(self.id))
        self.g_pre = []
        self.g_true = []
        self.g_pre_round = []
        self.rank_pre_round = []
        for g in self.g_id:
            indexs = [i for i,x in enumerate(self.id) if x==g]
            self.g_true.append(self.true[indexs[0]])
            all_pre = [self.pre[index] for index in indexs]
            pre_mean = sum(all_pre) / float(len(all_pre))
            self.g_pre.append(pre_mean)
            if pre_mean < self.threshold:
                self.g_pre_round.append(0)
            else:
                self.g_pre_round.append(1)

            all_pre_round = [self.pre_round[index] for index in indexs]
            pre_round = 0 if all_pre_round.count(0)>all_pre_round.count(1) else 1
            self.rank_pre_round.append(pre_round)

    def round_pre(self,p):
        for i in p:
            if i>self.threshold:
                self.pre_round.append(1)
            else:
                self.pre_round.append(0)

    def get_acc(self):
        pre_acc = accuracy_score(self.true, self.pre_round)
        acc=accuracy_score(self.g_true,self.g_pre_round)
        rank_acc = accuracy_score(self.g_true, self.rank_pre_round)
        # print("pre_acc:", pre_acc)
        # print("acc:",acc)
        # print("rank_acc:", rank_acc)
        return acc

    def get_auc(self):
        pre_auc = roc_auc_score(self.true, self.pre)
        auc=roc_auc_score(self.g_true,self.g_pre)
        # print("pre_auc:", pre_auc)
        # print("auc:",auc)
        return auc

    def get_pre(self):
        return self.g_pre;

    def get_pre_round(self):
        return self.g_pre_round;

    def get_true(self):
        return self.g_true;

    def get_id(self):
        # print(self.g_id)
        return self.g_id;

    def get_confusion_matrix(self):
        tp=0
        fp=0
        fn=0
        tn=0
        for i in range(len(self.g_pre_round)):
            if self.g_pre_round[i] == 1 and self.g_true[i] == 1:
                tp = tp + 1
            elif self.g_pre_round[i] == 1 and self.g_true[i] == 0:
                fp = fp + 1
            elif self.g_pre_round[i] == 0 and self.g_true[i] == 1:
                fn = fn + 1
            elif self.g_pre_round[i] == 0 and self.g_true[i] == 0:
                tn = tn + 1
        return [[tp,fp],[fn,tn]]

    def get_metric(self):
        [[tp,fp],[fn,tn]]=self.get_confusion_matrix()
        result = {}
        result['recall'] = (float(tp)+self.epsilon) / (tp + fn+self.epsilon)
        result['spec'] = (float(tn)+self.epsilon) / (tn + fp+self.epsilon)
        result['pre'] = (float(tp)+self.epsilon) / (tp + fp+self.epsilon)
        result['f1'] = (2 * result['recall'] * result['pre']+self.epsilon) / (result['recall'] + result['pre']+self.epsilon)
        # print("metric_auc:",result)
        return [result['recall'],result['spec'],result['pre'],result['f1']]

class Metric_auc_val(object):
    def __init__(self,threshold=0.5,dataset=None):
        self.reset()
        self.threshold=threshold
        self.dataset = dataset
        self.epsilon=1e-5

    def reset(self):
        self.pre = []
        self.true = []
        self.pre_round= []
        self.id=[]
        self.id_index = []
        self.g_pre = []
        self.g_pre_round = []
        self.g_id = []
        # self.pre_score = [{0:, 1:}]

    def update(self,t,p):
        # for i in range(len(p)):
        #     p[i] = round(p[i], 4)
        self.pre=self.pre+p
        self.true=self.true+t
        self.round_pre(p)

    def update_id(self,id_index):
        self.id_index = self.id_index + id_index
        self.id = [self.dataset[img_idx]['id'] for img_idx in self.id_index]

    def round_pre(self,p):
        for i in p:
            if i>self.threshold:
                self.pre_round.append(1)
            else:
                self.pre_round.append(0)

    def get_acc(self):
        acc=accuracy_score(self.true,self.pre_round)
        # print("acc:",acc)
        return acc

    def get_auc(self):
        auc=roc_auc_score(self.true,self.pre)
        # print("auc:",auc)
        return auc

    def get_pre(self):
        return self.pre;

    def get_pre_round(self):
        return self.pre_round;

    def get_true(self):
        return self.true;

    def get_id(self):
        return self.id;

    def get_confusion_matrix(self):
        tp=0
        fp=0
        fn=0
        tn=0
        for i in range(len(self.pre_round)):
            if self.pre_round[i] == 1 and self.true[i] == 1:
                tp = tp + 1
            elif self.pre_round[i] == 1 and self.true[i] == 0:
                fp = fp + 1
            elif self.pre_round[i] == 0 and self.true[i] == 1:
                fn = fn + 1
            elif self.pre_round[i] == 0 and self.true[i] == 0:
                tn = tn + 1
        return [[tp,fp],[fn,tn]]

    def get_metric(self):
        [[tp,fp],[fn,tn]]=self.get_confusion_matrix()
        result = {}
        result['recall'] = (float(tp)+self.epsilon) / (tp + fn+self.epsilon)
        result['spec'] = (float(tn)+self.epsilon) / (tn + fp+self.epsilon)
        result['pre'] = (float(tp)+self.epsilon) / (tp + fp+self.epsilon)
        result['f1'] = (2 * result['recall'] * result['pre']+self.epsilon) / (result['recall'] + result['pre']+self.epsilon)
        # print("metric_auc:",result)
        return [result['recall'],result['spec'],result['pre'],result['f1']]

class Metric_auc_gat(object):
    def __init__(self,threshold=0.5):
        self.reset()
        self.threshold=threshold
        self.epsilon=1e-5

    def reset(self):
        self.pre = []
        self.true = []
        self.pre_round= []
        self.id=[]
        self.g_pre = []
        self.g_pre_round = []
        self.g_id = []
        # self.pre_score = [{0:, 1:}]

    def update(self,t,p):
        # for i in range(len(p)):
        #     p[i] = round(p[i], 4)
        self.pre=self.pre+p
        self.true=self.true+t
        self.round_pre(p)

    def update_id(self,id):
        # Check if id is a tensor and convert it to a list if it is
        if isinstance(id, torch.Tensor):
            id = id.tolist()  # Convert tensor to list
        self.id=self.id+id

    def round_pre(self,p):
        for i in p:
            if i>self.threshold:
                self.pre_round.append(1)
            else:
                self.pre_round.append(0)

    def get_acc(self):
        acc=accuracy_score(self.true,self.pre_round)
        # print("acc:",acc)
        return acc

    def get_auc(self):
        auc=roc_auc_score(self.true,self.pre)
        # print("auc:",auc)
        return auc

    def get_pre(self):
        return self.pre;

    def get_pre_round(self):
        return self.pre_round;

    def get_true(self):
        return self.true;

    def get_id(self):
        return self.id;

    def get_confusion_matrix(self):
        tp=0
        fp=0
        fn=0
        tn=0
        for i in range(len(self.pre_round)):
            if self.pre_round[i] == 1 and self.true[i] == 1:
                tp = tp + 1
            elif self.pre_round[i] == 1 and self.true[i] == 0:
                fp = fp + 1
            elif self.pre_round[i] == 0 and self.true[i] == 1:
                fn = fn + 1
            elif self.pre_round[i] == 0 and self.true[i] == 0:
                tn = tn + 1
        return [[tp,fp],[fn,tn]]

    def get_metric(self):
        [[tp,fp],[fn,tn]]=self.get_confusion_matrix()
        result = {}
        result['recall'] = (float(tp)+self.epsilon) / (tp + fn+self.epsilon)
        result['spec'] = (float(tn)+self.epsilon) / (tn + fp+self.epsilon)
        result['pre'] = (float(tp)+self.epsilon) / (tp + fp+self.epsilon)
        result['f1'] = (2 * result['recall'] * result['pre']+self.epsilon) / (result['recall'] + result['pre']+self.epsilon)
        # print("metric_auc:",result)
        return [result['recall'],result['spec'],result['pre'],result['f1']]

def avgResult(result):
    result_array = np.array(result)
    avg_result = np.average(result_array, axis=0)
    return avg_result

class AvgResult(object):
    def __init__(self):
        self.result=[]

    def update(self,r):
        self.result.append(r)

    def get(self):
        self.result_array=np.array(self.result)
        self.avg_result=np.average(self.result_array,axis=0)
        return self.avg_result


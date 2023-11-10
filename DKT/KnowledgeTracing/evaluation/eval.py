import sys
sys.path.append('../')
import tqdm
import torch
import torch.nn as nn
from sklearn import metrics
from torch.autograd import Variable
from Constant import Constants as C

def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(), prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    recall = metrics.recall_score(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    precision = metrics.precision_score(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())

    print('auc:' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) + ' precision: ' + str(precision) + '\n')
    return auc, f1, recall, precision
class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

    def forward(self, pred, batch):
        loss = torch.Tensor([0.0]).to(batch.device)
        for student in range(pred.shape[0]):
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]]).to(temp.device)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[1:]
            for i in range(len(p)):
                if p[i] > 0: # mask all 0 padding data for evaluation
                    # breakpoint()
                    loss = loss - (a[i]*torch.log(p[i]) + (1-a[i])*torch.log(1-p[i]))
        return loss

def train_epoch(model, trainLoader, optimizer, loss_func, device):
    # pred_list[]
    model.to(device)
    tot_loss=0
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        #breakpoint()
        pred = model(batch)
        #breakpoint()
        loss = loss_func(pred, batch)
        optimizer.zero_grad()
        loss.backward()
        #breakpoint()
        optimizer.step()
        tot_loss+=loss.item()
    tot_loss=tot_loss/len(trainLoader) # len(trainLoader) is how many batches
    return model, optimizer, tot_loss


def test_epoch(model, testLoader, loss_func, device):
    model.to(device)
    tot_loss=0
    gold_epoch = torch.Tensor([]).to(device)
    pred_epoch = torch.Tensor([]).to(device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:    ', mininterval=2):
        batch = batch.to(device)
        pred = model(batch)# output: torch.size([batch_size, sequence_length, n_questions])
        loss = loss_func(pred, batch)
        tot_loss += loss.item()
        # data = batch # batch input: torch.size([batch_size,max_step,2*n_questions])
        # print(data)
        # breakpoint()
        for student in range(pred.shape[0]):
            temp_pred = torch.Tensor([]).to(device)
            temp_gold = torch.Tensor([]).to(device)
            delta = batch[student][:,0:C.NUM_OF_QUESTIONS] + batch[student][:,C.NUM_OF_QUESTIONS:]
            temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t()) # mask 0 padding
            index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]]).to(temp.device)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:C.NUM_OF_QUESTIONS] - batch[student][:, C.NUM_OF_QUESTIONS:]).sum(1) + 1)//2)[1:]
            for i in range(len(p)):
                if p[i] > 0:
                    temp_pred = torch.cat([temp_pred,p[i:i+1]])
                    temp_gold = torch.cat([temp_gold, a[i:i+1]])
            pred_epoch = torch.cat([pred_epoch, temp_pred])
            gold_epoch = torch.cat([gold_epoch, temp_gold])
          # breakpoint()
    tot_loss = tot_loss / len(testLoader)
    return pred_epoch, gold_epoch, tot_loss # torch.Size([2976])


def train(trainLoaders, model, optimizer, lossFunc,device):
    train_loss=0
    for i in range(len(trainLoaders)): # len(trainLoaders)=1
        model, optimizer, train_loss = train_epoch(model, trainLoaders[i], optimizer, lossFunc,device)
        #breakpoint()
    return model, optimizer, train_loss

def test(testLoaders, model,loss_func,device):
    ground_truth = torch.Tensor([]).to(device)
    prediction = torch.Tensor([]).to(device)
    val_loss=0
    for i in range(len(testLoaders)):
        pred_epoch, gold_epoch, val_loss = test_epoch(model, testLoaders[i],loss_func,device)
        prediction = torch.cat([prediction, pred_epoch]) # torch size([39200])
        ground_truth = torch.cat([ground_truth, gold_epoch])
        #breakpoint()
    auc, f1, recall, precision=performance(ground_truth, prediction)
    return auc, f1, recall, precision, val_loss
from torch_geometric.datasets import PPI
import torch
import torch.nn.functional as F
from args import *
from torch import nn
from torch.nn import Linear
from model import *
from utils import precompute_dist_data,preselect_anchor
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import numpy as np
args = make_args()
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')
device = torch.device('cpu')
class PGNN_Classification(nn.Module):
    def __init__(self,input_dim,feature_dim=args.feature_dim,hidden_dim=args.hidden_dim,output_dim=64,layer_num=args.layer_num,dropout=args.dropout,agg="mean"):
        super(PGNN_Classification,self).__init__()
        self.PGNN =PGNN(input_dim=input_dim,feature_dim=feature_dim,hidden_dim=hidden_dim,output_dim=output_dim,feature_pre=False,layer_num=layer_num,dropout=dropout,agg=agg).to(device)
        self.fc = nn.Linear(64,121)
    def forward(self,x):
        x = self.PGNN(x)
        print(x.shape)
        x = self.fc(x)
        return x


ppi_train = PPI(root="data/ppi",split="train")
ppi_val = PPI(root="data/ppi",split="val")
ppi_test = PPI(root="data/ppi",split="test")

print(len(ppi_train),len(ppi_val),len(ppi_test))
data_list = []
data_val = []
data_test = []
for i in range(0,len(ppi_train)):
    d = ppi_train[i]
    dists = precompute_dist_data(d.edge_index.numpy(),d.num_nodes,1)
    d.dists = torch.from_numpy(dists).float()
    data_list.append(d)

    if(i==3):
        break

for i in range(0,len(ppi_val)):
    break
    d = ppi_val[i]
    dists = precompute_dist_data(d.edge_index.numpy(),d.num_nodes,args.approximate)
    d.dists = torch.from_numpy(dists).float()
    data_val.append(d)

    d = ppi_test[i]
    dists = precompute_dist_data(d.edge_index.numpy(), d.num_nodes, args.approximate)
    d.dists = torch.from_numpy(dists).float()
    data_test.append(d)

print("Distances Calculated")
input_dim = 50
output_dim = 64
loss_fn = nn.BCEWithLogitsLoss()
model = PGNN_Classification(input_dim = input_dim,output_dim=output_dim,dropout=False,layer_num=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
out_act = nn.Softmax(dim=0)
print(model)
for epoch in range(0,args.epoch_num+1):
    model.train()
    for i,data in enumerate(data_list):
        if(args.permute):
            print("Y")
            preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
        print(data.dists_max.shape)
        out = model(data)

        loss = loss_fn(out,data.y)
        loss.backward()
        print("Loss : " ,loss.item())
        optimizer.step()
        optimizer.zero_grad()

    train_loss = 0
    train_acc = 0
    train_prec = 0
    train_recall = 0
    train_f1 = 0
    train_auc = 0

    val_loss = 0
    val_acc = 0
    val_prec = 0
    val_recall = 0
    val_f1 = 0
    val_auc = 0

    test_loss = 0
    test_acc = 0
    test_prec = 0
    test_recall = 0
    test_f1 = 0
    test_auc = 0

    if(epoch%args.epoch_log == 0):
        model.eval()
        for i,data in enumerate(data_list):
            pred = out_act(model(data))
            label = data.y
            train_loss += loss_fn(pred,label).data.numpy()
            pred_numpy = np.where(pred.detach().numpy()>=0.5,1,0)
            label_numpy = data.y.numpy()
            train_acc += accuracy_score(pred_numpy,label_numpy)
            train_prec += precision_score(pred_numpy,label_numpy,average="micro")
            try:
                train_auc += roc_auc_score(pred_numpy,label_numpy,average="micro")
            except:
                pass
            train_recall += recall_score(pred_numpy,label_numpy,average="micro")
            train_f1 += f1_score(pred_numpy,label_numpy,average="micro")

        train_loss /= i
        train_acc /= i
        train_prec /= i
        train_recall /= i
        train_f1 /= i
        train_auc /= i
        print("Epoch : {} , Train  Loss : {} , Accuracy : {}, Precision : {}, Recall : {} ,F1 : {},  AUC : {} ".format(epoch,train_loss,train_acc,train_prec,train_recall,train_f1,train_auc))




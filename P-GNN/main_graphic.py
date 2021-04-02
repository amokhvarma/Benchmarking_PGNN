from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from tensorboardX import SummaryWriter
import os
from args import *
from model import *
from utils import *
from dataset import *
import time
import numpy as np
import torch
if not os.path.isdir('results'):
    os.mkdir('results')
# args
args = make_args()
print(args)
np.random.seed(123)
np.random.seed()
writer_train = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_train')
writer_val = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_val')
writer_test = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_test')

print(args.gpu)
# set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')
a = None
model = None
for task in ['link']:
    args.task = task
    if args.dataset=='All':
        if task == 'link':
            datasets_name = ['grid','communities','ppi']
        else:
            datasets_name = ['communities', 'email', 'protein']
    else:
        datasets_name = [args.dataset]
    for dataset_name in datasets_name:
        # if dataset_name in ['communities','grid']:
        #     args.cache = False
        # else:
        #     args.epoch_num = 401
        #     args.cache = True
        results_auc,results_acc,results_prec,results_rec,results_f1 = [],[],[],[],[]
        for repeat in range(args.repeat_num):
            result_val = []
            result_auc,result_acc,result_prec,result_rec,result_f1 = [],[],[],[],[]
            time1 = time.time()
            data_list = get_tg_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature)
            time2 = time.time()
            print(dataset_name, 'load time',  time2-time1)

            num_features = data_list[0].x.shape[1]
            num_node_classes = None
            num_graph_classes = None
            if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
                num_node_classes = max([data.y.max().item() for data in data_list])+1
            if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
                num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list])+1
            print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features, 'Node Class', num_node_classes, 'Graph Class', num_graph_classes)
            nodes = [data.num_nodes for data in data_list]
            edges = [data.num_edges for data in data_list]
            print('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes)/len(nodes)))
            print('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges)/len(edges)))

            args.batch_size = min(args.batch_size, len(data_list))
            print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

            # data
            for i,data in enumerate(data_list):
                preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
                data = data.to(device)
                data_list[i] = data

            # model
            input_dim = num_features
            output_dim = args.output_dim
            model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                        hidden_dim=args.hidden_dim, output_dim=output_dim,
                        feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout,agg=args.agg).to(device)
            # loss
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            if 'link' in args.task:
                loss_func = nn.BCEWithLogitsLoss()
                out_act = nn.Sigmoid()


            for epoch in range(args.epoch_num):
                if epoch==200:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 10
                model.train()
                optimizer.zero_grad()
                shuffle(data_list)
                effective_len = len(data_list)//args.batch_size*len(data_list)
                for id, data in enumerate(data_list[:effective_len]):
                    if args.permute:
                        preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
                    out = model(data)
                    # get_link_mask(data,resplit=False)  # resample negative links
                    edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0,:]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1,:]).long().to(device))
                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    label_positive = torch.ones([data.mask_link_positive_train.shape[1],], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_link_negative_train.shape[1],], dtype=pred.dtype)
                    label = torch.cat((label_positive,label_negative)).to(device)
                    loss = loss_func(pred, label)

                    # update
                    loss.backward()
                    if id % args.batch_size == args.batch_size-1:
                        if args.batch_size>1:
                            # if this is slow, no need to do this normalization
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad /= args.batch_size
                        optimizer.step()
                        optimizer.zero_grad()


                if epoch % args.epoch_log == 0:
                    # evaluate
                    model.eval()
                    loss_train = 0
                    loss_val = 0
                    loss_test = 0
                    correct_train = 0
                    all_train = 0
                    correct_val = 0
                    all_val = 0
                    correct_test = 0
                    all_test = 0
                    auc_train = 0
                    auc_val = 0
                    auc_test = 0
                    emb_norm_min = 0
                    emb_norm_max = 0
                    emb_norm_mean = 0
                    accuracy_train=0
                    accuracy_val=0
                    accuracy_test=0
                    precision_train=0
                    precision_val=0
                    precision_test=0
                    recall_train=0
                    recall_val=0
                    recall_test=0
                    f1_train=0
                    f1_val=0
                    f1_test=0


                    for id, data in enumerate(data_list):
                        out = model(data)
                        emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
                        emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
                        emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()

                        # train
                        # get_link_mask(data, resplit=False)  # resample negative links
                        edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                        label = torch.cat((label_positive, label_negative)).to(device)
                        loss_train += loss_func(pred, label).cpu().data.numpy()
                        auc_train += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                        fpr, tpr, thresholds = roc_curve(label.flatten().cpu().numpy(),\
                                                         out_act(pred).flatten().data.cpu().numpy())
                        optimal_idx = np.argmax(tpr - fpr)
                        threshold = thresholds[optimal_idx]
                        #print(threshold)

                        label_train_numpy = np.where(label.flatten().cpu().numpy() > threshold , 1 , 0)
                        pred_train_numpy = np.where(out_act(pred).flatten().data.cpu().numpy()>threshold, 1 ,0)
                        accuracy_train += accuracy_score(label_train_numpy,pred_train_numpy)
                        precision_train += precision_score(label_train_numpy,pred_train_numpy)
                        recall_train += recall_score(label_train_numpy,pred_train_numpy)
                        f1_train += f1_score(label_train_numpy,pred_train_numpy)
                        # val
                        edge_mask_val = np.concatenate((data.mask_link_positive_val, data.mask_link_negative_val), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                        label = torch.cat((label_positive, label_negative)).to(device)
                        loss_val += loss_func(pred, label).cpu().data.numpy()
                        auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                        label_val_numpy = np.where(label.flatten().cpu().numpy()>threshold,1,0)
                        pred_val_numpy = np.where(out_act(pred).flatten().data.cpu().numpy()>threshold,1,0)
                        accuracy_val += accuracy_score(label_val_numpy, pred_val_numpy)
                        precision_val += precision_score(label_val_numpy, pred_val_numpy)
                        recall_val += recall_score(label_val_numpy, pred_val_numpy)
                        f1_val += f1_score(label_val_numpy, pred_val_numpy)

                        # test
                        edge_mask_test = np.concatenate((data.mask_link_positive_test, data.mask_link_negative_test), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
                        label = torch.cat((label_positive, label_negative)).to(device)
                        loss_test += loss_func(pred, label).cpu().data.numpy()
                        auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                        label_test_numpy = np.where(label.flatten().cpu().numpy()>threshold,1,0)
                        pred_test_numpy = np.where(out_act(pred).flatten().data.cpu().numpy()>threshold,1,0)
                        accuracy_test += accuracy_score(label_test_numpy, pred_test_numpy)
                        precision_test += precision_score(label_test_numpy, pred_test_numpy)
                        recall_test += recall_score(label_test_numpy, pred_test_numpy)
                        f1_test += f1_score(label_test_numpy, pred_test_numpy)

                    loss_train /= id+1
                    loss_val /= id+1
                    loss_test /= id+1
                    emb_norm_min /= id+1
                    emb_norm_max /= id+1
                    emb_norm_mean /= id+1
                    auc_train /= id+1
                    auc_val /= id+1
                    auc_test /= id+1
                    accuracy_train /= id+1
                    accuracy_val /= id+1
                    accuracy_test /= id+1
                    precision_train /= id+1
                    precision_val /= id+1
                    precision_test /= id+1
                    recall_train /= id+1
                    recall_val /= id+1
                    recall_test /= id+1
                    f1_train /= id+1
                    f1_val /= id+1
                    f1_test /= id+1

                    print("\n",repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                          'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test))
                    print(repeat, epoch, 'Train Acc {:.4f}'.format(accuracy_train), 'Val Acc: {:.4f}'.format(accuracy_val),
                          'Test Acc: {:.4f}'.format(accuracy_test), 'Train prec: {:.4f}'.format(precision_train), \
                            'Val prec: {:.4f}'.format(precision_val),'Test prec: {:.4f}'.format(precision_test))
                    print(repeat, epoch, 'Train Rec {:.4f}'.format(recall_train),
                          'Val Rec: {:.4f}'.format(recall_val),
                          'Test Rec: {:.4f}'.format(recall_test), 'Train F1: {:.4f}'.format(f1_train), \
                          'Val F1: {:.4f}'.format(f1_val), 'Test F1: {:.4f}'.format(f1_test))

                    writer_train.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_train, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_train, epoch)
                    writer_val.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_val, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_val, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_test, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_test, epoch)

                    writer_train.add_scalar('repeat_' + str(repeat) + '/acc_' + dataset_name, accuracy_train, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/prec_' + dataset_name, precision_train, epoch)
                    writer_val.add_scalar('repeat_' + str(repeat) + '/acc_' + dataset_name, accuracy_val, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/prec_' + dataset_name, precision_val, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/acc_' + dataset_name, accuracy_test, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/prec_' + dataset_name, precision_val, epoch)

                    writer_train.add_scalar('repeat_' + str(repeat) + '/rec_' + dataset_name, recall_train, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/f1_' + dataset_name, f1_train, epoch)
                    writer_val.add_scalar('repeat_' + str(repeat) + '/rec_' + dataset_name, recall_val, epoch)
                    writer_train.add_scalar('repeat_' + str(repeat) + '/f1_' + dataset_name, f1_val, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/rec_' + dataset_name, recall_test, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/f1_' + dataset_name, f1_test, epoch)


                    writer_test.add_scalar('repeat_' + str(repeat) + '/emb_min_'+dataset_name, emb_norm_min, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/emb_max_'+dataset_name, emb_norm_max, epoch)
                    writer_test.add_scalar('repeat_' + str(repeat) + '/emb_mean_'+dataset_name, emb_norm_mean, epoch)

                    result_val.append(auc_val)
                    result_auc.append(auc_test)
                    result_acc.append(accuracy_test)
                    result_prec.append(precision_test)
                    result_rec.append(recall_test)
                    result_f1.append(f1_test)


            result_val = np.array(result_val)
            result_auc = np.array(result_auc)
            index = np.argmax(result_val)
            results_auc.append(result_auc[index])
            results_acc.append(result_acc[index])
            results_prec.append(result_prec[index])
            results_rec.append(result_rec[index])
            results_f1.append(result_f1[index])



        results_auc = np.array(results_auc)
        results_acc = np.array(results_acc)
        results_prec = np.array(results_prec)
        results_rec = np.array(results_rec)
        results_f1 = np.array(results_f1)
        print('-----------------Final-------------------')
        #print(results_mean, results_std)
        with open('results/{}_{}_{}_layer{}_approximate{}.txt'.format(args.task,args.model,dataset_name,args.layer_num,args.approximate), 'w') as f:
            f.write('AUC : {}, {}\n'.format(np.mean(results_auc).round(6), np.std(results_auc).round(6)))
            f.write('ACC : {}, {}\n'.format(np.mean(results_acc).round(6), np.std(results_acc).round(6)))
            f.write('PREC : {}, {}\n'.format(np.mean(results_prec).round(6), np.std(results_prec).round(6)))
            f.write('REC : {}, {}\n'.format(np.mean(results_rec).round(6), np.std(results_rec).round(6)))
            f.write('F1 : {}, {}\n'.format(np.mean(results_f1).round(6), np.std(results_f1).round(6)))



print(out.shape)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(out.detach().cpu().numpy())
print(principalComponents.shape)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(principalComponents[:,0],principalComponents[:,1])
plt.savefig("img.png")


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(out.detach().cpu().numpy())
print(principalComponents.shape)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.scatter(principalComponents[:,0],principalComponents[:,1])
plt.savefig("grid.png")



# export scalar data to JSON for external processing
writer_train.export_scalars_to_json("./all_scalars.json")
writer_train.close()
writer_val.export_scalars_to_json("./all_scalars.json")
writer_val.close()
writer_test.export_scalars_to_json("./all_scalars.json")
writer_test.close()

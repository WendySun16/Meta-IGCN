import argparse
import numpy as np
import random
import os

import torch

from utils import shuffle_data, load_train, load_test
from meta import Meta

parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--subjects', type=int, default=365,
                    help='Number of subjects.')
parser.add_argument('--iteration', type=int, default=1000,
                    help='Number of iterations to train.')
parser.add_argument('--n_way', type=int, default=3, help='n way')
parser.add_argument('--meta_lr', type=float, default=0.003,
                    help='Meta-level outer learning rate.')
parser.add_argument('--update_lr', type=float, default=0.01,
                    help='Task-level inner update learning rate.')
parser.add_argument('--update_step', type=int, default=3,
                    help='Task-level inner update steps.')
parser.add_argument('--update_step_test', type=int, default=3,
                    help='Update steps for finetunning.')
parser.add_argument('--task_num', type=int, default=5, help='Meta batch size, namely task num.')
parser.add_argument('--k_spt', type=int, default=3, help='Size of support set for train.')
parser.add_argument('--test_k_spt', type=int, default=15, help='Size of support set for test.')
parser.add_argument('--k_qry', type=int, default=5, help='Size of query set for train.')
parser.add_argument('--num_qry', type=int, default=1, help='Size of query set for test.')
parser.add_argument('--nfeat', type=int, default=32, help='Number of input units.')
parser.add_argument('--nhid', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device('cuda')


def train_and_test(labels, features, record, model_dir):
    idx = []
    cn_idx = [[] for _ in range(5)]
    mci_idx = [[] for _ in range(5)]
    ad_idx = [[] for _ in range(5)]
    idx = [i for i in range(args.subjects)]
    sp_list = shuffle_data(idx)
    for i in range(5):
        for index in sp_list[i].get('train'):
            if labels[index] == 'CN':
                cn_idx[i].append(index)
            elif labels[index] == 'MCI':
                mci_idx[i].append(index)
            else:
                ad_idx[i].append(index)
    loss, acc = [], []
    best_acc = [0 for _ in range(5)]
    random.seed(42)

    with open(record, 'w') as f:
        for i in range(5):
            print("Cross Validation: {}".format((i + 1)))

            maml = Meta(args).to(device)

            for j in range(args.iteration):
                x, adj, label = [], [], []
                for _ in range(args.task_num):
                    task_x, adjs, task_labels, _, _ = load_train(cn_idx[i], mci_idx[i], ad_idx[i], labels, args.k_spt, args.k_qry, features)
                    x.append(task_x.to(device))
                    adj.append(adjs.to(device))
                    label.append(task_labels.to(device))
                maml.train(True)
                accs, loss_q = maml.forward(x, adj, label)
                loss.append(loss_q)
                acc.append(accs[-1])
                print('Iteration:', j, '\tMeta_Training_Accuracy:', accs, '\tMeta_Training_Loss:', loss_q.item(), file=f)
                print('Iteration:', j, '\tMeta_Training_Accuracy:', accs, '\tMeta_Training_Loss:', loss_q.item())
                if (j + 1) % 1 == 0:
                    model_file = f"{model_dir}/maml.pkl"
                    torch.save(maml.state_dict(), model_file)
                    meta_test_acc = []
                    test_dataset = load_test(sp_list, cn_idx[i], mci_idx[i], ad_idx[i], labels, args.test_k_spt, args.num_qry, features, i)
                    for k in range(len(test_dataset)):
                        model_meta_trained = Meta(args).to(device)
                        model_meta_trained.load_state_dict(torch.load(model_file))
                        model_meta_trained.eval()
                        accs = model_meta_trained.finetunning(test_dataset[k][0].to(device), test_dataset[k][1].to(device), test_dataset[k][2].to(device))
                        meta_test_acc.append(accs)
                    print('Cross Validation:{}, Meta-Test_Accuracy: {}'.format(i+1, np.array(meta_test_acc).mean(axis=0).astype(np.float16)), file=f)
                    print('Cross Validation:{}, Meta-Test_Accuracy: {}'.format(i+1, np.array(meta_test_acc).mean(axis=0).astype(np.float16)))
                    if np.array(meta_test_acc).mean(axis=0).astype(np.float16)[-1] > best_acc[i]:
                        best_acc[i] = np.array(meta_test_acc).mean(axis=0).astype(np.float16)[-1]
                        best_model = f"{model_dir}/best_model_{i+1}.pkl"
                        torch.save(maml.state_dict(), best_model)
                        print('Best model has saved, Cross Validation:{}, Meta-Test_Accuracy: {}'.format(i+1, np.array(meta_test_acc).mean(axis=0).astype(np.float16)), file=f)
            print('Cross Validation:{}, Best Accuracy:{}'.format(i+1, best_acc[i]), file=f)
            print('Cross Validation:{}, Best Accuracy:{}'.format(i+1, best_acc[i]))
        print('the Mean of Best Accuary:', np.mean(best_acc), file=f)
        print('the Mean of Best Accuary:', np.mean(best_acc))
        print('the Min of Best Accuary:', np.min(best_acc), file=f)
        print('the Min of Best Accuary:', np.min(best_acc))
        print('the Max of Best Accuary:', np.max(best_acc), file=f)
        print('the Max of Best Accuary:', np.max(best_acc))
    return loss, acc, best_acc


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'

    model_dir = "/data/ad/train/meta_igcn"
    record = f"{model_dir}/record.txt"
    labels = np.load("/data/ad/labels.npy", allow_pickle=True)
    features = np.load("/data/ad/features_encoded.npy", allow_pickle=True)
    
    loss, acc, best_acc = train_and_test(labels, features, record, model_dir)

    

if  __name__=='__main__':
    main()
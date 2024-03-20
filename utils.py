import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import random
import math
import os


def encode_onehot(labels):
    classes = list(set(labels))
    classes.sort(key = labels.index)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def shuffle_data(idx, seed=42):
    '''
    五折交叉验证划分训练集和测试集
    '''
    np.random.seed(seed)
    index_list = np.random.permutation(len(idx))
    cut1 = int(len(idx) / 5 * 1)
    cut2 = int(len(idx) / 5 * 2)
    cut3 = int(len(idx) / 5 * 3)
    cut4 = int(len(idx) / 5 * 4)
    index1 = index_list[:cut1]
    index2 = index_list[cut1:cut2]
    index3 = index_list[cut2:cut3]
    index4 = index_list[cut3:cut4]
    index5 = index_list[cut4:]
    list1 = np.take(np.array(idx), index1)
    list2 = np.take(np.array(idx), index2)
    list3 = np.take(np.array(idx), index3)
    list4 = np.take(np.array(idx), index4)
    list5 = np.take(np.array(idx), index5)

    sp_list=[]
    te_list1, te_list2, te_list3, te_list4, te_list5 = list1, list2, list3, list4, list5
    tr_list1, tr_list2, tr_list3, tr_list4, tr_list5 = np.concatenate((list2, list3, list4, list5)), np.concatenate((list3, list4, list5, list1)), np.concatenate((list4, list5, list1, list2)), np.concatenate((list5, list1, list2, list3)), np.concatenate((list1, list2, list3, list4))
    
    sp_list.append({'train': tr_list1, 'test': te_list1})
    sp_list.append({'train': tr_list2, 'test': te_list2})
    sp_list.append({'train': tr_list3, 'test': te_list3})
    sp_list.append({'train': tr_list4, 'test': te_list4})
    sp_list.append({'train': tr_list5, 'test': te_list5})
    
    return sp_list


def load_ori_data(csv_path, path):
    '''
    obtain the features, labels, sex, age, apoe of the original dataset from the file
    '''
    data = pd.read_csv(csv_path, encoding="utf-8")
    data = np.array(data)
    img_id, label, sex, age, apoe = [], [], [], [], []
    for d in data:
        img_id.append(d[0])
        label.append(d[2])
        sex.append(d[3])
        age.append(d[4])
        apoe.append(d[7])
    
    feature = []
    files = os.listdir(path)
    for id in img_id:
        for file in files:
            if id == file[:-4]:
                position = f"{path}/{file}"
                x = []
                with open(position) as f:
                    lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split()
                    for j in range(i + 1, len(line)):
                        x.append(float(line[j]))
                feature.append(x)
    return feature, label, sex, age, apoe


def get_adj(idx_spt, idx_qry):
    '''
    obtain the adjacency matrices of the graph by non-image factors
    '''
    sexes = np.load("/data/ad/sexes.npy", allow_pickle=True)
    ages = np.load("/data/ad/ages.npy", allow_pickle=True)
    apoes = np.load("/data/ad/apoes.npy", allow_pickle=True)
    nodes_num = len(idx_spt) + len(idx_qry)
    sexes_spt, sexes_qry = sexes[idx_spt], sexes[idx_qry]
    sex = list(sexes_spt) + list(sexes_qry)
    ages_spt, ages_qry = ages[idx_spt], ages[idx_qry]
    age = list(ages_spt) + list(ages_qry)
    apoes_spt, apoes_qry = apoes[idx_spt], apoes[idx_qry]
    apoe = list(apoes_spt) + list(apoes_qry)

    sexes = []
    for s in sex:
        if s == 'F':
            sexes.append(1)
        else:
            sexes.append(0)

    edges_start, edges_end = [], []
    a = np.zeros((nodes_num, nodes_num), dtype=np.int)
    for i in range(nodes_num):
        for j in range(i + 1, nodes_num):
            if (1 - abs(sexes[i]-sexes[j]) / 1) >= 1:
                a[i][j] += 1
            if (1 - abs(age[i]-age[j]) / (97 - 55)) >= 0.5:
                a[i][j] += 1
            if (1 - abs(apoe[i]-apoe[j]) / 2) >= 1:
                a[i][j] += 1

    for i in range(nodes_num):
        for j in range(i+1, nodes_num):
            if (1 / (3 + 1 - a[i][j])) >= 1:
                edges_start.append(i)
                edges_end.append(j)

    adj = sp.coo_matrix((np.ones(len(edges_start)), (edges_start, edges_end)),
                        shape=(nodes_num, nodes_num), dtype=np.int32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj


def load_train(cn_idx, mci_idx, ad_idx, labels, k_spt, k_qry, features):
    labels = encode_onehot(list(labels))
    spt_cn = random.sample(cn_idx, k_spt)
    spt_mci = random.sample(mci_idx, k_spt)
    spt_ad = random.sample(ad_idx, k_spt)

    spt_idx = spt_cn + spt_mci + spt_ad
    random.shuffle(spt_idx)
    qry_cn = [cn for cn in cn_idx if cn not in spt_cn]
    qry_cn = random.sample(qry_cn, k_qry)
    qry_mci = [mci for mci in mci_idx if mci not in spt_mci]
    qry_mci = random.sample(qry_mci, k_qry)
    qry_ad = [ad for ad in ad_idx if ad not in spt_ad]
    qry_ad = random.sample(qry_ad, k_qry)
    qry_idx = qry_cn + qry_mci + qry_ad
    random.shuffle(qry_idx)

    task_x = torch.FloatTensor(normalize(features[spt_idx + qry_idx]))

    adj = get_adj(spt_idx, qry_idx)
    adj = adj_normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    task_spt_labels = list(labels[spt_idx])
    task_qry_labels = list(labels[qry_idx])
    task_labels = task_spt_labels + task_qry_labels
    task_labels = torch.LongTensor(np.where(task_labels)[1])

    idx_spt = torch.LongTensor(spt_idx)
    idx_qry = torch.LongTensor(qry_idx)
    return task_x, adj, task_labels, idx_spt, idx_qry


def load_test(sp_list, cn_idx, mci_idx, ad_idx, labels, k_spt, num_qry, features, i):
    test_dataset = []
    labels = encode_onehot(list(labels))

    remain_qry = []
    test_task_num = math.ceil(len(sp_list[i].get('test')) / num_qry)
    for _ in range(test_task_num):
        spt_cn = random.sample(cn_idx, k_spt)
        spt_mci = random.sample(mci_idx, k_spt)
        spt_ad = random.sample(ad_idx, k_spt)

        spt_idx = spt_cn + spt_mci + spt_ad
        random.shuffle(spt_idx)
        qry_idx = [qry for qry in sp_list[i].get('test') if qry not in remain_qry]
        if num_qry > (len(sp_list[i].get('test')) - len(remain_qry)):
            qry_idx = random.sample(qry_idx, len(sp_list[i].get('test')) - len(remain_qry))
        else:
            qry_idx = random.sample(qry_idx, num_qry)
        random.shuffle(qry_idx)
        for q in qry_idx:
            remain_qry.append(q)
        test_x = torch.FloatTensor(normalize(features[spt_idx + qry_idx]))

        test_adj = get_adj(spt_idx, qry_idx)
        test_adj = adj_normalize(test_adj + sp.eye(test_adj.shape[0]))
        test_adj = sparse_mx_to_torch_sparse_tensor(test_adj)

        test_spt_labels = list(labels[spt_idx])
        test_qry_labels = list(labels[qry_idx])
        test_labels = test_spt_labels + test_qry_labels
        test_labels = torch.LongTensor(np.where(test_labels)[1])

        test_idx_spt = torch.LongTensor(spt_idx)
        test_idx_qry = torch.LongTensor(qry_idx)
        test_dataset.append([test_x, test_adj, test_labels, test_idx_spt, test_idx_qry])

    return test_dataset


def normalize(x):
    row_max = np.max(x, axis=1)
    row_min = np.min(x, axis=1)
    row_range = row_max - row_min
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j] = (x[i][j] - row_min[i]) / row_range[i]
    return x


def adj_normalize(adj):
    '''
    The adjacency matrix after adding the ring as the input.
    '''
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = np.dot(np.dot(r_mat_inv, adj), r_mat_inv)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''
    Convert a scipy sparse matrix to a torch sparse tensor.
    '''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
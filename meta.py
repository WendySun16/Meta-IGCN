import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch import optim
from models import Meta_GCN
from copy import deepcopy


class Meta(nn.Module):
    def __init__(self, args):
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.test_k_spt = args.test_k_spt
        self.k_qry = args.k_qry
        self.num_qry = args.num_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.nfeat = args.nfeat
        self.nhid = args.nhid
        self.dropout = args.dropout

        self.net = Meta_GCN(self.nfeat, self.nhid, self.n_way, self.dropout)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x, adj, labels):
        task_num = self.task_num
        supportsz = self.n_way * self.k_spt
        querysz = self.n_way * self.k_qry

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            logits = self.net(x[i], adj[i], vars=None)
            loss = F.cross_entropy(logits[:supportsz], labels[i][:supportsz])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            with torch.no_grad():
                para = list(self.net.parameters())
                logits_q = self.net(x[i], adj[i], para)
                loss_q = F.cross_entropy(logits_q[supportsz:], labels[i][supportsz:])
                losses_q[0] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q[supportsz:], labels[i][supportsz:]).sum().item()
                corrects[0] = corrects[0] + correct

            logits_q = self.net(x[i], adj[i], fast_weights)
            loss_q = F.cross_entropy(logits_q[supportsz:], labels[i][supportsz:])
            losses_q[1] += loss_q
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q[supportsz:], labels[i][supportsz:]).sum().item()
            corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                logits = self.net(x[i], adj[i], fast_weights)
                loss = F.cross_entropy(logits[:supportsz], labels[i][:supportsz])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(x[i], adj[i], fast_weights)
                loss_q = F.cross_entropy(logits_q[supportsz:], labels[i][supportsz:])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q[supportsz:], labels[i][supportsz:]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        return accs, loss_q

    def finetunning(self, x, adj, labels):
        supportsz = self.n_way * self.test_k_spt
        querysz = self.num_qry

        corrects = [0 for _ in range(self.update_step_test + 1)]

        net = deepcopy(self.net)

        logits = net(x, adj)
        loss = F.cross_entropy(logits[:supportsz], labels[:supportsz])
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        with torch.no_grad():
            para = list(net.parameters())
            logits_q = net(x, adj, para)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q[supportsz:], labels[supportsz:]).sum().item()
            corrects[0] = corrects[0] + correct

        with torch.no_grad():
            logits_q = net(x, adj, fast_weights)
            loss_q = F.cross_entropy(logits_q[supportsz:], labels[supportsz:])
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q[supportsz:], labels[supportsz:]).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            logits = net(x, adj, fast_weights)
            loss = F.cross_entropy(logits[:supportsz], labels[:supportsz])
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x, adj, fast_weights)
            loss_q = F.cross_entropy(logits_q[supportsz:], labels[supportsz:])

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q[supportsz:], labels[supportsz:]).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / querysz

        return accs
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import NR_GraphAttention
from tabulate import tabulate
import logging
from torch_scatter import scatter_add
import torch_geometric.utils as utils
import pickle
from util import *
from model_util import *

class ALL_entroy(nn.Module):
    def __init__(self, device):
        super(ALL_entroy, self).__init__()
        self.device = device

    def forward_one(self, train_set, x, e2):
        x1_train, x2_train = x[train_set[:, 0]], x[train_set[:, 1]]
        label = torch.arange(0, x1_train.shape[0]).to(self.device)
        d = {}
        for i in range(e2.shape[0]):
            d[int(e2[i])] = i
        x2 = x[e2]
        # print(x1_train.shape[0])
        pred = torch.matmul(x1_train, x2.transpose(0, 1))
        self.bias_0 = torch.nn.Parameter(torch.zeros(x2.shape[0])).to(self.device)
        pred += self.bias_0.expand_as(pred)
        for i in range(x1_train.shape[0]):
            label[i] = d[int(train_set[i, 1])]
        # label = train_set[:, 1].unsqueeze(1)
        label = label.unsqueeze(1)
        # print(label.shape)
        # print(label)
        # exit(0)
        # print( torch.zeros(x1_train.shape[0], x2.shape[0]).shape)
        soft_targets = torch.zeros(x1_train.shape[0], x2.shape[0]). \
            to(self.device).scatter_(1, label, 1)
        # print(soft_targets[2][train_set[2, 1]])
        soft = 0.8
        soft_targets = soft_targets * soft \
                       + (1.0 - soft_targets) \
                       * ((1.0 - soft) / (x2.shape[0] - 1))
        # print(soft_targets[2][train_set[2, 1]])
        logsoftmax = nn.LogSoftmax(dim=1)
        # exit(0)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

    def forward(self, train_set, x, e2):
        loss_l = self.forward_one(train_set, x, e2)
        # loss_r = self.forward_one(train_set[[1, 0]], x)
        return loss_l


class ST_Encoder_Module(nn.Module):
    def __init__(self, node_hidden, rel_hidden,
                 device, node_size, rel_size,
                 dropout_rate=0.0, depth=2):
        super(ST_Encoder_Module, self).__init__()
        self.ent_embedding_1 = None
        self.node_hidden = node_hidden

        self.depth = depth
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.loss = ALL_entroy(self.device)

        self.m_adj = None
        # original entity_emb
        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)

        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)

        self.e_encoder = NR_GraphAttention(
            node_dim=self.node_hidden,
            depth=self.depth,
            use_bias=True,
            device=self.device

        )
        self.r_encoder = NR_GraphAttention(
            node_dim=self.node_hidden,
            depth=self.depth,
            use_bias=True,
            device=self.device
        )

        self.i_encoder = NR_GraphAttention(
            node_dim=self.node_hidden,
            depth=self.depth,
            use_bias=True,
            device=self.device
        )
        embed_size = 384
        self.query_layer = nn.Linear(384, 384).to(self.device)
        self.key_layer = nn.Linear(embed_size, embed_size).to(self.device)
        self.value_layer = nn.Linear(embed_size, embed_size).to(self.device)

        self.scale = torch.sqrt(torch.FloatTensor([embed_size]))

        self.mk = nn.Linear(384,384,bias=False)
        self.mv = nn.Linear(384,384,bias=False)

        self.mk_1 = nn.Linear(384,384,bias=False)
        self.mv_1 = nn.Linear(384,384,bias=False)

        self.emb_num = 3
        self.weight = nn.Parameter(torch.ones((self.emb_num, 1)),
                                   requires_grad=True)

    def avg(self, adj, emb, size: int, node_size):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])

        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def avg_r(self, adj, emb, size: int, node_size):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[node_size, size])

        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)


    def fusion(self, embs):
        embs = [self.weight[idx] * F.normalize(embs[idx]) for idx in range(self.emb_num) if embs[idx] is not None]
        cat_emb = torch.cat(embs, dim=-1)
        return cat_emb

    def forward(self, ent_adj, rel_adj, node_size, rel_size, adj_list,
                r_index, r_val, triple_size, mask=None):
        ent_feature = self.avg(ent_adj, self.ent_embedding.weight, node_size, node_size)
        rel_feature = self.avg_r(rel_adj, self.rel_embedding.weight, rel_size, node_size)

        opt = [self.rel_embedding.weight, adj_list, r_index, r_val, triple_size, rel_size, node_size, mask]
        ent_feat = self.e_encoder([ent_feature] + opt)
        rel_feat = self.r_encoder([rel_feature] + opt)
        out_feature = torch.cat([ent_feat, rel_feat], dim=-1)

        out_feature = self.dropout(out_feature)

        return [out_feature]

    def Attention(self, m1, m2):
        Q = self.query_layer(m1)  # [27793, 384]
        K = self.key_layer(m2)  # [27793, 384]
        V = self.value_layer(m2)  # [27793, 384]
        self.window_size = 1024
        # Initialize output
        output = torch.zeros_like(V)

        # Compute local attention
        for i in range(0, Q.size(0), self.window_size):
            q_chunk = Q[i:i + self.window_size]  # Chunk of queries
            k_chunk = K[i:i + self.window_size]  # Corresponding chunk of keys
            v_chunk = V[i:i + self.window_size]  # Corresponding chunk of values

            attention_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / self.scale.to(self.device)  # Local attention scores
            attention_weights = F.softmax(attention_scores, dim=-1)  # Local attention weights

            output[i:i + self.window_size] = torch.matmul(attention_weights, v_chunk)

        return output

    def external_attention(self, queries, f_type='ent'):
        if f_type == 'ent':
            attn = self.mk(queries)  # bs,n,S
        else:
            attn = self.mk_1(queries)

        attn = F.softmax(attn, dim=-1)  # bs,n,S
        attn = attn / torch.sum(attn, dim=1, keepdim=True)  # bs,n,S

        if f_type == 'ent':
            out = self.mv(attn)  # bs,n,d_model
        else:
            out = self.mv_1(attn)

        return out



class Weight_train_2(nn.Module):
    def __init__(self, N_view, device='cuda:0'):
        super(Weight_train_2, self).__init__()
        self.N_view = N_view
        self.device = device
        self.fc = nn.Linear(self.N_view, 1, bias=False, dtype=torch.float).to(device)

    def forward(self, batch_pair, st_feature, side_modalities_list):
        side_m_list = []
        l, r = batch_pair[:, 0].long(), batch_pair[:, 1].long()
        st_score = torch.mm(st_feature[l], st_feature[r].t())
        side_m_list.append(st_score)

        for i, (_, score) in enumerate(side_modalities_list.items()):
            min_num = score.shape[0]
            r_ = r - min_num
            side_m_list.append(torch.tensor(score[np.ix_(l.tolist(), r_.tolist())]).to(self.device))

        assert len(side_m_list) == self.N_view
        side_m_tensor = torch.stack(side_m_list, dim=1)

        side_m_tensor = F.normalize(side_m_tensor)
        reshaped_input = side_m_tensor.permute(0, 2, 1).reshape(-1, self.N_view)
        reshaped_input = reshaped_input.float()
        weighted_sum = self.fc(reshaped_input)
        fuse_data_ = weighted_sum.view(side_m_tensor.shape[0], side_m_tensor.shape[2])

        gold_data = torch.eye(fuse_data_.size(0)).to(self.device)
        fuse_data_flat = fuse_data_.view(-1)
        gold_data_flat = gold_data.view(-1)
        cos_sim = F.cosine_similarity(fuse_data_flat, gold_data_flat, dim=0, eps=1e-8)
        cos_loss = 1 - cos_sim

        return self.fc.weight, cos_loss


class Loss_Module(nn.Module):
    def __init__(self, node_size, gamma=3):
        super(Loss_Module, self).__init__()
        self.gamma = gamma
        self.node_size = node_size

    def align_loss(self, pairs, emb):

        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        # print(pairs)
        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        del l_emb, r_emb

        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))

        del r_neg_dis, l_neg_dis

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()

        lamb, tau = 30, 10
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        return torch.mean(l_loss + r_loss)

    def align_loss_weight(self,  pairs, emb):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        # pairs 现在是一个 n*3 的矩阵，其中第三列是置信度
        l, r, confidence = pairs[:, 0].long(), pairs[:, 1].long(), pairs[:, 2]
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        # 使用置信度作为权重
        confidence = confidence.float()  # 确保置信度是浮点数
        pos_dis_weighted = pos_dis * confidence.unsqueeze(1)
        # pos_dis_weighted = pos_dis

        l_neg_dis_weighted = l_neg_dis
        r_neg_dis_weighted = r_neg_dis

        del l_emb, r_emb
        del r_neg_dis, l_neg_dis

        # 计算损失函数时考虑置信度
        l_loss = pos_dis_weighted - l_neg_dis_weighted + self.gamma
        r_loss = pos_dis_weighted - r_neg_dis_weighted + self.gamma

        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))


        del l_neg_dis_weighted, r_neg_dis_weighted

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()

        lamb, tau = 30, 10
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)

        total_loss = (l_loss + r_loss) / 2
        total_loss = total_loss * confidence

        final_loss = torch.mean(total_loss)
        del l_loss, r_loss, total_loss

        return final_loss

    def forward(self, train_pairs: torch.Tensor, feature, weight=False):
        if weight:
            loss = self.align_loss_weight(train_pairs, feature)
        else:
            loss = self.align_loss(train_pairs, feature)

        return loss


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class Loss_model2(nn.Module):
    def __init__(self, device, tau=0.05, ab_weight=0.5, n_view=2):
        super(Loss_model2, self).__init__()
        self.tau = tau
        self.device = device
        self.sim = cosine_sim
        self.weight = ab_weight  # the factor of a->b and b<-a
        self.n_view = n_view

    def softXEnt(self, target, logits):

        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, pairs, emb, weight=True):
        norm = True
        if norm:
            emb = F.normalize(emb, dim=1)
        num_ent = emb.shape[0]

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        zis, zjs = emb[l], emb[r]
        confidence = pairs[:,2]

        temperature = self.tau
        alpha = self.weight
        n_view = self.n_view

        LARGE_NUM = 1e9

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        num_classes = batch_size * n_view
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
        labels = labels.to(self.device)

        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.to(self.device).float()

        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

        logits_ab *= confidence.unsqueeze(1)
        logits_ba *= confidence.unsqueeze(1)

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        loss_a = self.softXEnt(labels, logits_a)
        loss_b = self.softXEnt(labels, logits_b)

        return alpha * loss_a + (1 - alpha) * loss_b


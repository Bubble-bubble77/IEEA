import time
import torch.nn.functional as F
from dataset import MMEaDtaset
import argparse
from model import ST_Encoder_Module, Loss_Module, Weight_train_2
import numpy as np
import torch
import gc
import random
import os
from util import *
from tqdm import trange
from datetime import datetime
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import pickle
import math
from matplotlib import pyplot as plt
import pytz

timezone = pytz.timezone('Asia/Shanghai')
current_date = datetime.now(timezone).strftime("%m%d_%H%M")
print(current_date)
max_hit0 = 0
st_max_hit0 = 0


def seed_torch(seed=1029):
    print('set seed')
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True  # 选择确定性算法



class RUN():
    def __init__(self):
        super(RUN, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.args = self.parse_options(self.parser)
        self.semi_pred = True
        self.csp = self.args.csp
        self.encoder_type = self.args.encoder
        if self.args.exp_m ==0:
            self.exp_mod = False
        else:
            self.exp_mod = True

        self.Exwithout = None

        self.L = 0
        self.Imp = 'ST'
        out_str = ''
        if self.Imp:
            out_str = out_str + str(self.Imp)

        if self.exp_mod:
            if self.Exwithout == 'A':
                out_str = out_str + '+V+N'
            elif self.Exwithout == 'V':
                out_str = out_str + '+A+N'
            elif self.Exwithout == 'N':
                out_str = out_str + '+V+A'
            else:
                out_str = out_str + '+A+V+N'

        if self.semi_pred:
            out_str = out_str + '+csp{}'.format(self.csp)

        print('L={}'.format(self.L))

        print("L:{}, semi_pred={}, "
              "side_mode={}, self.Imp ={}".format(self.L, self.semi_pred, self.exp_mod, self.Imp))

        self.semantic_encoder = None
        self.loss_model = None
        self.structure_encoder = None
        self.weight_model = None
        self.pre_pair_ = False

        self.confidence = True
        self.thred = 0.95
        self.thred_weight = 0.95

        self.sig = self.args.sig
        self.img_feature = None

        self.remove_rest_set_1 = set()
        self.remove_rest_set_2 = set()
        self.rest_pair = None
        self.G_dataset = None
        self.getembedding = None
        self.triple_size = None
        self.r_val = None
        self.r_index = None
        self.adj_list = None
        self.rel_size = None
        self.node_size = None
        self.rel_adj = None
        self.ent_adj = None
        self.dev_pair = None
        self.train_pair = None
        self.train_pair_confidence = None
        self.st_result = 0


        self.device = 'cuda:0'
        self.train_epoch = 70
        self.batchsize = 512  # db
        print("batchsize is {}".format(self.batchsize))
        self.lr = 0.005
        self.droprate = 0.3

        self.side_weight = 1
        self.side_weight_rate = 1

        self.trainset_shuffle = True
        self.st_score = []
        self.all_score = []
        self.epoch_list = []


    @staticmethod
    def parse_options(parser):
        parser.add_argument('dataset', type=str, help='which dataset, FB_DB or FB_YAGO', default='FB_DB')
        parser.add_argument('ratio', type=float, help='which ratio, 0.2', default='0.2')
        parser.add_argument('encoder', type=str, help='', default='Qwen7b, llmembed')
        parser.add_argument('sig', type=float, help='which sig, 0.2', default='0.2')
        parser.add_argument('csp', type=float, help='which csp', default='0,1,2')
        parser.add_argument('exp_m', type=float, help='which csp', default='0,1')
        parser.add_argument('epoch_weight', type=float, help='which weight', default='2, 5, 8')
        parser.add_argument('exp_N', type=float, help='w/o N', default='0,1')

        return parser.parse_args()

    def load_dataset(self):
        print("1. load dataset....")
        dataset = self.args.dataset
        with open("../results/run0127_{}_rate{}_{}_sig{}.txt".format(self.args.dataset, self.args.ratio, current_date, str(self.sig)), 'a') as f:
            f.write(dataset + '_' + current_date + ':\n')
            f.write('sig: ' + str(self.sig) + '\n')
            f.write('droprate: ' + str(self.droprate) + '\n')

        if dataset == 'FB_DB':
            DATASET_NAME = 'FB15K-DB15K'
        elif dataset == 'FB_YAGO':
            DATASET_NAME = 'FB15K-YAGO15K'
        else:
            DATASET_NAME = 'FB15K-DB15K'  # default

        self.G_dataset = MMEaDtaset('../data/MMEA-data/seed{}/{}'.format(str(self.args.ratio), DATASET_NAME),
                                    device=self.device, ratio=self.args.ratio)

        self.train_pair, self.dev_pair = self.G_dataset.train_pair, self.G_dataset.test_pair
        self.train_pair_set = set()
        self.train_pair_set.update((i,j,1) for (i,j) in self.train_pair)
        self.pair_gold = np.concatenate((self.train_pair, self.dev_pair))
        self.train_pair = torch.tensor(self.train_pair).to(self.device)
        self.dev_pair = torch.tensor(self.dev_pair)
        self.kg1_num = self.G_dataset.kg1_num

        confidence_column = torch.ones(self.train_pair.shape[0], 1)
        self.train_pair_confidence = torch.cat((self.train_pair, confidence_column.to(self.device)), dim=1)
        right_pair_num, pair_accuracy = self.pair_accuacy(self.train_pair_confidence.cpu().numpy()[:, :2], self.pair_gold)
        with open(
                "../accuracy/run0127_{}_rate{}_{}_sig{}_accuracy_csp{}.txt".format(self.args.dataset, self.args.ratio, current_date,
                                                           str(self.sig), str(self.csp)), 'a') as f:
            f.write(str(right_pair_num) + ' ' + str(pair_accuracy) + '\n')


        self.rest_set_1 = self.G_dataset.rest_set_1
        self.rest_set_2 = self.G_dataset.rest_set_2

        self.rows_to_keep = self.G_dataset.rest_set_1.copy()
        self.cols_to_keep = self.G_dataset.rest_set_2.copy()
        self.r_len = len(self.rows_to_keep)
        self.c_len = len(self.cols_to_keep)

        print("train set: " + str(len(self.train_pair)))
        print("dev set: " + str(len(self.dev_pair)))
        self.true_num = len(self.train_pair)

        self.ent_adj, self.rel_adj, self.node_size, self.rel_size, \
        self.adj_list, self.r_index, self.r_val, self.triple_size, self.adj = self.G_dataset.reconstruct_search(None,
                                                                                                                None,
                                                                                                                self.G_dataset.kg1,
                                                                                                                self.G_dataset.kg2,
                                                                                                                new=True)

        print("1. load dataset over....")

        self.side_modalities = {}
        all_es_path = '../data/ES_ALL'
        es_path_list = []
        self.modal_num = 1
        for filename in os.listdir(all_es_path):
            if dataset == 'FB_DB':
                if 'DB' in filename:
                    es_path_list.append(os.path.join(all_es_path, filename))
            else:
                if 'YAGO' in filename:
                    es_path_list.append(os.path.join(all_es_path, filename))
        new_es_path_list = []
        for es_path in es_path_list:
            if self.args.exp_N ==1:
                if ('name' in es_path and self.encoder_type in es_path) or ('attr' in es_path and 'Qwen7b_attr.npy' in es_path):  # 有name的时候
                    new_es_path_list.append(es_path)
            else:
                if ('attr' in es_path and 'Qwen7b_attr.npy' in es_path):  # 无name的时候
                    new_es_path_list.append(es_path)
        es_path_list = new_es_path_list

        all_moda = []
        for es_path in es_path_list:
            print(es_path)
            moda_np = np.load(es_path)
            moda_np = (moda_np - moda_np.min()) / (moda_np.max() - moda_np.min())
            print(moda_np.shape)
            all_moda.append(moda_np)
            explicit_score_test(moda_np, self.dev_pair, self.device)
            self.s_num, self.t_num = moda_np.shape
            self.side_modalities[es_path.split('/')[-1].split('.')[0]] = moda_np
            self.modal_num += 1

    def pair_accuacy(self, train_pair, pair_gold):
        pair_gold = {tuple(row) for row in pair_gold}
        train_pair_ = {tuple(row) for row in train_pair}
        return len(train_pair_.intersection(pair_gold)), len(train_pair_.intersection(pair_gold)) / len(train_pair_)

    def init_model(self):
        self.depth = 2
        self.structure_encoder = ST_Encoder_Module(
            node_hidden=128,
            rel_hidden=128,
            node_size=self.node_size,
            rel_size=self.rel_size,
            device=self.device,
            dropout_rate=self.droprate,
            depth=self.depth).to(self.device)

        self.loss_model = Loss_Module(node_size=self.node_size, gamma=2).to(self.device)

        self.optimizer = torch.optim.RMSprop(self.structure_encoder.parameters(), lr=self.lr)  # 3-9
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.999)
        self.weight_model = Weight_train_2(N_view=self.modal_num, device=self.device)
        self.optimizer_2 = torch.optim.RMSprop(self.weight_model.parameters(), lr=self.lr)
        self.scheduler_2 = ExponentialLR(self.optimizer_2, gamma=0.999)

        total_params = sum(p.numel() for p in self.structure_encoder.parameters() if p.requires_grad)
        print(total_params)


    def run(self):
        self.load_dataset()
        self.init_model()

        train_hit = []
        trainset = []

        train_epoch = self.train_epoch
        batch_size = self.batchsize

        print("2. start training....\n")
        for epoch in range(train_epoch):
            time3 = time.time()
            print("now is epoch " + str(epoch))
            self.structure_encoder.train()
            self.weight_model.train()

            if self.trainset_shuffle:
                num_rows = self.train_pair_confidence.size(0)
                random_indices = torch.randperm(num_rows)
                self.train_pair_confidence = self.train_pair_confidence[random_indices]
            for i in range(0, len(self.train_pair_confidence), batch_size):
                batch_pair = self.train_pair_confidence[i:i + batch_size]
                if len(batch_pair) == 0:
                    continue
                feature_list = self.structure_encoder(
                    self.ent_adj, self.rel_adj, self.node_size,
                    self.rel_size, self.adj_list, self.r_index, self.r_val,
                    self.triple_size, mask=None)

                loss = self.loss_model(batch_pair, feature_list[0], weight=True)

                if epoch > self.args.epoch_weight:
                    self.weight_t, cos_loss = self.weight_model(batch_pair.clone(), feature_list[0].clone(), self.side_modalities)
                    cos_loss.backward(retain_graph=True)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            self.optimizer_2.step()
            self.optimizer_2.zero_grad()
            self.scheduler_2.step()
            try:
                print(self.weight_t)
            except:
                pass

            # test code
            if epoch % 2 == 0 and epoch > 0:
                time5 = time.time()
                gid1, gid2 = self.dev_pair.T
                print(len(gid1))

                self.structure_encoder.eval()
                self.weight_model.eval()
                with torch.no_grad():
                    feature_list = self.structure_encoder(
                        self.ent_adj.to(self.device), self.rel_adj.to(self.device),
                        self.node_size, self.rel_size,
                        self.adj_list.to(self.device),
                        self.r_index.to(self.device), self.r_val.to(self.device),
                        self.triple_size,
                        mask=None)

                    out_feature = feature_list[0].cpu()
                    del feature_list
                    torch.cuda.empty_cache()

                    out_feature = out_feature / (torch.linalg.norm(out_feature, dim=-1, keepdim=True) + 1e-5)
                    index_a = torch.LongTensor(gid1)
                    index_b = torch.LongTensor(gid2)

                    Lvec = out_feature[index_a]
                    Rvec = out_feature[index_b]
                    Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
                    Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)

                    time_ss = time.time()
                    St_result = self.sinkhorn_test_ST(Lvec, Rvec, device=self.device,
                                                            len_dev=len(self.dev_pair))

                    st_hit0 = float(St_result['hits@1'])
                    result = self.sinkhorn_test(Lvec, Rvec, device=self.device,
                                                            len_dev=len(self.dev_pair), gid1=gid1, gid2=gid2)
                    if self.st_result > float(result['hits@1']) and epoch > 20:
                        sim_mat = self.sim_results(out_feature[:self.kg1_num], out_feature[self.kg1_num:])
                        print(sim_mat.shape)
                        sim_mat_list = []
                        sim_mat_list.append(sim_mat.detach().numpy())
                        if self.exp_mod:
                            for _, side_score in self.side_modalities.items():
                                sim_mat_list.append(side_score)
                        try:
                            weight_norm = F.softmax(self.weight_t, dim=0)
                            weight_norm_ = weight_norm.detach().cpu().numpy()
                            fuse_data = [weight_norm_[idx] * sim_mat_list[idx] for idx in range(len(sim_mat_list))]
                            sim_mat1 = sum(fuse_data) / len(sim_mat_list)
                        except:
                            sim_mat1 = sum(sim_mat_list) / len(sim_mat_list)

                        sim_mat1 = torch.Tensor(sim_mat1).to(self.device)
                        sim_mat_r = 1 - sim_mat1

                        # matrix_sinkhorn
                        if sim_mat_r.dim == 3:
                            M = sim_mat_r
                        else:
                            M = sim_mat_r.view(1, sim_mat_r.size(0), -1)

                        M = M.to(self.device)
                        m, n = sim_mat_r.shape
                        a = torch.ones([1, m], requires_grad=False, device=self.device)
                        b = torch.ones([1, n], requires_grad=False, device=self.device)
                        P = sinkhorn(a, b, M, 0.02, max_iters=100, stop_thresh=1e-3)
                        sim_mat_ =  np.squeeze(P.detach().cpu().numpy())
                        print(sim_mat_.shape)
                        if self.args.exp_N:
                            np.save(f'../data/Sinkhorn_MMKG/Fuse_{self.args.dataset}_ratio{self.args.ratio}_total_ST_N.npy', sim_mat_)
                        else:
                            np.save(f'../data/Sinkhorn_MMKG/Fuse_{self.args.dataset}_ratio{self.args.ratio}_total_ST.npy', sim_mat_)
                        exit(0)
                    else:
                        self.st_result = float(result['hits@1'])

                    time_se = time.time()
                    print("sinkhorn_test time is {}".format(time_se-time_ss))
                    hits0 = float(result['hits@1'])
                    self.st_score.append(st_hit0)
                    self.all_score.append(hits0)
                    self.epoch_list.append(epoch)
                    global st_max_hit0
                    global max_hit0
                    if hits0 > max_hit0:
                        max_hit0 = hits0
                        with open(
                                "../results/run0127_{}_rate{}_{}_sig{}.txt".format(self.args.dataset, self.args.ratio,
                                                                           current_date,
                                                                           str(self.sig)), 'a') as f:
                            f.write(str(epoch) + ':\n')
                            f.write(str(result) + '\n')
                            f.write('max_hit0={}\n'.format(max_hit0))

                    print("max_hit0={}".format(max_hit0))
                    train_hit.append(round(hits0, 4))
            if epoch >= 9 and epoch % 2 == 0 and self.semi_pred:
                self.structure_encoder.eval()
                self.weight_model.eval()
                with torch.no_grad():

                    gid1 = torch.tensor(np.array(self.rest_set_1))
                    gid2 = torch.tensor(np.array(self.rest_set_2))
                    # print('rest set shape is : {} {}'.format(len(gid1), len(gid2)))

                    Lvec = out_feature[gid1]
                    Rvec = out_feature[gid2]
                    Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
                    Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)

                    scores = self.sim_results(Lvec, Rvec)

                    rows_to_keep = [i for i in range(self.r_len) if i not in self.remove_rest_set_1]
                    cols_to_keep = [j for j in range(self.c_len) if j not in self.remove_rest_set_2]
                    assert scores.shape[0] == len(rows_to_keep) and len(rows_to_keep) == len(cols_to_keep)

                    new_pair = set()
                    new_pair.update(self.pred_pair_confidence(scores))

                    # 累加score生成pair
                    if self.exp_mod:
                        for i, (_, side_score) in enumerate(self.side_modalities.items()):
                            new_pair.update(self.pred_pair_confidence(torch.Tensor(side_score[np.ix_(gid1.tolist(), (gid2-side_score.shape[0]).tolist())])
                                                                      * self.side_weight))


                    # 方案1：删除conflict的pair
                    if self.csp == 1:
                        new_pair = self.delete_repeat_pair(new_pair)
                    elif self.csp == 2:
                        new_pair = self.choose_repeat_pair(new_pair)
                    else:
                        pass

                    # 对训练集进行csp检查
                    self.train_pair_set.update(new_pair)
                    if self.csp == 1:
                        self.train_pair_set = self.delete_repeat_pair(self.train_pair_set)
                    elif self.csp == 2:
                        self.train_pair_set = self.choose_repeat_pair(self.train_pair_set)
                    else:
                        pass

                    self.train_pair_confidence = torch.tensor(list(self.train_pair_set)).to(self.device)
                    trainset.append(round(len(new_pair), 4))

                    right_pair_num, pair_accuracy = self.pair_accuacy(self.train_pair_confidence.cpu().numpy()[:, :2],
                                                                      self.pair_gold)
                    with open(
                            "../accuracy/run0127_{}_rate{}_{}_sig{}_accuracy_csp{}.txt".format(self.args.dataset,
                                                                                      self.args.ratio, current_date,
                                                                                      str(self.sig), str(self.csp)),
                            'a') as f:
                        f.write(str(self.train_pair_confidence.shape[0]) + ' ' + str(pair_accuracy) + '\n')

                    count = 0
                    for (e1, e2, conf) in new_pair:

                        if e1 in self.rest_set_1 and e2 in self.rest_set_2:
                            try:
                                if e1 in self.rows_to_keep and e2 in self.cols_to_keep:
                                    index_1 = self.rows_to_keep.index(e1)
                                    index_2 = self.cols_to_keep.index(e2)
                            except ValueError:
                                print(f"元素 {e1} 或 {e2} 不在rows_to_keep, cols_to_keep集合中。")

                            if index_1 in self.remove_rest_set_1 or index_2 in self.remove_rest_set_2:
                                continue
                            else:
                                self.remove_rest_set_1.add(index_1)
                                self.remove_rest_set_2.add(index_2)
                                count = count + 1

                            try:
                                if e1 in self.rest_set_1 and e2 in self.rest_set_2:
                                    self.rest_set_1.remove(e1)
                                    self.rest_set_2.remove(e2)

                            except ValueError:
                                print(f"元素 {e1} 或 {e2} 不在rest_set_1, rest_set_2。")

                    print("number of new_pair is {}, real remove number is {}".format(len(new_pair), count))
                    self.thred = self.thred * self.thred_weight

    def pred_pair_confidence(self, score):

        new_set = set()
        A = score.argmax(axis=0)
        B = score.argmax(axis=1)
        for i, j in enumerate(A):
            if B[j] == i and (score[j][i] > self.sig or score[i][j] > self.sig):
                if score[j][i] < self.sig or score[i][j] < self.sig:
                    sc = (score[j][i] + score[i][j]) / 2
                    new_conf = math.exp(-0.5 * (sc - self.sig) ** 2)
                else:
                    new_conf = 1
                if self.confidence:
                    new_set.add((self.rest_set_1[j], self.rest_set_2[i], new_conf))
                else:
                    new_set.add((self.rest_set_1[j], self.rest_set_2[i], 1))

        return new_set

    def delete_repeat_pair(self, pair):
        a_to_bs = {}
        b_to_as = {}
        for a, b, conf in pair:
            if a in a_to_bs:
                a_to_bs[a].append(b)
            else:
                a_to_bs[a] = [b]
            if b in b_to_as:
                b_to_as[b].append(a)
            else:
                b_to_as[b] = [a]

        conflicting_tuples = set()
        for a, bs in a_to_bs.items():
            if len(bs) > 1:
                # 存在相同的a对应不同的b
                for b in bs:
                    conflicting_tuples.add((a, b))
        for b, a_s in b_to_as.items():
            if len(a_s) > 1:
                # 存在相同的a对应不同的b
                for a in a_s:
                    conflicting_tuples.add((a, b))
        print("conflicting_tuples: {}".format(len(conflicting_tuples)))

        conflicting_tuples_v = set()
        for (x, y) in conflicting_tuples:
            for triple in pair:
                if triple[0] == x and triple[1] == y:
                    # print(triple)
                    conflicting_tuples_v.add((triple[0], triple[1], triple[2]))
                    # print(conflicting_tuples_v)

        new_pair = pair.difference(conflicting_tuples_v)

        return new_pair

    def choose_repeat_pair(self, pair):
        max_triplets = {}
        for triplet in pair:
            x, y, z = triplet
            # 更新字典以 x 为键
            if x not in max_triplets or max_triplets[x][2] < z:
                max_triplets[x] = triplet
            # 更新字典以 y 为键
            if y not in max_triplets or max_triplets[y][2] < z:
                max_triplets[y] = triplet

        new_pair = set()
        for triplet in max_triplets.values():
            x, y, z = triplet
            # 确保三元组是当前最大值
            if (x not in max_triplets or max_triplets[x] == triplet) and (
                    y not in max_triplets or max_triplets[y] == triplet):
                new_pair.add(triplet)

        return new_pair

    def sim_results(self, Matrix_A, Matrix_B):
        # A x B.t
        A_sim = torch.mm(Matrix_A, Matrix_B.t())
        return A_sim


    def sinkhorn_test_ST(self, sourceVec, targetVec, device, len_dev):
        sim_mat = self.sim_results(sourceVec, targetVec)
        if self.L == 0:
            sim_mat = sim_mat.T

        if self.L == 1:
            gid1, gid2 = self.dev_pair.T
            new_adj_1 = self.adj[gid1, :][:, gid1]
            new_adj_2 = self.adj[gid2, :][:, gid2]
            new_adj_1 = new_adj_1 / (np.linalg.norm(new_adj_1, axis=-1, keepdims=True) + 1e-5)
            new_adj_2 = new_adj_2 / (np.linalg.norm(new_adj_2, axis=-1, keepdims=True) + 1e-5)
            new_adj_1 = torch.FloatTensor(new_adj_1)
            new_adj_2 = torch.FloatTensor(new_adj_2)
            sim_mat = sim_mat.T + new_adj_2 * sim_mat.T * new_adj_1.T

        sim_mat_r = 1 - sim_mat

        # matrix_sinkhorn
        if sim_mat_r.dim == 3:
            M = sim_mat_r
        else:
            M = sim_mat_r.view(1, sim_mat_r.size(0), -1)
        M = M.to(device)
        m, n = sim_mat_r.shape
        a = torch.ones([1, m], requires_grad=False, device=device)
        b = torch.ones([1, n], requires_grad=False, device=device)
        P = sinkhorn(a, b, M, 0.02, max_iters=100, stop_thresh=1e-3)
        P = view2(P)
        del M, a, b
        torch.cuda.empty_cache()

        # evaluate_sim
        result = evaluate_sim_matrix(link=torch.stack([torch.arange(len_dev),
                                                       torch.arange(len_dev)], dim=0),
                                     sim_x2y=P,
                                     no_csls=True)
        return result

    def sinkhorn_test(self, sourceVec, targetVec, device, len_dev, gid1, gid2):
        st_sim_mat = self.sim_results(sourceVec, targetVec)
        sim_mat_list = []
        sim_mat_list.append(st_sim_mat.detach().numpy())
        if self.exp_mod:
            for _, side_score in self.side_modalities.items():

                sim_mat_list.append(side_score[np.ix_(gid1.tolist(), (gid2-side_score.shape[0]).tolist())])

        try:
            weight_norm = F.softmax(self.weight_t, dim=0)
            # weight_norm = self.weight_t
            weight_norm_ = weight_norm.detach().cpu().numpy()
            fuse_data = [weight_norm_[idx] * sim_mat_list[idx] for idx in range(len(sim_mat_list))]
            sim_mat = sum(fuse_data) / len(sim_mat_list)

        except:
            sim_mat = sum(sim_mat_list) / len(sim_mat_list)
        if self.L == 0:
            sim_mat = sim_mat.T

        if self.L == 1:
            gid1, gid2 = self.dev_pair.T
            new_adj_1 = self.adj[gid1, :][:, gid1]
            new_adj_2 = self.adj[gid2, :][:, gid2]
            new_adj_1 = new_adj_1 / (np.linalg.norm(new_adj_1, axis=-1, keepdims=True) + 1e-5)
            new_adj_2 = new_adj_2 / (np.linalg.norm(new_adj_2, axis=-1, keepdims=True) + 1e-5)
            sim_mat = sim_mat.T + new_adj_2 * sim_mat.T * new_adj_1.T

        sim_mat_r = 1 - sim_mat

        # matrix_sinkhorn
        if sim_mat_r.ndim == 3:
            M = sim_mat_r
        else:
            try:
                M = sim_mat_r.view(1, sim_mat_r.size(0), -1)
            except:
                M = sim_mat_r.reshape(1, sim_mat_r.shape[0], -1)
                M = torch.tensor(M, dtype=torch.float32)
        M = M.to(device)
        m, n = sim_mat_r.shape
        a = torch.ones([1, m], requires_grad=False, device=device)
        b = torch.ones([1, n], requires_grad=False, device=device)
        P = sinkhorn(a, b, M, 0.02, max_iters=100, stop_thresh=1e-3)
        P = view2(P)
        del M, a, b
        torch.cuda.empty_cache()

        # evaluate_sim
        result = evaluate_sim_matrix(link=torch.stack([torch.arange(len_dev),
                                                       torch.arange(len_dev)], dim=0),
                                     sim_x2y=P,
                                     no_csls=True)
        return result

    def get_topk_sim(sim: Tensor, k_ent=10) -> Tuple[Tensor, Tensor, Tensor]:
        return torch.topk(sim, k=k_ent) + (sim,)

if __name__ == "__main__":
    seed_torch()
    model = RUN()
    model.run()



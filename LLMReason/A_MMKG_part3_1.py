
from MMKG_retrieval import *
from prompt import *
from openai import OpenAI
import os
import pytz
from datetime import datetime
from Prepare_EMB.MMKG.util import *
from Prepare_EMB.MMKG.dataset import MMEaDtaset
from Prepare_EMB.MMKG.MMKGc.model import ST_Encoder_Module, Loss_Module
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange
import time
import numpy as np



timezone = pytz.timezone('Asia/Shanghai')
current_date = datetime.now(timezone).strftime("%m%d_%H%M")


class PART_3():
    def __init__(self, dataset, side=None, ratio= None, equal_rule= None, align_rule=None):
        print(f"{dataset}_{ratio}_{side}")
        self.dataset = dataset
        self.side = side
        self.ratio = ratio
        self.re = Retrivel(dataset)
        self.en1_list, self.en2_list = self.ent1_ent2_load()
        self.device = 'cuda:0'
        self.seed_dict = self.seed_load()
        # self.es_score = self.sim_score_load()
        self.structure_learn()

        self.Id_Attr = self.re.id_Attr
        self.Id_Name = self.re.id_Name
        self.Name_id = self.Name_Id()

        self.Id_rel_dic = self.Rel_list_to_dict(self.re.id_Rel)
        self.trip_1_dic, self.trip_2_dic = self.Triple_load()
        self.trip_dic = self.trip_1_dic | self.trip_2_dic


        self.LLM_load()
        # self.LLM_type='qwen3-max'
        self.LLM_type = 'deepseek-v3.2-exp-thinking'
        # self.LLM_type = 'qwen-flash'
        # self.LLM_type = 'deepseek-v3.2-exp'
        # self.LLM_type = 'deepseek-v3'
        # self.LLM_type = 'gpt-5'
        # self.LLM_type = 'claude-haiku-4-5-20251001'
        # self.LLM_type = 'gemini-2.5-flash'
        # self.LLM_type = 'qwen3-235b-a22b-instruct-2507'

        self.equal = equal_rule
        self.align = align_rule

        if self.equal is not None:
            self.euql_ent_dict = self.equal_ent_load()

        if self.align is not None:
            self.align_ent_dict = self.complete_aligned_neighbor()
 
    def seed_load(self):
        filepath = f'../data/MMKG/seed'
        filename = os.path.join(filepath, self.dataset, 'ref_ent_ids')
        seed_pair = {}
        with open(filename, 'r') as f:
            for line in f:
                try:
                    e1, e2 = line.split()
                    # e1和e2之间可以互查
                    seed_pair[int(e1)] = int(e2)
                    seed_pair[int(e2)] = int(e1)
                except:
                    continue
        return seed_pair


    def Name_Id(self):
        Name_id = {}
        path = '../data/MMKG/MMEA_name'
        if 'DB' in self.dataset:
            f1_name = 'DB_name.txt'
            f2_name = 'FB_DB_name.txt'
        else:
            f1_name = 'YAGO_name.txt'
            f2_name = 'FB_YAGO_name.txt'
        # id_name = {}
        with open(os.path.join(path, f1_name), 'r') as f:
            for line in f:
                id_, name = line.strip().split(' ')
                # 处理重名情况
                name_ = name[1:-1]
                if name_ in Name_id.keys():
                    Name_id[name_].append(int(id_))
                else:
                    Name_id[name_] = [int(id_)]
                # print(name_)
                # exit(0)
        with open(os.path.join(path, f2_name), 'r') as f:
            for line in f:
                id_, name = line.strip().split(' ')
                # id_name[int(id_)] = name
                name_ = name[:-1]
                if name_ in Name_id.keys():
                    Name_id[name_].append(int(id_))
                else:
                    Name_id[name_] = [int(id_)]
                # print(name_)
                # exit(0)
        return Name_id
        

    def Rel_list_to_dict(self, Rel_list):
        Id_rel_dic = {} #name:rel_Name
        for ent in Rel_list.keys():
            rel_list = Rel_list[ent]
            rel_dic = {}
            for rel in rel_list:
                try:
                    rel_name, ent_name = rel.split('--')
                    if ent_name != 'Null':
                        if ent_name[-1] == '_':
                            ent_name = ent_name[:-1]
                        rel_dic[ent_name] = rel_name
                except:
                    continue
            Id_rel_dic[ent] = rel_dic
        return Id_rel_dic
    

    def Triple_load(self):
        triple1_path = f"../data/MMKG/seed{self.ratio}/{self.dataset.replace('_','-')}/triples_1_"
        triple2_path = f"../data/MMKG/seed{self.ratio}/{self.dataset.replace('_','-')}/triples_2_"

        def trip_load(path):
            trip_dic = {}
            with open(path,'r') as f:
                for line in f:
                    ent1, _, ent2 = line.strip().split('\t')
                    if ent1 not in trip_dic.keys():
                        trip_dic[int(ent1)] = [int(ent2)]
                    else:
                        trip_dic[int(ent1)].append(int(ent2))
            return trip_dic
        
        trip_1_dic = trip_load(triple1_path)
        trip_2_dic = trip_load(triple2_path)

        # print(max(trip_1_dic.keys()))
        return trip_1_dic, trip_2_dic


    def ent1_ent2_load(self):
        Imcomplete_data_save_path = f'../data/MMKG/IC_data/seed{self.ratio}/'
        Ic_e1_path = os.path.join(Imcomplete_data_save_path, f'{self.dataset}_{self.side}_Ic_e1.txt')
        # Ic_e2_path = os.path.join(Imcomplete_data_save_path, f'{self.dataset}_{self.side}_Ic_e2.txt')
        Ic_e2_path = os.path.join(Imcomplete_data_save_path, f'{self.dataset}_{self.side}_top_Ic_e2.txt')

        ent1_list = []
        with open(Ic_e1_path, 'r') as f:
            for line in f:
                ent1_list.append(int(line.strip()))
        print(f"ent1 读取完成，共 {len(ent1_list)} 条记录")

        ent2_list = []
        with open(Ic_e2_path, 'r') as f:
            for line in f:
                ent2_list.append(int(line.strip()))
        print(f"ent2 读取完成，共 {len(ent2_list)} 条记录")

        return ent1_list, ent2_list

    def sim_score_load(self):
        folder_path = '../data/MMKG/Sinkhorn_MMKG'

        if self.side is not None:
            file_name = os.path.join(folder_path,
                                     f'Fuse_{self.dataset}_ratio{self.ratio}_total_ST_{self.side}.npy')
        else:
            file_name = os.path.join(folder_path, f'Fuse_{self.dataset}_ratio{self.ratio}_total_ST.npy')
        moda_matrix = np.load(file_name)
        m, _ = moda_matrix.shape
        es_score = (moda_matrix - moda_matrix.min()) / (moda_matrix.max() - moda_matrix.min())
        return es_score

    def topk_e1(self, k=10):
        ent2_list = [item - self.es_score.shape[0] for item in self.en2_list]
        new_esscore_matrix = self.es_score[np.ix_(self.en1_list, ent2_list)]
        new_esscore_matrix = (new_esscore_matrix - new_esscore_matrix.min()) / (new_esscore_matrix.max() - new_esscore_matrix.min())
        x_topk_indices = np.argsort(-new_esscore_matrix, axis=1)[:, :k]

        st_score_path = r'../data/MMKG/ST_feature/FB_DB_ratio0.2_ST_N2.npy'
        moda_matrix = np.load(st_score_path)
        # print(moda_matrix.shape)
        
        return x_topk_indices

    def structure_learn(self):
        print('Graph retrieval....')


        self.G_dataset = MMEaDtaset('../data/MMKG/seed{}/{}'.format(str(self.ratio), self.dataset.replace('_','-')), device=self.device, ratio=self.ratio)
        self.train_pair = self.G_dataset.train_pair

        train_pair = [(ent1, self.seed_dict[ent1]) for ent1 in self.en1_list]
        import random
        random.shuffle(train_pair)  
        sample_size = int(len(train_pair) * 0.7)
        random_sample = train_pair[:sample_size] 
        self.train_pair.extend(random_sample)
        # 可以考虑将前一阶段都识别过的作为训练集
        # rest_ent = set(self.en1_list)
        # total_ent = set(np.array(self.G_dataset.test_pair).T[0])
        # train_ent = total_ent - rest_ent
        # train_pair = [(ent1, self.seed_dict[ent1]) for ent1 in train_ent]
        # self.train_pair.extend(train_pair)

        self.train_pair = np.array(self.train_pair)
        self.train_pair = torch.tensor(self.train_pair).to(self.device)

        self.test_pair = [(ent1, self.seed_dict[ent1]) for ent1 in self.en1_list]
        self.test_pair = np.array(self.test_pair)
        self.test_pair = torch.tensor(self.test_pair)

        self.ent_adj, self.rel_adj, self.node_size, self.rel_size, \
            self.adj_list, self.r_index, self.r_val, self.triple_size, self.adj = self.G_dataset.reconstruct_search(None,
                                                                                                                None,
                                                                                                                self.G_dataset.kg1,
                                                                                                                self.G_dataset.kg2,
                                                                                                                new=True)
        self.structure_encoder = ST_Encoder_Module(
                node_hidden=128,
                rel_hidden=128,
                node_size=self.node_size,
                rel_size=self.rel_size,
                device=self.device,
                dropout_rate=0.1,
                depth=2).to(self.device)

        self.loss_model = Loss_Module(node_size=self.node_size, gamma=2).to(self.device)
        self.lr = 0.005
        self.optimizer = torch.optim.RMSprop(self.structure_encoder.parameters(), lr=self.lr)  # 3-9
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.999)
        train_epoch = 20
        batch_size = 500
        for epoch in trange(train_epoch):
            self.structure_encoder.train()
            for i in range(0, len(self.train_pair), batch_size):
                batch_pair = self.train_pair[i:i+batch_size]
                if len(batch_pair) == 0:
                    continue
                feature_list = self.structure_encoder(
                    self.ent_adj, self.rel_adj, self.node_size,
                    self.rel_size, self.adj_list, self.r_index, self.r_val,
                    self.triple_size, mask=None)
                loss = self.loss_model(batch_pair, feature_list[0], weight=True)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
        self.structure_encoder.eval()
        with torch.no_grad():
            feature_list = self.structure_encoder(
                self.ent_adj.to(self.device), self.rel_adj.to(self.device),
                self.node_size, self.rel_size,
                self.adj_list.to(self.device),
                self.r_index.to(self.device), self.r_val.to(self.device),
                self.triple_size,
                mask=None)
            self.out_feature = feature_list[0].cpu()

    def topk_ST_e1(self, k=10):
        gid1 = np.array(self.en1_list)
        gid2 = np.array(self.en2_list)
        
        self.out_feature = self.out_feature / (torch.linalg.norm(self.out_feature, dim=-1, keepdim=True) + 1e-5)
        index_a = torch.LongTensor(gid1)
        index_b = torch.LongTensor(gid2)

        Lvec = self.out_feature[index_a]
        Rvec = self.out_feature[index_b]
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)
        A_sim = torch.mm(Lvec, Rvec.t())
        self.topk_st_sim_matrix = A_sim.numpy()
        x_topk_indices = np.argsort(-self.topk_st_sim_matrix, axis=1)[:, :k]
        return x_topk_indices


    def complete_aligned_neighbor(self):
        # only rule2:邻居且已有对齐实体，该对齐实体的邻居作为当前实体的候选
        def rule2(ent_list):
            rule2_dict = {}
            for ent1 in ent_list:
                candi_tuple_list = []
               # A->[rel->B--B'->[(r1, b1)(r2, b2)(r3, b3)...]]
                # ent1的邻居
                if ent1 in self.trip_dic.keys():
                    neighbor_of_ent = self.trip_dic[ent1]
                else:
                    neighbor_of_ent = []
                has_aligned_dict = {}
                has_aligned_set = set()
                # 邻居已有对齐的实体
                for nei in neighbor_of_ent: #所有邻居
                    neigbor_name = self.Id_Name[nei]
                    if neigbor_name in self.Id_rel_dic[ent1].keys():
                        rel_name = self.Id_rel_dic[ent1][neigbor_name] #与ent1之间的关系
                    else:
                        rel_name = 'Unknown'

                    if nei in self.seed_dict.keys():
                        rel_propm_list = [] 
                        # 2.2已有对齐的邻居
                        aligned_nei = self.seed_dict[nei]
                        has_aligned_dict[nei] = aligned_nei
                        if aligned_nei in self.trip_dic.keys():
                            probli_neigbor = self.trip_dic[aligned_nei] # aligned_nei的邻居
                            has_aligned_set.add(nei)
                            for pn in probli_neigbor:
                                pn_name = self.Id_Name[pn]
                                if pn_name in self.Id_rel_dic[aligned_nei].keys():
                                    rel_name = self.Id_rel_dic[aligned_nei][pn_name]
                                    rel_propm_list.append((rel_name, pn, pn_name))
                            if len(rel_propm_list):
                                candi_tuple_list.append((rel_name, nei, aligned_nei, rel_propm_list))                                
                if len(candi_tuple_list):
                    rule2_dict[ent1] = candi_tuple_list
            return rule2_dict

        e1_complete = rule2(self.en1_list)
        e2_complete = rule2(self.en2_list)
        ent_complete = e1_complete | e2_complete
        return ent_complete

    

    def complete_e1_e2(self):
        def rule1_3_dic(ent_list):
            complet_dic = {}
            for ent1 in ent_list:
                # print(f"ent is {ent1}, its name is {self.Id_Name[ent1]}")
                rule1_list = [] # equal and aligned (equal_id, equal_id_aligned, Attr_list)
                rule2_list = [] # only aligned (rel, neigbor_id, neigbor_aligned)
                rule3_list = [] # only equal (rel, equal_name)
                if ent1 in self.euql_ent_dict.keys():
                    equal_ent1 = self.euql_ent_dict[ent1]
                else:
                    equal_ent1 = []
                equal_ent1_id_set = set()
                if len(equal_ent1):
                    equal_ent1_id_list = [int(it) for it in equal_ent1]
                    equal_ent1_id_set = set(equal_ent1_id_list)

                if ent1 in self.trip_dic.keys():
                    neighbor_of_ent = self.trip_dic[ent1]
                else:
                    neighbor_of_ent = []
                has_aligned_dict = {}
                has_aligned_set = set()
                for nei in neighbor_of_ent:
                    if nei in self.seed_dict.keys():
                        # 2.2已有对齐的邻居
                        has_aligned_dict[nei] = self.seed_dict[nei]
                        has_aligned_set.add(nei)
                
                # print(f"its neibor_has_aligned:{has_aligned_set}")

                # 有等价且等价实体存在对齐；交集部分；用等价实体的对齐实体的信息补充ent1
                # rule1_list: (equal_id, equal_id_aligned, Attr_list)
                align_and_equal = equal_ent1_id_set & has_aligned_set
                if len(align_and_equal):
                    for ent in align_and_equal:
                        equal_id_aligned = has_aligned_dict[ent]
                        Attr_list = self.Id_Attr[equal_id_aligned]
                        rule1_list.append((ent, equal_id_aligned, Attr_list))

                # 还有其它对齐实体 # only aligned 
                # rule2_list = [] (rel_name, neigbor_id, neigbor_aligned)
                remain_align = has_aligned_set - align_and_equal
                if len(remain_align):
                    for ent in remain_align:
                        neigbor_id = ent
                        neigbor_aligned_id = has_aligned_dict[ent]
                        neigbor_name = self.Id_Name[neigbor_id]
                        if neigbor_name in self.Id_rel_dic[ent1].keys():
                            rel_name = self.Id_rel_dic[ent1][neigbor_name]
                        else:
                            # print(self.Id_rel_dic[ent1])
                            # print(neigbor_name)
                            rel_name = 'Unknown'
                        rule2_list.append((rel_name, neigbor_id, neigbor_aligned_id))

                # 还有其它等价实体
                remain_equal = equal_ent1_id_set - align_and_equal
                # rule3_list = [] # only equal (rel, equal_id)
                if len(remain_equal):
                    for ent in remain_equal:
                        equal_id = ent
                        try:
                            if self.Id_Name[ent] in self.Id_rel_dic[ent1].keys():
                                rel_name = self.Id_rel_dic[ent1][self.Id_Name[ent]]
                            else:
                                continue
                        except:
                            print(self.Id_Name[ent])
                            print(self.Id_rel_dic[ent1])
                            exit(0)
                        rule3_list.append((rel_name, equal_id))

                rule_dic = {
                    'Rule1': rule1_list,
                    'Rule2': rule2_list,
                    'Rule3': rule3_list,
                }
                complet_dic[ent1] = rule_dic
                # print(rule_dic)
                # exit(0)
            return complet_dic
        
        e1_complete = rule1_3_dic(self.en1_list)
        e2_complete = rule1_3_dic(self.en2_list)
        ent_complete = e1_complete | e2_complete
        return ent_complete
    


    def complete_e1_e2_noequal(self):
        def rule2_dic(ent_list):
            complet_dic = {}
            for ent1 in ent_list:
                rule2_list = [] # only aligned (rel, neigbor_id, neigbor_aligned
                if ent1 in self.trip_dic.keys():
                    neighbor_of_ent = self.trip_dic[ent1]
                else:
                    neighbor_of_ent = []
                has_aligned_dict = {}
                has_aligned_set = set()
                for nei in neighbor_of_ent:
                    if nei in self.seed_dict.keys():
                        # 2.2已有对齐的邻居
                        has_aligned_dict[nei] = self.seed_dict[nei]
                        has_aligned_set.add(nei)
                # 还有其它对齐实体 # only aligned 
                # rule2_list = [] (rel_name, neigbor_id, neigbor_aligned)
                remain_align = has_aligned_set

                if len(remain_align):
                    for ent in remain_align:
                        neigbor_id = ent
                        neigbor_aligned_id = has_aligned_dict[ent]
                        neigbor_name = self.Id_Name[neigbor_id]
                        if neigbor_name in self.Id_rel_dic[ent1].keys():
                            rel_name = self.Id_rel_dic[ent1][neigbor_name]
                        else:
                            rel_name = 'Unknown'
                        rule2_list.append((rel_name, neigbor_id, neigbor_aligned_id))

                rule_dic = {
                    'Rule1': [],
                    'Rule2': [],
                    'Rule3': [],
                }
                complet_dic[ent1] = rule_dic
            return complet_dic
        
        e1_complete = rule2_dic(self.en1_list)
        e2_complete = rule2_dic(self.en2_list)
        ent_complete = e1_complete | e2_complete
        return ent_complete


    def equal_ent_load(self):
        equal_rel_path = f'../data/MMKG/Equal_rel/{self.dataset}_ratio{self.ratio}_{self.side}.txt'
        equal_dict = {}
        if os.path.exists(equal_rel_path):
            print("loading equal_rel_dictionary....")
            with open(equal_rel_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ent = parts[0]
                        equal_ents = parts[1:]  
                        equal_dict[int(ent)] = equal_ents
            
        else:
            print('...start equal_rel finding....')
            self.rel_equal()
            self.equal_ent_load()

        return equal_dict

    def rel_equal(self):
        rel_equal_list = []
        global total_tokens, has_equal_rel_num, avg_time
        avg_time = 0
        total_tokens = 0
        has_equal_rel_num = 0

        def ent_euqal_entity(ent_list):
            global total_tokens, has_equal_rel_num, avg_time
            ent_equa_list = {}
            count = 0
            for ent1 in ent_list:
                rel_dic = self.Id_rel_dic[ent1]
                rel_if_prompt = rel_if_equal_en_2(rel_dic, ent1, self.Id_Name[ent1])
                start_time = time.time()
                try:
                    completion = self.LLM.chat.completions.create(
                        model=self.LLM_type,
                        extra_body={"enable_thinking": False},
                        messages=rel_if_prompt)
                    llm_response = completion.choices[0].message.content
                    rel_equal_list.append(completion)
                    LLM_results = self.parse_llm_response(llm_response)
                    total_tokens += int(completion.usage.total_tokens)

                except Exception as e:
                    print(f"LLM推理出错: {e}")
                    LLM_results = []

                end_time = time.time()
                avg_time += (end_time - start_time)
                count += 1
                if count == 10:
                    break
                if LLM_results:  # 非空
                    has_equal_rel_num += 1
                    Id_list = []
                    for name in LLM_results:
                        if name in self.Name_id.keys():
                            Id_list.extend(self.Name_id[name])  # 修正为方括号索引
                    ent_equa_list[ent1] = Id_list
                else:
                    ent_equa_list[ent1] = []
                # break
            return ent_equa_list
        
        en1_equal_list = ent_euqal_entity(self.en1_list)
        en2_equal_list = ent_euqal_entity(self.en2_list)
        equal_rel_path = f'../data/MMKG/Equal_rel/{self.dataset}_ratio{self.ratio}_{self.side}.txt'
        with open(equal_rel_path, 'w') as f:
            for ent, equal_ent_list in en1_equal_list.items():
                line = str(ent) + '\t' + '\t'.join(str(item) for item in equal_ent_list) + '\n'
                f.write(line)
            
            for ent, equal_ent_list in en2_equal_list.items():
                line = str(ent) + '\t' + '\t'.join(str(item) for item in equal_ent_list) + '\n'
                f.write(line)
        record_path = f'/home/wluyao/VEA/Results/MMKG.txt'
        with open(record_path, 'a') as f:
            f.write("\n\n")
            f.write(f"{current_date}\n")
            f.write(f"Equal relation generation of {self.dataset}, side:{self.side}, ratio:{self.ratio}\n")
            f.write(f"Total entities :{len(self.en1_list) + len(self.en2_list)}\n")
            f.write(f"Has equal relation entities :{has_equal_rel_num}, ratio is {has_equal_rel_num/(len(self.en1_list) + len(self.en2_list)):.2f}\n")
            f.write(f"average token: {total_tokens/(len(self.en1_list) + len(self.en2_list)):.2f}\n")
            f.write(f"average time: {avg_time/(len(self.en1_list) + len(self.en2_list)):.2f}\n")


    def LLM_load(self):
        self.LLM = OpenAI(
        api_key="your_api_key_here",
        base_url="https://www.dmxapi.cn/v1"
    )
        
    def parse_llm_response(self, response):
        import re
        import ast
        response = response.strip()
        try:
            list_pattern = r'\[.*\]'
            match = re.search(list_pattern, response)
            if match:
                list_str = match.group()
                result_list = ast.literal_eval(list_str)
                if isinstance(result_list, list):
                    return result_list
        except:
            pass
        try:
            entity_pattern = r'["\']([^"\']+)["\']'
            entities = re.findall(entity_pattern, response)
            return entities
        except:
            return []
        
    # run
    def inference(self):
        hard_count = 0
        hit1 = 0
        hit10 = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        avg_time = 0
        print("start inference....")
        for epoch in range(0,1):
            if not len(self.en1_list):
                break
            else:
                for i, ent1 in enumerate(tqdm(self.en1_list)):
                    if ent1 == 2654:
                        print(self.seed_dict[ent1])
                        topk_indices = self.topk_e1(10)
                        top_ent2_list = [self.en2_list[int(it)] for it in topk_indices[i]]
                        top5_list = top_ent2_list[:5]
                        align_prompt = self.rule1_2_prompt_gen_en(ent1, top5_list, self.equal, self.align)
                        print(align_prompt)
                        messages = [
                                {"role": "user", "content": align_prompt}
                                ]   
                        completion = self.LLM.chat.completions.create(
                            model=self.LLM_type,
                            extra_body={"enable_thinking": True},
                            messages=messages,
                            stream=False)
                        llm_response = completion.choices[0].message.content
                        print(completion)
                        exit(0)
                    else:
                        continue
                    # topk_indices = self.topk_e1(10)
                    topk_indices = self.topk_ST_e1(10)
                    top_ent2_list = [self.en2_list[int(it)] for it in topk_indices[i]]
                    top5_list = top_ent2_list[:5]

                    align_prompt = self.rule1_2_prompt_gen_en(ent1, top5_list, self.equal, self.align)
                    messages = [
                        {"role": "user", "content": align_prompt}
                    ]
                    start_time = time.time()
                    try:
                        completion = self.LLM.chat.completions.create(
                            model=self.LLM_type,
                            messages=messages)

                        llm_response = completion.choices[0].message.content
                        LLM_results = self.parse_llm_response(llm_response)

                        prompt_tokens += int(completion.usage.prompt_tokens)
                        completion_tokens += int(completion.usage.completion_tokens)
                        total_tokens += int(completion.usage.total_tokens)

                    except Exception as e:
                        print(f"LLM推理出错: {e}")
                        LLM_results = []
                    end_time = time.time()
                    avg_time += (end_time - start_time)

                    if len(LLM_results):
                        try:
                            if int(LLM_results[0]) == self.seed_dict[ent1]:
                                hit1 += 1
                                self.en2_list.remove(int(LLM_results[0]))

                            if self.seed_dict[ent1] in top_ent2_list:
                                hit10 += 1
                            
                            hard_count += 1
                        except:
                            print("LLM 输出不符合规范")
                        self.en1_list.remove(ent1)
                        
        if len(self.en1_list):
            topk_indices = self.topk_ST_e1(10)
            for i in range(len(topk_indices)):
                hard_count += 1
                ent1 = self.en1_list[i]
                top_ent2_list = [self.en2_list[int(it)] for it in topk_indices[i]]
                ent2 = self.seed_dict[ent1]
                if ent2 == top_ent2_list[0]:
                    hit1 += 1
                if ent2 in top_ent2_list:
                    hit10 += 1

        print(f"hit@1:{hit1},")
        print(f"hit@10:{hit10}, ")
        print(f"hard_count:{hard_count}")

        print(f"prompt_tokens:{prompt_tokens/hard_count}")
        print(f"completion_tokens:{completion_tokens/hard_count}")
        print(f"total_tokens:{total_tokens/hard_count}")

        with open('../Results/MMKG.txt', 'a') as f:
            f.write("\n\n")
            f.write(f"{current_date}\n")
            f.write(f"LLM_type: {self.LLM_type}\n")
            f.write(f"Reranker_test_batch result of {self.dataset}, ratio:{self.ratio}, side:{self.side}\n")
            f.write(f"hard_count:{hard_count},\n")
            f.write(f"dev_hit@1:{hit1}, \n")
            f.write(f"hit@10:{hit10},\n")

            f.write("Average Token Usage per Case:\n")
            f.write(f" - Prompt tokens: {prompt_tokens/hard_count:.2f}\n")
            f.write(f" - Completion tokens: {completion_tokens/hard_count:.2f}\n")
            f.write(f" - Total tokens: {total_tokens/hard_count:.2f}\n")
            f.write(f" -Avg time:{avg_time/hard_count:.2f}\n")
      
    def test_code(self):
        hit1 = 0
        hit10 = 0

        hit1_2 = 0
        hit10_2 = 0

        for epoch in range(0,1):
            if not len(self.en1_list):
                break
            else:
                for i, ent1 in enumerate(tqdm(self.en1_list)):
                    topk_indices = self.topk_ST_e1(10)
                    top_ent2_list = [self.en2_list[int(it)] for it in topk_indices[i]]

                    # topk_indices2 = self.topk_e1(10)
                    # top_ent2_list2 = [self.en2_list[int(it)] for it in topk_indices2[i]]

                    if self.seed_dict[ent1] in top_ent2_list:
                        hit10 += 1
                        if self.seed_dict[ent1] == top_ent2_list[0]:
                            hit1 += 1

    def align_prompt_gen_en(self, ent1, top_ent2_list):
        task_desp = f"""
            Task: Please analyze the alignment possibility between source entity {ent1} and the candidate entity set.
            Requirements:
            1. Determine whether source entity {ent1} can potentially align with any candidate entity
            2. If alignment is possible, rank the candidate entity IDs from highest to lowest alignment probability
            3. If no candidate entity matches, output an empty list

            Output format: ["candidate_entity_id", "candidate_entity_id", ...] or []

            Note: Output only the list without any additional explanations.
            """
        
        source_info = f""" Source entity information: ID {ent1}, """
        if self.Id_Name[ent1] != 'Null':
            source_info += f"Name: {self.Id_Name[ent1]}, "
        if self.Id_Attr[ent1] != "No_property":
            source_info += f"Attributes: {self.Id_Attr[ent1]}, "
        
        source_info += self.complete_info_prompt(ent1)

        # Candidate entity information
        candidate_info = "Candidate entity information:\n"
        for i, ent2 in enumerate(top_ent2_list):
            candidate_info += f"Candidate entity {i+1} (ID: {ent2}): "
            if ent2 in self.Id_Name and self.Id_Name[ent2] != 'Null':
                candidate_info += f"Name: {self.Id_Name[ent2]}, "
            if ent2 in self.Id_Attr and self.Id_Attr[ent2] != "No_property":
                candidate_info += f"Attributes: {self.Id_Attr[ent2]}, "
            candidate_info += self.complete_info_prompt(ent2)
            candidate_info += "\n"  # Newline after each candidate entity
        
        # Construct full prompt
        full_prompt = task_desp + "\n" + source_info + "\n\n" + candidate_info + "\nPlease output the result:"
        
        return full_prompt

    def complete_info_prompt_en(self, entity):
        complete_dict = self.ent_complet[entity]
        info = """ """
        
        if complete_dict['Rule1'] != []:
            # (ent, equal_id_aligned, Attr_list)
            for ent, equal_id_aligned, Attr_list in complete_dict['Rule1']:
                if self.Id_Name[equal_id_aligned] != 'Null':
                    info += f"""
                        It is the same type of entity as entity {equal_id_aligned} \t {self.Id_Name[equal_id_aligned]},
                    """
                else:
                    info += f"""
                        It is the same type of entity as entity {equal_id_aligned},
                    """
                info += f"""
                    Therefore, it may have similar attributes such as {Attr_list} (may have the same attributes but different attribute values).
                """
                    
        if complete_dict['Rule2'] != []:
            # (rel_name, neigbor_id, neigbor_aligned)
            for rel_name, neigbor_id, neigbor_aligned in complete_dict['Rule2']:
                if self.Id_Name[neigbor_id] != 'Null':
                    info += f"""
                        The relationship between it and entity {neigbor_id}\t{self.Id_Name[neigbor_id]} is {rel_name},
                    """
                else:
                    info += f"""
                        The relationship between it and entity {neigbor_id} is {rel_name},
                    """
                if self.Id_Name[neigbor_aligned] != 'Null':
                    info += f"""
                        and entity {neigbor_id} is aligned with entity {neigbor_aligned}\t{self.Id_Name[neigbor_aligned]},
                    """
                else:
                    info += f"""
                        and entity {neigbor_id} is aligned with entity {neigbor_aligned},
                    """
                info += f"""
                        Therefore, the entity aligned with {entity} may have the relationship {rel_name} with entity {neigbor_aligned}.
                    """
                        

        if complete_dict['Rule3'] != []:
            # only equal (rel, equal_id)
            for rel_name, equal_id in complete_dict['Rule3']:
                if self.Id_Name[equal_id] != 'Null':
                    info += f"""
                        It is the same type of entity as entity {equal_id}\t{self.Id_Name[equal_id]},
                    """
                else:
                    info += f"""
                        It is the same type of entity as entity {equal_id},"""
                info += f"""
                        and it has a {rel_name} relationship with entity {equal_id};"""
        
        return info
    

    def rule2_prompt_gen_en(self, ent1, top_ent2_list):
        task_desp = f"""
            Task: Please analyze the alignment possibility between source entity {ent1} and the candidate entity set.
            Requirements:
            1. Determine whether source entity {ent1} can potentially align with any candidate entity (the candidate entity can be from the candidate entity list or additional information or additional information of source entity.)
            2. If no candidate entity matches, output an empty list

            Output format: ["candidate_entity_id", "candidate_entity_id", ...] or []

            Note: Output only the list without any additional explanations.
            """
        source_info = f""" Source entity information: ID {ent1}, """
        if self.Id_Name[ent1] != 'Nul' and self.Id_Name[ent1] != 'Null':
            source_info += f"Name: {self.Id_Name[ent1]}, "
        if self.Id_Attr[ent1] != []:
            source_info += f"Attributes: {self.Id_Attr[ent1]}, "

        if self.neighbor_align is not None and ent1 in self.ent_rule2_complete.keys():
            source_info += "Additional information:"
            for rel_name, nei, aligned_nei, rel_propm_list in self.ent_rule2_complete[ent1]:
                source_info += f"The relationship between it and entity {nei} {self.Id_Name[nei]} is {rel_name}, \
                                and the entity {nei} has aligned with entity {aligned_nei} {self.Id_Name[aligned_nei]}. \
                                thus, one of neigbors of the entity {aligned_nei} is probable to align with entity {nei}.\
                                there are the neigbor and relation of entity {aligned_nei} (format: (relation, entity_id, entity_name)). \
                                "
                for rpl in rel_propm_list:
                    source_info += f"({rpl[0]},{rpl[1]},{rpl[2]})\n"

        candidate_info = "Candidate entity information:\n"
        for i, ent2 in enumerate(top_ent2_list):
            candidate_info += f"Candidate entity {i+1} (ID: {ent2}): "
            if ent2 in self.Id_Name and self.Id_Name[ent2] != 'Null' and self.Id_Name[ent2] != 'Nul':
                candidate_info += f"Name: {self.Id_Name[ent2]}, "
            if ent2 in self.Id_Attr and self.Id_Attr[ent2] != []:
                candidate_info += f"Attributes: {self.Id_Attr[ent2]}, "


            if self.neighbor_align is not None and ent2 in self.ent_rule2_complete.keys():
                candidate_info += "Additional information:"
                for rel_name, nei, aligned_nei, rel_propm_list in self.ent_rule2_complete[ent2]:
                    candidate_info += f"The relationship between it and entity {nei} {self.Id_Name[nei]} is {rel_name}, \
                                    and the entity {nei} has aligned with entity {aligned_nei} {self.Id_Name[aligned_nei]}. \
                                    thus, one of neigbors of the entity {aligned_nei} is probable to align with entity {nei}.\
                                    there are the neigbor and relation of entity {aligned_nei} (format: (relation, entity_id, entity_name)). "
                    for rpl in rel_propm_list:
                        candidate_info += f"({rpl[0]},{rpl[1]},{rpl[2]})\n"

        full_prompt = task_desp + "\n" + source_info + "\n\n" + candidate_info + "\nPlease output the result:"
        
        return full_prompt
    

    def rule2_prompt_gen2_en(self, ent1, top_ent2_list):
        task_desp = f"""
            Task: Please analyze the alignment possibility between source entity {ent1} and the candidate entity set.
            Requirements:
            1. Determine whether source entity {ent1} can potentially align with any candidate entity; 
            2. We are not looking for entities that are "exactly identical," but rather for those that are "most likely to refer to the same thing." Minor discrepancies in non-essential attributes are allowed, as long as the core identifiers, key attributes, or relational networks exhibit sufficient similarity.
            3. If no candidate entity matches, output an empty list;

            Output format: ["candidate_entity_id", "candidate_entity_id", ...] or []

            Note: Output only the list without any additional explanations.
            """
        source_info = f""" Source entity information: ID {ent1}, """
        length1 = len(source_info)
        if self.Id_Name[ent1] != 'Null' and self.Id_Name[ent1] != 'Nul':
            source_info += f"Name: {self.Id_Name[ent1]}, "

        if self.Id_Attr[ent1] != [] and self.Id_Attr[ent1] != '':
            source_info += f"Attributes: {self.Id_Attr[ent1]}, "
        length2 = len(source_info)
        if length2 - length1 < 500 and ent1 in self.Id_rel_dic:
            source_info += "It has neighbors with the following relations:\n"
            rel_dic = self.Id_rel_dic[ent1]
            for ent_name in rel_dic.keys():
                rel_name = rel_dic[ent_name]
                length3 = len(source_info)
                if length3 - length1>500:
                    break
                if ent_name not in ('Nul', 'Null'):
                    entity_id = self.Name_id.get(ent_name, 'unknown')
                    source_info += f"- It has a {rel_name} relation with entity {ent_name} (ID: {entity_id});\n"

        candidate_info = "Candidate entity information:\n"
        for i, ent2 in enumerate(top_ent2_list):
            candidate_info += f"Candidate entity {i+1} (ID: {ent2}): "
            length1 = len(candidate_info)
            if ent2 in self.Id_Name and self.Id_Name[ent2] != 'Null' and self.Id_Name[ent2] != 'Nul':
                candidate_info += f"Name: {self.Id_Name[ent2]}, "
            if ent2 in self.Id_Attr and self.Id_Attr[ent2] != [] and self.Id_Attr[ent2] != '':
                candidate_info += f"Attributes: {self.Id_Attr[ent2]}, "
            length2 = len(candidate_info)

            if length2 - length1 < 500 and ent2 in self.Id_rel_dic:
                candidate_info += "It has neighbors with the following relations:\n"
                rel_dic = self.Id_rel_dic[ent2]
                for ent_name in rel_dic.keys():
                    rel_name = rel_dic[ent_name]
                    length3 = len(candidate_info)
                    if length3 - length1>500:
                        break
                    if ent_name not in ('Nul', 'Null'):
                        entity_id = self.Name_id.get(ent_name, 'unknown')
                        candidate_info += f"- It has a {rel_name} relation with entity {ent_name} (ID: {entity_id});\n"

        full_prompt = task_desp + "\n" + source_info + "\n\n" + candidate_info + "\nPlease output the result:"
        
        return full_prompt


    def rule1_2_prompt_gen_en(self, ent1, top5_list, equal_rule=False, align_rule=False):
        task_desp = f"""
            Task: Using the provided entity neighbor/attribute information, analyze the alignment possibility between source entity {ent1} and the candidate entity set.
            Requirements:
            1. Determine whether source entity {ent1} can potentially align with any candidate entity; 
            2. We are not looking for entities that are "exactly identical," but rather for those that are "most likely to refer to the same thing." 
               Minor discrepancies in non-essential attributes are allowed, 
               as long as the core neiborhood, key attributes exhibit sufficient similarity.
           

            Output format: ["candidate_entity_id", "candidate_entity_id", ...] 
            Note: Output only the list without any additional explanations.
            """
            #  3. If no candidate entity matches, output an empty list; or []
        source_info = f""" Source entity information: ID {ent1}, \n"""
        length1 = len(source_info)
        if self.Id_Name[ent1] != 'Null' and self.Id_Name[ent1] != 'Nul':
            source_info += f"Name: {self.Id_Name[ent1]}, \n"

        if self.Id_Attr[ent1] != [] and self.Id_Attr[ent1] != '':
            source_info += f"Attributes: {self.Id_Attr[ent1]}, \n"

        equal_set = set()
        if equal_rule and ent1 in self.euql_ent_dict.keys(): #如果有等价邻居，就把邻居的属性（只保留属性名称）might be
            for equal_ent in self.euql_ent_dict[ent1]:
                if equal_ent in self.Id_Attr.keys() and self.Id_Attr[equal_ent] != '' and self.Id_Attr[equal_ent] != []:
                    equal_set.add(equal_ent)
                    item_list = self.Id_Attr[equal_ent].split('|||')
                    might_attr_list = []
                    for item in item_list:
                        might_attr_list.append(item.split('^^^'))
                    might_attr = ';'.join(might_attr_list)
                    source_info += f"It MIGHT be equal type with {self.Id_Name[equal_ent]} (ID:{equal_ent}), so it MIGHT have the Attributes like: {might_attr};\n"

        align_set = set()
        if align_rule and ent1 in self.align_ent_dict.keys():
            candi_tuple_list = self.align_ent_dict[ent1]
            for rel_name, nei, aligned_nei, rel_propm_list in candi_tuple_list:
                align_set.add(nei)
                source_info += f"It has a {rel_name} relation with entity {self.Id_Name[nei]} (ID: {nei}),\n \
                                which is aligned with entity {self.Id_Name[aligned_nei]} (ID: {aligned_nei}),\n \
                                SO, the aligned entity of source entity MIGHT have the relation {rel_name} with ID: {aligned_nei};\n"

        intersection = equal_set & align_set
        if intersection:
            for equal_ent in intersection:
                aligned_ = self.seed_dict[equal_ent]
                if aligned_ in self.Id_Attr.keys() and self.Id_Attr[aligned_] != '' and self.Id_Attr[aligned_] != []:
                    item_list = self.Id_Attr[equal_ent].split('|||')
                    might_attr_list = []
                    for item in item_list:
                        might_attr_list.append(item.split('^^^'))
                    might_attr = ';'.join(might_attr_list)
                    source_info += f"It MIGHT be equal type with ID:{aligned_} {self.Id_Name[aligned_]}, so it MIGHT have the Attributes like: {might_attr};\n"


        if ent1 in self.Id_rel_dic:
            source_info += "It has neighbors with the following relations:\n"
            rel_dic = self.Id_rel_dic[ent1]
            for ent_name in rel_dic.keys():
                rel_name = rel_dic[ent_name]
                length3 = len(source_info)
                if length3 - length1>500: # 控制relation的长度
                    break
                if ent_name not in ('Nul', 'Null'):
                    entity_id = self.Name_id.get(ent_name, 'unknown')
                    source_info += f"- It has a {rel_name} relation with entity {ent_name} (ID: {entity_id});\n"

        candidate_info = "Candidate entity information:\n"
        for i, ent2 in enumerate(top5_list):
            candidate_info += f"Candidate entity {i+1} (ID: {ent2}): "
            length1 = len(candidate_info)
            if ent2 in self.Id_Name and self.Id_Name[ent2] != 'Null' and self.Id_Name[ent2] != 'Nul':
                candidate_info += f"Name: {self.Id_Name[ent2]}, "
            if ent2 in self.Id_Attr and self.Id_Attr[ent2] != [] and self.Id_Attr[ent2] != '':
                candidate_info += f"Attributes: {self.Id_Attr[ent2]}, "
            length2 = len(candidate_info)

            if length2 - length1 < 500 and ent2 in self.Id_rel_dic:
                candidate_info += "It has neighbors with the following relations:\n"
                rel_dic = self.Id_rel_dic[ent2]
                for ent_name in rel_dic.keys():
                    rel_name = rel_dic[ent_name]
                    length3 = len(candidate_info)
                    if length3 - length1>500:
                        break
                    if ent_name not in ('Nul', 'Null'):
                        entity_id = self.Name_id.get(ent_name, 'unknown')
                        candidate_info += f"- It has a {rel_name} relation with entity {ent_name} (ID: {entity_id});\n"

        full_prompt = task_desp + "\n" + source_info + "\n\n" + candidate_info + "\nPlease output the result:"
        
        return full_prompt
            

# t2 = PART_3('FB15K_DB15K', None, 0.2, equal_rule= True, align_rule=True)
# t2.inference()

# t4 = PART_3('FB15K_YAGO15K', None, 0.2, equal_rule= True, align_rule=True)
# t4.inference()

# t0 = PART_3('FB15K_DB15K', None, 0.5, equal_rule= True, align_rule=True)
# t0.inference()

# t1 = PART_3('FB15K_DB15K', None, 0.8, equal_rule= True, align_rule=True)
# t1.inference()

# t1 = PART_3('FB15K_DB15K', None, 0.8, equal_rule= True, align_rule=True)
# t1.inference()

# t4 = PART_3('FB15K_YAGO15K', 'N', 0.8, equal_rule= True, align_rule=True)
# t4.inference()




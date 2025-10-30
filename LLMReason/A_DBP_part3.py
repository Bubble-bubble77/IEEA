    
from DBP_retrieval import *
from prompt import *
from openai import OpenAI
import os
import pytz
from datetime import datetime
from Prepare_EMB.DBP15K.util import *
from Prepare_EMB.DBP15K.dataset import DBPDataset
from Prepare_EMB.DBP15K.model import ST_Encoder_Module, Loss_Module
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange
import time

timezone = pytz.timezone('Asia/Shanghai')
current_date = datetime.now(timezone).strftime("%m%d_%H%M")

class DBP_Part_3():
    def __init__(self, dataset, side=None, equal_rule= None, align_rule=None, rule3=None):
        self.dataset = dataset
        print('-------------')
        print(current_date)
        print(self.dataset)
        self.side = side
        self.re = Retrivel(dataset)
        self.en1_list, self.en2_list = self.ent1_ent2_load()
        self.seed_dict = self.seed_load()

        self.EntN_Attr = self.re.original_attr
        self.Id_Attr = self.re.id_Attr
        self.Id_Name = self.re.id_o_Name
        self.index1, self.index2 = self.re.index_1, self.re.index_2
        self.Name_id = self.Name_Id()
        self.Id_rel_dic = self.Rel_list_to_dict()
        self.trip_1_dic, self.trip_2_dic = self.Triple_load()
        self.trip_dic = self.trip_1_dic | self.trip_2_dic

        self.LLM_load()
        # self.LLM_type='qwen3-max'
        self.LLM_type = 'deepseek-v3.2-exp'
        # self.LLM_type = 'deepseek-v3.1'

        self.equal = equal_rule
        self.align = align_rule
        self.rule3 = rule3

        # rule1
        if self.equal is not None:
            self.euql_ent_dict = self.equal_ent_load()
            print("equal_ent has loaded..")

        # rule2
        if self.align is not None:
            self.align_ent_dict = self.complete_aligned_neighbor()
            print("align_neighbor has loaded..")
            with open('../Results/DBP.txt', 'a') as f:
                f.write("\n\n")
                f.write(f"{current_date}\n")
                f.write(f"align neighbor num of {self.dataset}, side:{self.side}: {len(self.align_ent_dict)}\n")
                f.write(f"align neighbor ratio of {self.dataset}, side:{self.side}: {len(self.align_ent_dict)/len(self.en1_list)}\n")

        # rule3
        if self.equal is not None and self.align is not None:
            self.equal_AND_align = 0
            for ent in self.align_ent_dict.keys():
                if ent in self.euql_ent_dict.keys() and self.euql_ent_dict[ent] != []:
                    self.equal_AND_align += 1
            with open('../Results/DBP.txt', 'a') as f:
                f.write(f"equal AND align num of {self.dataset}, side:{self.side}: {self.equal_AND_align}\n")
                f.write(f"equal AND align ratio of {self.dataset}, side:{self.side}: {self.equal_AND_align/len(self.en1_list)}\n")
        

        self.device = 'cuda:0'
        if self.equal is not None and self.align is not None and self.rule3 is not None:
            self.structure_learn()
        else:
            self.es_score = self.sim_score_load()


    def seed_load(self):
        filepath = f'../data/DBP15K/DBP_1/'
        filename = os.path.join(filepath, self.dataset, 'ill_ent_ids')
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
        with open(f'../data//DBP15K/DBP_1/{self.dataset}/ent_ids_1', 'r') as f:
            for line in f:
                id_1, ol_name = line.strip().split('\t')
                Name_id[ol_name.split('/')[-1]] = int(id_1)

        with open(f'../data//DBP15K/DBP_1/{self.dataset}/ent_ids_2', 'r') as f:
            for line in f:
                id_2, ol_name = line.strip().split('\t')
                Name_id[ol_name.split('/')[-1]] = int(id_2)
        return Name_id
    
    def Triple_load(self):
        triple1_path =  f"../data/DBP15K/DBP_1/{self.dataset}/triples_1"
        triple2_path = f"../data/DBP15K/DBP_1/{self.dataset}/triples_2"

        def trip_load(path, index_dict):
            trip_dic = {}
            with open(path,'r') as f:
                for line in f:
                    ent1, _, ent2 = line.strip().split('\t')
                    if ent1 not in trip_dic.keys():
                        trip_dic[int(ent1)] = [int(ent2)]
                    else:
                        trip_dic[int(ent1)].append(int(ent2))
            return trip_dic
        
        trip_1_dic = trip_load(triple1_path, self.index1)
        trip_2_dic = trip_load(triple2_path, self.index2)
        return trip_1_dic, trip_2_dic
    
    def Rel_list_to_dict(self):
        s_d, t_d = self.dataset.split('_')
        rel_path_1 = f"../data/DBP15K/dbpdata/DBP15k/{self.dataset}/{s_d}_rel_triples"
        rel_path_2 = f"../data/DBP15K/dbpdata/DBP15k/{self.dataset}/{t_d}_rel_triples"

        def rel_ent2_load(path):
            Id_rel_dic = {}
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    ent1 = line.split('\t')[0].split('/')[-1]
                    rel = line.split('\t')[1].split('/')[-1]
                    ent2 = line.split('\t')[2].split('/')[-1] 

                    ent1_id = self.Name_id[ent1]
                    if ent1_id not in Id_rel_dic.keys():
                        Id_rel_dic[ent1_id] = {ent2: rel} # {ent2_name:rel_Name}
                    else:
                        Id_rel_dic[ent1_id][ent2] = rel
            return Id_rel_dic
            
        s_d_rel_list = rel_ent2_load(rel_path_1)  #[ent_id:{ent_name: rel_name}]
        t_d_rel_list = rel_ent2_load(rel_path_2)

        rel_list = s_d_rel_list | t_d_rel_list

        return rel_list


    def complete_aligned_neighbor(self):
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
                for nei in neighbor_of_ent: 
                    if nei in self.seed_dict.keys():
                        rel_propm_list = [] 
                        aligned_nei = self.seed_dict[nei]
                        has_aligned_dict[nei] = aligned_nei
                        if aligned_nei in self.trip_dic.keys():
                            probli_neigbor = self.trip_dic[aligned_nei] # aligned_nei的邻居
                            has_aligned_set.add(nei)
                            for pn in probli_neigbor:
                                pn_name = self.Id_Name[pn]
                                if aligned_nei in self.Id_rel_dic.keys() and pn_name in self.Id_rel_dic[aligned_nei].keys():
                                    rel_name = self.Id_rel_dic[aligned_nei][pn_name]
                                    rel_propm_list.append((rel_name, pn, pn_name))
                            if len(rel_propm_list):
                                candi_tuple_list.append((rel_name, nei, aligned_nei, rel_propm_list))                                
                if len(candi_tuple_list):
                    rule2_dict[ent1] = candi_tuple_list
            return rule2_dict

        e1_complete = rule2(self.en1_list)
        ent_complete = e1_complete
        return ent_complete


    def ent1_ent2_load(self):
        Imcomplete_data_save_path = f'../data/DBP/IC_data/'
        Ic_e1_path = os.path.join(Imcomplete_data_save_path, f'{self.dataset}_{self.side}_top_Ic_e1.txt')
        Ic_e2_path = os.path.join(Imcomplete_data_save_path, f'{self.dataset}_{self.side}_Ic_e2.txt')

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

    def structure_learn(self):
        print('Graph retrieval....')


        self.G_dataset = DBPDataset('../data/DBP15K/DBP_1/{}'.format(self.dataset),
                                    device=self.device)
        self.train_pair = self.G_dataset.train_pair

        train_pair = [(ent1, self.seed_dict[ent1]) for ent1 in self.en1_list]
        import random
        random.shuffle(train_pair)  
        sample_size = int(len(train_pair) * 0.2)
        random_sample = train_pair[:sample_size] 
        self.train_pair.extend(random_sample)

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
                depth=4).to(self.device)

        self.loss_model = Loss_Module(node_size=self.node_size, gamma=2).to(self.device)
        self.lr = 0.005
        self.optimizer = torch.optim.RMSprop(self.structure_encoder.parameters(), lr=self.lr)  # 3-9
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.999)
        # train_epoch = 10
        train_epoch = 50
        batch_size = 1024
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
    
    def sim_score_load(self):
        folder_path = '../data/DBP15K/DBP_fuse'

        if self.side is not None:
            file_name = os.path.join(folder_path,
                                     f'Fuse_{self.dataset}_{self.side}.npy')
        else:
            file_name = os.path.join(folder_path, f'Fuse_{self.dataset}.npy')
        moda_matrix = np.load(file_name)
        m, _ = moda_matrix.shape
        es_score = (moda_matrix - moda_matrix.min()) / (moda_matrix.max() - moda_matrix.min())
        return es_score
    
    def topk_e1(self, k=10):
        ent2_list = [item - self.es_score.shape[0] for item in self.en2_list]
        new_esscore_matrix = self.es_score[np.ix_(self.en1_list, ent2_list)]
        topk_indices = np.argsort(-new_esscore_matrix, axis=1)[:, :k]
        return topk_indices
    
    def equal_ent_load(self):
        equal_rel_path = f'../data/DBP15K/Equal_rel/{self.dataset}_{self.side}.txt'
        equal_dict = {}
        if os.path.exists(equal_rel_path):
            with open(equal_rel_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ent = parts[0]
                        equal_ents = parts[1:]  
                        equal_dict[int(ent)] = equal_ents
        else:
            print('...')
            self.rel_equal()
            self.equal_ent_load()
        return equal_dict
    

    def rel_equal(self):
        global total_tokens, has_equal_rel_num, avg_time
        avg_time = 0
        total_tokens = 0
        has_equal_rel_num = 0

        def ent_euqal_entity(ent_list):
            global total_tokens, has_equal_rel_num, avg_time
            ent_equa_list = {}
            for ent1 in ent_list:
                if ent1 in self.Id_rel_dic.keys():
                    rel_dic = self.Id_rel_dic[ent1]
                else:
                    continue
                rel_if_prompt = rel_if_equal_en(rel_dic, ent1, self.Id_Name[ent1])
                start_time = time.time()
                try:
                    completion = self.LLM.chat.completions.create(
                        model=self.LLM_type,
                        messages=rel_if_prompt)
                    llm_response = completion.choices[0].message.content
                    LLM_results = self.parse_llm_response(llm_response)
                    total_tokens += int(completion.usage.total_tokens)

                except Exception as e:
                    print(f"LLM推理出错: {e}")
                    LLM_results = []
                end_time = time.time()
                avg_time += (end_time - start_time)

                if LLM_results:  # 非空
                    has_equal_rel_num += 1
                    Id_list = []
                    for name in LLM_results:
                        if name in self.Name_id.keys():
                            Id_list.append(self.Name_id[name])  
                    ent_equa_list[ent1] = Id_list
                else:
                    ent_equa_list[ent1] = []

            return ent_equa_list
            
        en1_equal_list = ent_euqal_entity(self.en1_list)
        # en2_equal_list = ent_euqal_entity(self.en2_list)

        # save
        equal_rel_path = f'../data/DBP15K/Equal_rel/{self.dataset}_{self.side}.txt'
        with open(equal_rel_path, 'w') as f:
            for ent, equal_ent_list in en1_equal_list.items():
                line = str(ent) + '\t' + '\t'.join(str(item) for item in equal_ent_list) + '\n'
                f.write(line)
            
            # for ent, equal_ent_list in en2_equal_list.items():
            #     line = str(ent) + '\t' + '\t'.join(str(item) for item in equal_ent_list) + '\n'
            #     f.write(line)

        # 记录推理时间；推理token平均数；真正有等价实体的比例
        record_path = f'/home/wluyao/VEA/Results/DBP.txt'
        with open(record_path, 'a') as f:
            f.write("\n\n")
            f.write(f"{current_date}\n")
            f.write(f"Equal relation generation of {self.dataset}, side:{self.side}\n")
            f.write(f"Total entities :{len(self.en1_list)}\n")
            f.write(f"Has equal relation entities :{has_equal_rel_num}, ratio is {has_equal_rel_num/(len(self.en1_list)):.2f}\n")
            f.write(f"average token: {total_tokens/(len(self.en1_list)):.2f}\n")
            f.write(f"average time: {avg_time/(len(self.en1_list)):.2f}\n")

        # exit(0)


    def LLM_load(self):
        self.LLM = OpenAI(
                api_key="sk-yourapikeyhere",
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
        llm_ratio = len(self.en1_list)
        for epoch in range(0,1):
            if not len(self.en1_list):
                break
            else:
                for i, entity in enumerate(tqdm(self.en1_list)):
                    ent1 = entity
                    if self.equal is not None and self.align is not None and self.rule3 is not None:
                        topk_indices = self.topk_ST_e1(10)
                    else:
                        topk_indices = self.topk_e1(10)
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
                            hard_count += 1
                            if int(LLM_results[0]) == self.seed_dict[ent1]:
                                hit1 += 1
                            if self.seed_dict[ent1] in top_ent2_list:
                                hit10 += 1
                            if int(LLM_results[0]) in self.en2_list:
                                self.en2_list.remove(int(LLM_results[0]))
                            self.en1_list.remove(ent1)
                        except:
                            continue
        LLM_right_num = hit1
        LLM_num = llm_ratio-len(self.en1_list)

        if len(self.en1_list):
            llm_ratio = 1 - (len(self.en1_list) / llm_ratio)
            if self.equal is not None and self.align is not None and self.rule3 is not None:
                topk_indices = self.topk_ST_e1(10)
            else:
                topk_indices = self.topk_e1(10)
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
        print(f"hard_count:{hard_count} ")

        print(f"prompt_tokens:{prompt_tokens/hard_count}")
        print(f"completion_tokens:{completion_tokens/hard_count}")
        print(f"total_tokens:{total_tokens/hard_count}")

        with open('../Results/DBP.txt', 'a') as f:
            f.write("\n\n")
            f.write(f"{current_date}\n")
            f.write(f"equal:{self.equal}; align: {self.align} rule3: {self.rule3} \n")
            f.write(f"Reranker_test_batch result of {self.dataset}, side:{self.side}\n")
            f.write(f"hard_count:{hard_count},\n")
            f.write(f"dev_hit@1:{hit1}, \n")
            f.write(f"hit@10:{hit10},\n")

            f.write("Average Token Usage per Case:\n")
            f.write(f"- Prompt tokens: {prompt_tokens/hard_count:.2f}\n")
            f.write(f"- Completion tokens: {completion_tokens/hard_count:.2f}\n")
            f.write(f"- Total tokens: {total_tokens/hard_count:.2f}\n")
            f.write(f"- Avg time:{avg_time/hard_count:.2f}\n")
            f.write(f"- LLM ratio:{llm_ratio:4f}\n\n")

            f.write(f"[LLM_num : {LLM_num}]\n")
            f.write(f"[LLM_right_num : {LLM_right_num}]\n")


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
        source_info = f""" Source entity information: ID {ent1}, \n"""

        if self.Id_Name[ent1] != 'Null' and self.Id_Name[ent1] != 'Nul':
            source_info += f"Name: {self.Id_Name[ent1]}, \n"
        if ent1 in self.Id_Name.keys() and self.Id_Name[ent1] in self.EntN_Attr.keys() and self.EntN_Attr[self.Id_Name[ent1]] != [] and self.EntN_Attr[self.Id_Name[ent1]] != '':
            source_info += f"Attributes: {' ; '.join(self.EntN_Attr[self.Id_Name[ent1]])}, \n"

        equal_set = set()
        if equal_rule and ent1 in self.euql_ent_dict.keys(): 
            for equal_ent in self.euql_ent_dict[ent1]:
                if equal_ent in self.Id_Name.keys() and self.Id_Name[equal_ent] in self.EntN_Attr.keys() and self.EntN_Attr[self.Name_id[equal_ent]] != []:
                    equal_set.add(equal_ent)
                    might_attr_list = self.EntN_Attr[self.Name_id[equal_ent]]
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
        if intersection and self.rule3 is not None:
            for equal_ent in intersection:
                aligned_ = self.seed_dict[equal_ent]
                if aligned_ in self.Id_Name.keys() and self.Id_Name[aligned_] in self.EntN_Attr.keys() and self.EntN_Attr[self.Id_Name[aligned_]] != []:
                    might_attr_list = self.EntN_Attr[self.Id_Name[aligned_]]
                    might_attr = ';'.join(might_attr_list)
                    source_info += f"It MIGHT be equal type with {self.Id_Name[aligned_]}(ID:{aligned_}), so it MIGHT have the Attributes like: {might_attr};\n"
        
        length1 = len(source_info)
        if ent1 in self.Id_rel_dic:
            source_info += "It has neighbors with the following relations:\n"
            rel_dic = self.Id_rel_dic[ent1]
            for ent_name in rel_dic.keys():
                rel_name = rel_dic[ent_name]
                length3 = len(source_info)
                if length3 - length1 > 500: # 控制relation的长度
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

            if ent2 in self.Id_Name.keys() and self.Id_Name[ent2] in self.EntN_Attr.keys() and self.EntN_Attr[self.Id_Name[ent2]] != []:
                candidate_info += f"Attributes: {' ; '.join(self.EntN_Attr[self.Id_Name[ent2]])}, \n"

            length2 = len(candidate_info)

            if ent2 in self.Id_rel_dic:
                candidate_info += "It has neighbors with the following relations:\n"
                rel_dic = self.Id_rel_dic[ent2]
                for ent_name in rel_dic.keys():
                    rel_name = rel_dic[ent_name]
                    length3 = len(candidate_info)
                    if length3 - length2>500:
                        break
                    if ent_name not in ('Nul', 'Null'):
                        entity_id = self.Name_id.get(ent_name, 'unknown')
                        candidate_info += f"- It has a {rel_name} relation with entity {ent_name} (ID: {entity_id});\n"

        full_prompt = task_desp + "\n" + source_info + "\n\n" + candidate_info + "\nPlease output the result:"
        
        return full_prompt

    def test_code(self):
        hit1 = 0
        hit10 = 0
        for epoch in range(0,1):
            if not len(self.en1_list):
                break
            else:
                for i, ent1 in enumerate(tqdm(self.en1_list)):
                    topk_indices = self.topk_ST_e1(10)
                    top_ent2_list = [self.en2_list[int(it)] for it in topk_indices[i]]

                    if self.seed_dict[ent1] in top_ent2_list:
                        hit10 += 1
                        if self.seed_dict[ent1] == top_ent2_list[0]:
                            hit1 += 1

        print(len(self.en1_list))
        print(hit1)
        print(hit10)

# ts4 = DBP_Part_3('zh_en', None, equal_rule= None, align_rule=None, rule3 = None)
# ts4.inference()

# ts1 = DBP_Part_3('zh_en', None, equal_rule= True, align_rule=None, rule3 = None)
# ts1.inference()

# ts2 = DBP_Part_3('zh_en', None, equal_rule= None, align_rule=True, rule3 = None)
# ts2.inference()

# ts3 = DBP_Part_3('zh_en', None, equal_rule= True, align_rule=True, rule3 = None)
# ts3.inference()

# ts5 = DBP_Part_3('zh_en', None, equal_rule= True, align_rule=True, rule3 = True)
# ts5.inference()


# # ja-en
# ts4 = DBP_Part_3('ja_en', None, equal_rule= None, align_rule=None, rule3 = None)
# ts4.inference()

# ts1 = DBP_Part_3('ja_en', None, equal_rule= True, align_rule=None, rule3 = None)
# ts1.inference()

# ts2 = DBP_Part_3('ja_en', None, equal_rule= None, align_rule=True, rule3 = None)
# ts2.inference()

# ts3 = DBP_Part_3('ja_en', None, equal_rule= True, align_rule=True, rule3 = None)
# ts3.inference()

# ts5 = DBP_Part_3('ja_en', None, equal_rule= True, align_rule=True, rule3 = True)
# ts5.inference()


# fr-en
# ts4 = DBP_Part_3('fr_en', None, equal_rule= None, align_rule=None, rule3 = None)
# ts4.inference()

# ts1 = DBP_Part_3('fr_en', None, equal_rule= True, align_rule=None, rule3 = None)
# ts1.inference()

# ts2 = DBP_Part_3('fr_en', None, equal_rule= None, align_rule=True, rule3 = None)
# ts2.inference()

# ts3 = DBP_Part_3('fr_en', None, equal_rule= True, align_rule=True, rule3 = None)
# ts3.inference()

# ts5 = DBP_Part_3('fr_en', None, equal_rule= True, align_rule=True, rule3 = True)
# ts5.inference()
import json
import os
from lib2to3.pgen2.grammar import opmap_raw
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
import logging

# load similar_matrix
# load original text
# generate top-k file for each source_entity

class Retrivel():
    def __init__(self, dataset, ratio=0.2, attr_opt=True, imagecaption_opt = False, name_opt=False, Rel_opt=False):
        super().__init__()
        self.dataset = dataset
        self.ratio = ratio
        self.id_Attr = self.attr_load()
        self.id_Name = self.name_load()
        self.id_Rel = self.rel_load()

    def attr_load(self):
        if 'DB' in self.dataset:
            filename = 'DB_FB_Attr_short_dsv3_v2.txt'
        else:
            filename = 'YG_FB_Attr_short_dsv3_v2.txt'
        path = '../data/MMKG/Attr_text'
        id_attr = {}
        with open(os.path.join(path,filename), 'r') as f:
            for i, line in enumerate(f):
                line_ = line.strip()
                attr_item_list = line_.split('^^^')
                new_attr = []
                for it in attr_item_list:
                    if '|||' in it:
                        new_attr.append(it)
                if len(new_attr) == 0:
                    id_attr[i] = ''
                else:
                    id_attr[i] = '^^^'.join(new_attr)
        return id_attr

    def rel_filter(self, text):
        result = []
        relation_part = text.split("Relation: ")[-1]
        relations = [r.strip() for r in relation_part.split(",") if r.strip()]
        for relation in relations:
            parts = relation.split("--")
            if len(parts) == 2:
                predicate = parts[0].split("/")[-1].replace(">", "").replace("_", ":")
                obj = parts[1].split("/")[-1].replace("<", "").replace(">", "").strip()
                result.append(f"{predicate}--{obj}")
        # filted_rel = " ".join(result)
        return result

    def rel_load(self):
        if 'DB' in self.dataset:
            filename = 'DB_FB_Rel_Rel_v2.txt'
        else:
            filename = 'YG_FB_Rel_Rel_v2.txt'
        path = '../data/MMKG/Rel-text'
        id_rel = {}
        with open(os.path.join(path, filename), 'r') as f:
            for i, line in enumerate(f):
                line_ = line.strip()
                try:
                    id_rel[i] = self.rel_filter(line_)
                except:
                    print('mm')

        return id_rel

    def name_load(self):
        path = '../data/MMKG/MMEA_name'
        if 'DB' in self.dataset:
            f1_name = 'DB_name.txt'
            f2_name = 'FB_DB_name.txt'
        else:
            f1_name = 'YAGO_name.txt'
            f2_name = 'FB_YAGO_name.txt'
        id_name = {}
        with open(os.path.join(path, f1_name), 'r') as f:
            for line in f:
                id_, name = line.strip().split(' ')
                id_name[int(id_)] = name[1:-1]

        with open(os.path.join(path, f2_name), 'r') as f:
            for line in f:
                id_, name = line.strip().split(' ')
                id_name[int(id_)] = name[:-1]
        return id_name


    def Bidirect_reranker_data_make_ND(self, ratio=None, side=None):
        def seed_load(dataset):
            filepath = f'../data/MMKG/seed'
            filename = os.path.join(filepath, dataset, 'ref_ent_ids')
            seed_pair = {}
            with open(filename, 'r') as f:
                for line in f:
                    try:
                        e1, e2 = line.split()
                        seed_pair[int(e1)] = int(e2)
                    except:
                        continue
            return seed_pair

        folder_path = '../data/MMKG/topk_id_score'
        k = 10
        if side is not None:
            file_name = os.path.join(folder_path,
                                    f'{self.dataset}_ratio_{ratio}_top{k}_Fuse_Bidirect_{side}.npy')
        else:
            file_name = os.path.join(folder_path, f'{self.dataset}_ratio_{ratio}_top{k}_Fuse_Bidirect.npy')
        seed_pair_dir = seed_load(self.dataset)
        print(len(seed_pair_dir))
        topk_matrix = np.load(file_name)
        original_m_list = ['A', 'N']

        json_list = []
        easy_list= []
        len_analyse = 0 
        max_len = 0

        for e1 in seed_pair_dir.keys():
            e2 = seed_pair_dir[e1]
            topk = list(topk_matrix[e1])
            if e2 not in topk:
                continue

            self_dir = {}
            k1id_name = self.id_Name[e1] if 'N' in original_m_list else None
            self_dir['N'] = k1id_name
            k1id_attr = self.id_Attr[e1] if 'A' in original_m_list else None
            self_dir['A'] = k1id_attr

            pos_dir = {}
            k2id_name = self.id_Name[e2] if 'N' in original_m_list else None
            pos_dir['N'] = k2id_name
            k2id_attr = self.id_Attr[e2] if 'A' in original_m_list else None
            pos_dir['A'] = k2id_attr

            data_dic = {}
            neg_dir = {}
            for i in range(0, len(topk)):
                if topk[i] != e2:
                    topi_attr = self.id_Attr[topk[i]] if 'A' in original_m_list else None
                    topi_name = self.id_Name[topk[i]] if 'N' in original_m_list else None
                    neg_dir[topk[i]] = {'N': topi_name, 'A': topi_attr}

            query = str(self_dir)
            pos = [str(pos_dir)]
            neg = [str(neg_dir[x]) for x in neg_dir.keys()]

            if len(neg) == 0:
                continue

            data_dic = {
                "query": query,
                "pos": pos,
                "neg": neg
            }

            json_list.append(data_dic)
            if topk[0]==e2:
                easy_list.append(data_dic)
            len_analyse += (len(query)+len(pos)+len(neg))
            if (len(query)+len(pos)+len(neg))> max_len:
                max_len = (len(query)+len(pos)+len(neg))
        
        print(len(json_list))
        print(len(easy_list))
        print(len_analyse/len(json_list))
        print(max_len)

        if side is None:
            easy_json_path = f"../data/MMKG/Rerank_data/{self.dataset}_{len(seed_pair_dir)}_easy" 
            combined_json_path = f"../data/MMKG/Rerank_data/{self.dataset}_{len(seed_pair_dir)}_easy_complex" 

        else:
            easy_json_path = f"../data/MMKG/Rerank_data/{self.dataset}_{len(seed_pair_dir)}_Name_easy" 
            combined_json_path = f"../data/MMKG/Rerank_data/{self.dataset}_{len(seed_pair_dir)}_Name_easy_complex" 
        
        easy_json_path += '_kv'
        combined_json_path += '_kv'
        
        easy_json_path += '_cpl'
        easy_json_path += '.json'

        combined_json_path += '_cpl'
        combined_json_path += '.json'

        with open(combined_json_path, "w", encoding="utf-8") as f:
            json.dump(json_list, f, ensure_ascii=False, indent=4)
        with open(easy_json_path, "w", encoding="utf-8") as f:
            json.dump(easy_list, f, ensure_ascii=False, indent=4)
        print(f"JSON 文件已生成：{combined_json_path}")
        print(f"JSON 文件已生成：{easy_json_path}")

    def Reranker_test_batch(self, ratio, side, ranker_name):
        def dev_dic_load():
            filepath = f'../data/MMKG/seed{ratio}/'
            filename = os.path.join(filepath, self.dataset.replace('_','-'), 'dev_ent_ids')
            dev_dic = {}
            with open(filename, 'r') as f:
                for line in f:
                    try:
                        e1, e2 = line.split()
                        dev_dic[int(e1)] = int(e2)
                    except:
                        continue
            return dev_dic
        dev_pair = dev_dic_load()
        folder_path = '../data/MMKG/topk_id_score'
        k = 10
        if side is not None:
            file_name = os.path.join(folder_path,
                                     f'{self.dataset}_ratio_{ratio}_top{k}_Fuse_Bidirect_{side}.npy')
        else:
            file_name = os.path.join(folder_path, f'{self.dataset}_ratio_{ratio}_top{k}_Fuse_Bidirect.npy')
        topk_matrix = np.load(file_name)
        original_m_list = ['A', 'N']  

        ranker_path = f'{ranker_name}'
        from FlagEmbedding import FlagReranker
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        reranker = FlagReranker(ranker_path, use_fp16=True, batch_size = 1500, device=device)
        logging.getLogger('FlagReranker').setLevel(logging.ERROR)
        print(f"reranker loading:{ranker_path}")

        hit1 = 0
        top1_hit1 = 0
        hit10 = 0
        easy_count = 0

        # 准备批量处理数据
        all_queries = []
        all_candidates = []
        all_topk = []
        all_e2 = []

        # print("准备数据...")

        e1_list = list(dev_pair.keys())
        e2_set = set(dev_pair.values())
        top1_e2_set = set(dev_pair.values())

        Kong_attr_ent_list = []

        print(len(e1_list))

        for e1 in tqdm(dev_pair.keys()):
            e2 = dev_pair[e1]
            topk = list(topk_matrix[e1])

            print(topk)
            # 查询
            self_dir = {}
            # if side is not None:
            k1id_name = self.id_Name[e1] if 'N' in original_m_list else None
            self_dir['N'] = k1id_name
            k1id_attr = self.id_Attr[e1] if 'A' in original_m_list else None
            self_dir['A'] = k1id_attr
            if k1id_attr == '' and side is None:
                Kong_attr_ent_list.append(e1)
                continue
            query_str = str(self_dir)
            
            # 候选 
            candi_strs = []
            for candidate in topk:
                candi_dir = {}
                candi_attr = self.id_Attr[candidate] if 'A' in original_m_list else None
                candi_name = self.id_Name[candidate] if 'N' in original_m_list else None
                
                # if side is not None:
                candi_dir[candidate] = {'N': candi_name, 'A': candi_attr}
                candi_strs.append(str(candi_dir[candidate]))

            Flag = True
            while len(candi_strs) < 10:
                if len(candi_strs) == 0:
                    Flag = False
                    break
                else:
                    candi_strs.append(candi_strs[-1])
            
            if not Flag:
                Kong_attr_ent_list.append(e1)
                continue
            
            # 存储批量处理数据
            for i in range(len(candi_strs)):
                all_queries.append(query_str)
            all_candidates.extend(candi_strs)
            all_topk.extend(topk)
            all_e2.append(e2)

        # 批量计算所有分数
        all_scores = []
        batch_size = 100000  # 根据内存调整批次大小

        for i in tqdm(range(0, len(all_queries), batch_size)):
            batch_queries = all_queries[i:i+batch_size]
            batch_candidates = all_candidates[i:i+batch_size]
            batch_pairs = []
            batch_pairs = [[q, doc] for q, doc in zip(batch_queries, batch_candidates)]
            batch_scores = reranker.compute_score(batch_pairs)
            all_scores.extend(batch_scores)
        
        print(f"len(all_scores):{len(all_scores)}")

        Imcomplete_data_save_path = f'../data/MMKG/IC_data/seed{ratio}/'
        if not os.path.exists(Imcomplete_data_save_path):
            os.makedirs(Imcomplete_data_save_path)

        Ic_e1_list = []

        for idx in range(0, len(all_scores), len(topk)):
            batch_scores = all_scores[idx:idx+len(topk)]
            if max(batch_scores) > 0:
                easy_count += 1
                if all_topk[idx+np.argmax(batch_scores)] == all_e2[idx//len(topk)]:
                    hit1 += 1
                    e2_set.discard(all_e2[idx//len(topk)])
                if all_topk[idx] == all_e2[idx//len(topk)]:
                    top1_hit1 += 1
                    top1_e2_set.discard(all_e2[idx//len(topk)])
                if all_e2[idx//len(topk)] in all_topk[idx:idx+len(topk)][:10]:
                    hit10 += 1 
            else:
                Ic_e1_list.append(e1_list[idx//len(topk)])

        Ic_e2_list = list(e2_set)
        top_Ic_en_list = list(top1_e2_set)
        Ic_e1_list.extend(Kong_attr_ent_list)
        
        total_pairs = len(dev_pair)
        print(f"dev_hit@1:{hit1},  {hit1}/{total_pairs} = {hit1/total_pairs:.4f}")
        print(f"easy_dev_hit@10:{hit10},  {hit10}/{total_pairs} = {hit10/total_pairs:.4f}")
        print(f"easy_count:{easy_count},  {easy_count}/{total_pairs} = {easy_count/total_pairs:.4f}")

        # 保存不完整数据
        Ic_e1_save_path = os.path.join(Imcomplete_data_save_path, f'{self.dataset}_{side}_Ic_e1.txt')
        with open(Ic_e1_save_path, 'w') as f:
            for ic in Ic_e1_list:
                f.write(f"{ic}\n")
        print(f"不完整数据已保存到 {Ic_e1_save_path}, 共 {len(Ic_e1_list)} 条记录。")

        Ic_e2_save_path = os.path.join(Imcomplete_data_save_path, f'{self.dataset}_{side}_Ic_e2.txt')
        with open(Ic_e2_save_path, 'w') as f:
            for ic in Ic_e2_list:
                f.write(f"{ic}\n")
        print(f"不完整数据已保存到 {Ic_e2_save_path}, 共 {len(Ic_e2_list)} 条记录。")

        Top_Ic_e2_save_path = os.path.join(Imcomplete_data_save_path, f'{self.dataset}_{side}_top_Ic_e2.txt')
        with open(Top_Ic_e2_save_path, 'w') as f:
            for ic in top_Ic_en_list:
                f.write(f"{ic}\n")
        print(f"不完整数据已保存到 {Top_Ic_e2_save_path}, 共 {len(top_Ic_en_list)} 条记录。")

        with open('../Results/MMKG.txt', 'a') as f:
            f.write("\n\n")
            f.write(f"Reranker_test_batch result of {self.dataset}, ratio:{ratio}, side:{side}, ranker:{ranker_name}\n")
            f.write(f"Ranker_name:{ranker_name}\n")
            f.write(f"dev_hit@1:{hit1},  {hit1}/{total_pairs} = {hit1/total_pairs:.4f}\n")
            f.write(f"dev_top_hit@1:{top1_hit1},  {top1_hit1}/{total_pairs} = {top1_hit1/total_pairs:.4f}\n")

            f.write(f"easy_dev_hit@10:{hit10},  {hit10}/{total_pairs} = {hit10/total_pairs:.4f}\n")
            f.write(f"easy_count:{easy_count},  {easy_count}/{total_pairs} = {easy_count/total_pairs:.4f}\n")
            f.write(f"Incomplete data e1 count: {len(Ic_e1_list)}\n")
            f.write(f"Incomplete data e2 count: {len(Ic_e2_list)}   \n")    
                

    def Reranker_test(self, ratio, side, ranker_name):
        def dev_dic_load():
            filepath = f'../data/MMKG/seed{ratio}/'
            filename = os.path.join(filepath, self.dataset.replace('_','-'), 'dev_ent_ids')
            dev_dic = {}
            with open(filename, 'r') as f:
                for line in f:
                    try:
                        e1, e2 = line.split()
                        dev_dic[int(e1)] = int(e2)
                    except:
                        continue
            return dev_dic
        dev_pair = dev_dic_load()
        folder_path = '../data/MMKG/topk_id_score'
        folder_path = '../data/MMKG/topk_id_score'
        k = 10
        if side is not None:
            file_name = os.path.join(folder_path,
                                     f'{self.dataset}_ratio_{ratio}_top{k}_Fuse_Bidirect_{side}.npy')
        else:
            file_name = os.path.join(folder_path, f'{self.dataset}_ratio_{ratio}_top{k}_Fuse_Bidirect.npy')
        topk_matrix = np.load(file_name)
            
        if side is not None:
            original_m_list = ['A', 'N']  # orignal_information 参考哪些模态
        else:
            original_m_list = ['A']

        ranker_path = f'{ranker_name}'
        from FlagEmbedding import FlagReranker
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        reranker = FlagReranker(ranker_path, use_fp16=True, batch_size = 1500, device=device)
        logging.getLogger('FlagReranker').setLevel(logging.ERROR)
        print(f"reranker loading:{ranker_path}")

        hit1 = 0
        top1_hit1 = 0
        hit10 = 0
        easy_count = 0
        for e1 in tqdm(dev_pair.keys()):
            if e1 != 2654:
                continue
            e2 = dev_pair[e1]
            topk = list(topk_matrix[e1])
            print(topk)

            self_dir = {}
            # if side is not None:
            k1id_name = self.id_Name[e1] if 'N' in original_m_list else None
            self_dir['Name'] = k1id_name
            k1id_attr = self.id_Attr[e1] if 'A' in original_m_list else None
            self_dir['Attribute'] = k1id_attr
            if k1id_attr == '' and side is None:
                continue
            query_str = str(self_dir)
            
            # 候选 
            candi_strs = []
            for candidate in topk:
                candi_dir = {}
                candi_attr = self.id_Attr[candidate] if 'A' in original_m_list else None
                candi_name = self.id_Name[candidate] if 'N' in original_m_list else None
                
                # if side is not None:
                candi_dir[candidate] = {'Name': candi_name, 'Attribute': candi_attr}
                # else:
                    # candi_dir[candidate] = {'Attribute': candi_attr}

                candi_strs.append(str(candi_dir[candidate]))

            query = [str(query_str)] * len(candi_strs)
            documents = [str(x) for x in candi_strs]


            pairs = [[q, doc] for q, doc in zip(query, documents)]
            print(documents)
            scores = reranker.compute_score(pairs)
            print(scores)
            if max(scores) > 0:
                easy_count += 1
                if topk[np.argmax(scores)] == e2:
                    hit1 += 1
                if topk[0]==e2:
                    top1_hit1 += 1
                if e2 in topk:
                    hit10 += 1

        total_pairs = len(dev_pair)
        print(f"dev_hit@1:{hit1},  {hit1}/{total_pairs} = {hit1/total_pairs:.4f}")
        print(f"dev_hit@1:{top1_hit1},  {top1_hit1}/{total_pairs} = {hit1/total_pairs:.4f}")
        print(f"easy_dev_hit@10:{hit10},  {hit10}/{total_pairs} = {hit10/total_pairs:.4f}")
        print(f"easy_count:{easy_count},  {easy_count}/{total_pairs} = {easy_count/total_pairs:.4f}")

        with open('../Results/MMKG.txt', 'a') as f:
            f.write("\n\n")
            f.write(f"Reranker_test_result of {self.dataset}, ratio:{ratio}, side:{side}, ranker:{ranker_name}\n")
            f.write(f"Ranker_name:{ranker_name}\n")
            f.write(f"dev_hit@1:{hit1},  {hit1}/{total_pairs} = {hit1/total_pairs:.4f}\n")
            f.write(f"dev_top_hit@1:{top1_hit1},  {top1_hit1}/{total_pairs} = {top1_hit1/total_pairs:.4f}\n")

            f.write(f"easy_dev_hit@10:{hit10},  {hit10}/{total_pairs} = {hit10/total_pairs:.4f}\n")
            f.write(f"easy_count:{easy_count},  {easy_count}/{total_pairs} = {easy_count/total_pairs:.4f}\n")


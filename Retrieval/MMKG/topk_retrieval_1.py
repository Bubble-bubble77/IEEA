import os
import numpy as np


class Retrieval():
    def __init__(self, dataset, ratio=0.2):
        super().__init__()
        self.dataset = dataset
        self.ratio = ratio

    
    def Bi_direct_retrieval2(self, k, ratio, side=None):
        def dev_pair_load():
            filepath = f'../data/MMKG/seed{ratio}'
            filename = os.path.join(filepath, self.dataset.replace('_','-'), 'dev_ent_ids')
            dev_pair = []
            with open(filename, 'r') as f:
                for line in f:
                    try:
                        e1, e2 = line.split()
                        dev_pair.append((int(e1), int(e2)))
                    except:
                        continue
            return dev_pair

        dev_pair = dev_pair_load()

        if side is not None:
                file_path = f'Fuse_{self.dataset}_ratio{ratio}_total_ST_N.npy'
        else:
            file_path = f'Fuse_{self.dataset}_ratio{ratio}_total_ST.npy'
        print(file_path)

        moda_matrix = np.load(file_path, allow_pickle=True)
        es_score = (moda_matrix - moda_matrix.min()) / (moda_matrix.max() - moda_matrix.min())

        # 纵向检索
        x_topk_indices = np.argsort(-es_score, axis=1)[:, :k]
        x_topk_values = np.zeros((es_score.shape[0], k))
        for i in range(es_score.shape[0]):
            x_topk_values[i] = es_score[i, x_topk_indices[i]]

        original_entity_alignment = x_topk_indices[:, :10] + x_topk_indices.shape[0]

        yk = 30
        y_topk_indices = np.argsort(-es_score, axis=0)[:yk, :]
        y_topk_indices_T = y_topk_indices.T

        buchong = []
        x_topk = x_topk_indices.copy()
        x_topk2 = x_topk_indices.copy()
        flat_indices = x_topk2.flatten()
        from collections import Counter
        frequency_counter = Counter(flat_indices)


        for i in range(x_topk.shape[0]):
            bu = set()
            rows, _ = np.where(y_topk_indices_T == i)
            for row in rows:
                bu.add((row, es_score[i][row]))
            bu = list(bu)
            bu = sorted(bu, key=lambda x: x[1], reverse=True)
            buchong.append(bu[:min(20, len(bu))])

        import math

        entity_alignment_matrix = np.zeros((x_topk.shape[0], k), dtype=int)
        for i in range(entity_alignment_matrix.shape[0]):
            ori_ents = dict()
            for rank, e2 in enumerate(x_topk[i]):
                if e2 != -1:
                    try:
                        x_value = x_topk_values[i][rank]
                        x_value = x_value * (1 - math.log10(frequency_counter[e2]) / (math.log10(len(frequency_counter))))
                        ori_ents[e2] = x_value / math.exp(rank+1)
                    except:
                        print(x_topk_values[i])
                        print(x_topk_values[i][rank]/(rank+1))
                        exit(0)

            if i < len(buchong) and buchong[i]:
                for rank, (a, b) in enumerate(buchong[i]):
                    b_value = b
                    if a in frequency_counter.keys():
                        b_value = b_value * (1 - math.log10(frequency_counter[a])/math.log10(len(frequency_counter)))
                    if a not in ori_ents.keys():
                        try:
                            ori_ents[a] = b_value / math.exp(rank+1)
                        except:
                            print(rank)
                            print((a,b))
                            exit(0)
                    else:
                        ori_ents[a] = ori_ents[e2] + b_value / math.exp(rank+1)
            ori_ents_ls = []
            for e, sc in ori_ents.items():
                ori_ents_ls.append((e, sc))

            filtered_list = sorted(list(ori_ents_ls), key=lambda x: x[1], reverse=True)
            top_10_list = [e[0] for e in filtered_list]

            top_k = np.array(top_10_list)[:k]
            entity_alignment_matrix[i][:len(top_k)] = top_k

        entity_alignment_matrix = entity_alignment_matrix + entity_alignment_matrix.shape[0]


        hit1 = 0
        hit10 = 0

        o_hit1 = 0
        o_hit10 = 0

        in_yindix = 0
        in_xindix = 0


        for e1, e2 in dev_pair:
            if e2 == entity_alignment_matrix[e1][0]:
                hit1 += 1
            if e2 in entity_alignment_matrix[e1]:
                hit10 += 1
            else:
                if e2 in x_topk_indices[e1] + x_topk_indices.shape[0]:
                    in_xindix += 1
                if e1 in y_topk_indices_T[e2 - x_topk_indices.shape[0]]:
                    in_yindix += 1

            if e2 == original_entity_alignment[e1][0]:
                o_hit1 += 1
            if e2 in original_entity_alignment[e1]:
                o_hit10 += 1

        print(f'o_hit1:{o_hit1 / len(dev_pair)}')
        print(f'o_hit10:{o_hit10 / len(dev_pair)}')

        print(f'hit1:{hit1 / len(dev_pair)}')
        print(f'hit10:{hit10 / len(dev_pair)}')

        print(f'in_xindix:{in_xindix}')
        print(f'in_yindix:{in_yindix}')


        folder_path = '../data/MMKG/topk_id_score'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if side is not None:
            file_name = os.path.join(folder_path,
                                     f'{self.dataset}_ratio_{ratio}_top{k}_Fuse_Bidirect_{side}.npy')
        else:
            file_name = os.path.join(folder_path, f'{self.dataset}_ratio_{ratio}_top{k}_Fuse_Bidirect.npy')
        
        print(entity_alignment_matrix.shape)
        print(file_name)
        np.save(file_name, entity_alignment_matrix)



re =  Retrieval('FB15K_DB15K')
re.Bi_direct_retrieval2(10, 0.2, 'N')
re.Bi_direct_retrieval2(10, 0.2, None)
re.Bi_direct_retrieval2(10, 0.5, 'N')
re.Bi_direct_retrieval2(10, 0.5, None)
re.Bi_direct_retrieval2(10, 0.8, 'N')
re.Bi_direct_retrieval2(10, 0.8, None)

print('FB_DB end!')

re1 =  Retrieval('FB15K_YAGO15K')
re1.Bi_direct_retrieval2(10, 0.2, 'N')
re1.Bi_direct_retrieval2(10, 0.2, None)
re1.Bi_direct_retrieval2(10, 0.5, 'N')
re1.Bi_direct_retrieval2(10, 0.5, None)
re1.Bi_direct_retrieval2(10, 0.8, 'N')
re1.Bi_direct_retrieval2(10, 0.8, None)
print('FB_YAGO end!')

import json
import os
from lib2to3.pgen2.grammar import opmap_raw
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
import random

random.seed(42)

class Retrivel():
    def __init__(self, dataset, ratio=0.2):
        super().__init__()
        self.dataset = dataset
        self.ratio = ratio
        self.index_1, self.index_2 = self.load_index(dataset)
        self.id_Attr = self.attr_load()
        self.id_Name = self.name_load()
        self.id_o_Name = self.orilang_name_load()
        self.original_attr = self.original_attr_load()

        self.id_Imagecaption = None


    def attr_load(self):
        id_attr = {}
        or_dataset = self.dataset.split('_')[0]
        with open(f'../data/DBP15K/attr_trans/{self.dataset}_id_att_1', 'r') as f:
            for line in f:
                id_1, att_list = line.strip().split('\t')
                att_list_ = att_list.split(' ')
                new_list = []
                for al in att_list_:
                    new_list.append(al.replace(f'http://{or_dataset}.dbpedia.org/property/', '').replace(
                        'http://dbpedia.org/property/', ''))
                id_attr[int(id_1)] = new_list
        with open(f'../data/DBP15K/attr_trans/{self.dataset}_id_att_2', 'r') as f:
            for line in f:
                id_2, att_list = line.strip().split('\t')
                att_list_ = att_list.split(' ')
                new_list = []
                for al in att_list_:
                    new_list.append(al.replace('http://dbpedia.org/property/', ''))
                id_attr[int(id_2)] = new_list
        return id_attr

    def original_attr_load(self):
        id_attr = {}
        s_d, t_d = self.dataset.split('_')
        attr_path_1 = f"../data/DBP15K/dbpdata/DBP15k/{self.dataset}/{s_d}_att_triples"
        attr_path_2 = f"../data/DBP15K/dbpdata/DBP15k/{self.dataset}/{t_d}_att_triples"
        with open(attr_path_1, 'r') as f:
            for line in f:
                attr_list = line.strip().split(' ')
                ent_n = attr_list[0].split('/')[-1][:-1]
                att = attr_list[1].split('/')[-1][:-1]
                att_v = attr_list[2].split('/')[-1]

                if ent_n not in id_attr.keys():
                    id_attr[ent_n] = [f'{att}--{att_v}']
                else:
                    id_attr[ent_n].append(f'{att}--{att_v}')

        with open(attr_path_2, 'r') as f:
            for line in f:
                attr_list = line.strip().split(' ')
                ent_n = attr_list[0].split('/')[-1][:-1]
                att = attr_list[1].split('/')[-1][:-1]
                att_v = attr_list[2].split('/')[-1]

                if ent_n not in id_attr.keys():
                    id_attr[ent_n] = [f'{att}--{att_v}']
                else:
                    id_attr[ent_n].append(f'{att}--{att_v}')
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
        path = '../data/MMEA-data/Rel-text'
        id_rel = {}
        with open(os.path.join(path, filename), 'r') as f:
            for i, line in enumerate(f):
                line_ = line.strip()
                try:
                    id_rel[i] = self.rel_filter(line_)
                except:
                    print('mm')

        return id_rel

    def load_index(self, dataset):
        # load_index1
        index_1 = {}
        with open('../data/DBP15K/DBP_1/{}/index_1'.format(dataset), 'r') as f:
            for line in f:
                line = line.strip()
                ent_id, ent_index = line.split('\t')
                index_1[int(ent_id)] = int(ent_index)
        # load_index2
        index_2 = {}
        with open('../data/DBP15K/DBP_1/{}/index_2'.format(dataset), 'r') as f:
            for line in f:
                line = line.strip()
                ent_id, ent_index = line.split('\t')
                index_2[int(ent_id)] = int(ent_index)
        return index_1, index_2

    def name_load(self):
        json_path = r'../data/DBP15K/translated_ent_name/dbp_{}.json'.format(self.dataset)
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        id_name = {}
        for i in range(len(data)):
            id_name[data[i][0]] = ' '.join(data[i][1])

        new_name_dir = {}
        for k in id_name.keys():
            if int(k) in self.index_1.keys():
                new_k = self.index_1[int(k)]
            elif int(k) in self.index_2:
                new_k = self.index_2[int(k)]
            else:
                print('error')
            new_name_dir[int(new_k)] = id_name[k]
        return new_name_dir

    def orilang_name_load(self):
        id_name = {}
        source_lang = self.dataset.split('_')[0]
        with open(f'../data//DBP15K/DBP_1/{self.dataset}/ent_ids_1', 'r') as f:
            for line in f:
                id_1, ol_name = line.strip().split('\t')
                id_name[int(id_1)] = ol_name.replace(f'http://{source_lang}.dbpedia.org/resource/', '')
        with open(f'../data//DBP15K/DBP_1/{self.dataset}/ent_ids_2', 'r') as f:
            for line in f:
                id_1, ol_name = line.strip().split('\t')
                id_name[int(id_1)] = ol_name.replace(f'http://dbpedia.org/resource/', '')
        return id_name
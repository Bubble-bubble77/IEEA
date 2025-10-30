
import os

class Retrivel():
    def __init__(self, dataset, ratio=0.2):
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

                # id_attr[i] = line_
                
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



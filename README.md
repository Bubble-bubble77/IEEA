# IEEA
The source code of IEEA framework.

🔧 Overview

The system consists of four main components:

1. MMKG Embedding Preparation
2. Retrieval Module
3. Identifier Fine-Tuning
4. Information Expansion & LLM-Based Reasoning

📦 Installation

bash
git clone
cd IEEA
pip install -r requirements.txt

🔗 Data Download

The **MMEA-data** and **ECS-results** can be downloaded from  [GoogleDrive](https://drive.google.com/drive/folders/1wfErYdAV93yxPtPHqkGanbmb_Ztv-LRU?usp=drive_link).
The original MMEA dataset can be downloaded from  [MMKB](https://github.com/mniepert/mmkb) and [MMEA](https://github.com/lzxlin/mclea?tab=readme-ov-file).

Finetuned Identifier can be downloaded from [ModelScope](https://www.modelscope.cn/collections/ICEA-88f1bd52936f4e) ;
The dataset for finetuning can be downloaded from [DBP15K](https://www.modelscope.cn/datasets/hahawang111/DBP15K) [MMKG](https://www.modelscope.cn/datasets/hahawang111/MMKG_information_insufficiency_identifier_finetune_dataset)



🗂️ Project Structure
```
IEEA/
├── data/
│ ├── DBP15K/
│ └── MMKG/
│
├── Identifier/
│ ├── DBP15K/
│ │ ├── generate_test.py
│ │ ├── identifier_util.py
│ │ └── train.sh
│ └── MMKG/
│ ├── generate_test.py
│ ├── identifier_util.py
│ └── train.sh
│
├── LLMReason/
│ ├── A_DBP_part3.py
│ ├── A_MMKG_part3_1.py # Reasoning script for MMKG
│ ├── DBP_retrieval.py
│ ├── MMKG_retrieval.py # Retrieval + prompt construction for MMKG
│ └── prompt.py # Prompt templates and formatting utilities
│
├── Prepare_EMB/ # MMKG embedding preparation (EIEA-inspired)
│ ├── DBP15K/
│ └── MMKG/
│ ├── count.py
│ ├── dataset.py
│ ├── gcn_layer.py
│ ├── model.py
│ ├── run.py
│ └── run.sh
│
└── Retrieval/ # Retrieval module
├── DBP/
│ ├── run.sh # Run retrieval for DBP15K
│ └── topk_retrieval.py
└── MMKG/
└── ... # (likely similar structure for MMKG retrieval)
```

🌐 1. MMKG Embedding Preparation

We follow the [EIEA_code](https://github.com/Bubble-bubble77/EIEA) [EIEA_paper](https://dl.acm.org/doi/10.1145/3711896.3736948) framework to generate unified embeddings for entities in multi-modal knowledge graphs.

🔍 2. Retrieval Module
Output: List of candidate entities ranked by relevance.

🔤 3. Identifier Fine-Tuning

Fine-tunes a lightweight neural classifier (e.g., BGE-Reranker-m3) to identify the information-insufficient entity from retrieved candidates.
Model outputs probability scores for each candidate. 

🤖 4. LLM-Reasoning Module


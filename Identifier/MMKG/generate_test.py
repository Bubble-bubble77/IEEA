from identifier_util import *

re = Retrivel('FB15K_DB15K')
re.Bidirect_reranker_data_make_ND(0.2,'N')
re.Bidirect_reranker_data_make_ND(0.2,None)

re1 = Retrivel('FB15K_YAGO15K')
re1.Bidirect_reranker_data_make_ND(0.2,None)
re1.Bidirect_reranker_data_make_ND(0.2,'N')


##
# use Train.sh to finetune the identifier model
##

# test the identifier with reranker
# re.Reranker_test_batch(0.2, 'N', '/data/Ranker/MMKG/FB_DB_N_1015_CL_2')
# re.Reranker_test_batch(0.5, 'N', '/data/Ranker/MMKG/FB_DB_N_1015_CL_2')
# re.Reranker_test_batch(0.8, 'N', '/data/Ranker/MMKG/FB_DB_N_1015_CL_2')
# re1.Reranker_test_batch(0.2, 'N', '/data/Ranker/MMKG/FB_YG_N_1015_CL_2')
# re1.Reranker_test_batch(0.5, 'N', '/data/Ranker/MMKG/FB_YG_N_1015_CL_2')
# re1.Reranker_test_batch(0.8, 'N', '/data/Ranker/MMKG/FB_YG_N_1015_CL_2')

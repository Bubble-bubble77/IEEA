from identifier_util import *

re = Retrivel('zh_en')
re.Bi_direct_fuse_rerank_check_data_make(None, o_name=False)
re.Bi_direct_fuse_rerank_check_data_make('N', o_name=False)

re1 = Retrivel('ja_en')
re1.Bi_direct_fuse_rerank_check_data_make(None, o_name=False)
re1.Bi_direct_fuse_rerank_check_data_make('N', o_name=False)

re2 = Retrivel('fr_en')
re2.Bi_direct_fuse_rerank_check_data_make(None, o_name=False)
re2.Bi_direct_fuse_rerank_check_data_make('N', o_name=False)

"""
Train Identifier

"""

# test
# re.Reranker_test_batch(None, '/zh_en_1013_CL_2', o_name=False)
# re.Reranker_test_batch('N', '/zh_en_N_on_1023_2', o_name=True)
# re1.Reranker_test_batch(None, '/ja_en_1014_CL_2', o_name=False)
# re1.Reranker_test_batch('N', '/ja_en_N_1016_CL_on_2', o_name=False)
# re2.Reranker_test_batch(None, '/fr_en_1014_CL_2', o_name=False)
# re2.Reranker_test_batch('N', '/fr_en_N_1016_CL_on_2', o_name=False)




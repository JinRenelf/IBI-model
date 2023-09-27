# -*- coding: utf-8 -*-

# @File    : cupy vs numpy vs pytorch.py
# @Date    : 2023-09-25
# @Author  : ${ RenJin}

import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def read_single_csv(input_path):
    df_chunk=pd.read_csv(input_path,chunksize=1000)
    res_chunk=[]
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df=pd.concat(res_chunk)
    return res_df

st=time.time()

# df_variants = pd.read_csv(os.path.join("..", "data", "chrm21__KidsFirst_snp01_dominant_withCorrect_Index_RR1.csv"),
#                           index_col=0)

df_variants=pd.read_pickle(os.path.join("..", "data", "chrm21__KidsFirst_snp01_dominant_withCorrect_Index_RR1.pkl.gzip"))

# read_single_csv(os.path.join("..", "data", "chrm21__KidsFirst_snp01_dominant_withCorrect_Index_RR1.csv"))
print("elapsed time:",time.time()-st)


df_variants.to_pickle(os.path.join("..", "data", "chrm21__KidsFirst_snp01_dominant_withCorrect_Index_RR1.pkl.gzip"))




# df_traits = pd.read_csv(os.path.join("..", "data", "Phenotype__KidsFirst_withCorrect_Index.csv"), index_col=0)
#
# df_traits.to
# variants = np.array(df_variants, dtype=np.int16)
# traits = np.array(df_traits, dtype=np.int16)
#
# cvariants = cp.asarray(variants)
# ctraits = cp.asarray(traits)
#
# n = 1000
# st = time.time()
# for i in range(n):
#     index1=variants[i]==1
#     v1=variants[:,index1]
#     d1=traits[index1]
#     res = v1 @ d1
# et = time.time()
# print("numpy elpased time:", et - st)
#
# st = time.time()
# for i in range(n):
#     index1=cvariants[i]==1
#     cv=cvariants[:,index1]
#     res = cv @ ctraits[index1]
# et = time.time()
# print("cupy elpased time:", et - st)
#
#
# def mm(a, b,i):
#     index1=a[i]==1
#     v=a[:,index1]
#     return v @ b[index1]
#
# st = time.time()
# element_run = Parallel(n_jobs=-1)(delayed(mm)(variants, traits,var) for var in range(n))
# et = time.time()
# print("numpy parallel elpased time:", et - st)

# import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tvariants=torch.tensor(variants,device=device,dtype=torch.float16)
# ttraits=torch.tensor(traits,device=device,dtype=torch.float16)
#
# st=time.time()
# for i in range(10):
#     res=tvariants@ttraits
# et=time.time()
# print("torch elpased time:",et-st)

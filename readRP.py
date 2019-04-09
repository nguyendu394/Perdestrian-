import pandas as pd
import os
import numpy as np

# print(files[0:5])
bbs_csv = 'mydata/rois_train_thr70.csv'
bbs = pd.read_csv(bbs_csv)
bbs.set_index('id', inplace=True)
aa = bbs.loc[46126].iloc[:3].reset_index().as_matrix()
# #
# tworois = [46126,46115]
# aa= np.concatenate((aa,aa),axis=0)
print(aa)
# min = 999
# idx = -1
# for i in range(50184):
#     if i not in tworois:
#         aa = bbs.loc[i].reset_index().as_matrix()
#         if aa.shape[0] <= min:
#             min = aa.shape[0]
#             idx = i
# print(min,idx)
# print(aa.shape[0])
# aa[:, 0] = aa[:, 0] - aa[0, 0]

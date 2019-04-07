import pandas as pd
import os

# print(files[0:5])
imgs_csv = 'mydata/rois_set00_v000.csv'
bbs = pd.read_csv(imgs_csv)
bbs.set_index('id', inplace=True)
aa = bbs.loc[[5,6]].reset_index().as_matrix()
#
aa[:, 0] = aa[:, 0] - aa[0, 0]
print(aa)
# aa[:, 0] = aa[:, 0] - aa[0, 0]
print(aa)

import os
import pandas as pd

data_root_dir = '/home/luben/CSIE5452_FinalProj/dataset/VPGNet-DB-5ch'
csv_save_dir = '/home/luben/CSIE5452_FinalProj/dataset'

listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(data_root_dir):
    rel_dirpath = dirpath.split(data_root_dir)[1] 
    listOfFiles += [os.path.join(rel_dirpath, file) for file in filenames]

df = pd.DataFrame(data=listOfFiles, columns=['mat_path'])
df.to_csv(os.path.join(csv_save_dir,'mat_paths.csv'),index=False)

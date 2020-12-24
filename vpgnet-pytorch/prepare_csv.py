import os
import pandas as pd

data_root_dir = '/home/master/09/luben3485/data/VPGNet-DB-5ch'
csv_save_dir = '/home/master/09/luben3485/data'

listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(data_root_dir):
    rel_dirpath = dirpath.split(data_root_dir)[1] 
    listOfFiles += [os.path.join(rel_dirpath, file) for file in filenames]

df = pd.DataFrame(data=listOfFiles[:100], columns=['mat_path'])
df.to_csv(os.path.join(csv_save_dir,'mat_paths.csv'),index=False)

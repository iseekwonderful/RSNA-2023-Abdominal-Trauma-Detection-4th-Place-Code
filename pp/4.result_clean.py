import os
import pandas as pd
from shutil import copyfile
import glob

path = os.path.dirname(os.path.realpath(__file__))

runs = glob.glob(path + '/../results/*')

for r in runs:
    # print(r)
    df = pd.read_csv(r + '/train.log', sep='\t')
    for f in df.Fold.unique():
        sub = df[df.Fold == f].copy()
        eph = int(sub.sort_values('score').iloc[0].Epochs)
        print(eph)
        copyfile(r + f'/checkpoints/f{f}_epoch-{eph}.pth', r + f'/f{f}_epoch-{eph}.pth')
    # print(df.head())
    # break
# print(runs)

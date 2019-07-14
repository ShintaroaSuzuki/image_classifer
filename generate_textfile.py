import os
import glob

folders = [r'.\data\_train', r'.\data\_test']

for folder in folders:
    with open(folder + r'\data.txt','w') as f:
        for i, dir in enumerate(glob.glob(folder + r'\*')):
            img_files = glob.glob(dir + r'\*')
            for file in img_files:
                f.write(file + ' ' + str(i - 1) + '\n')

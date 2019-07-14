import os
import shutil
import glob
import random
import re

pattern = r'(?<=\\).+(?=_cutted)'
project_dir = os.getcwd()
os.chdir('./data')
dirs = glob.glob('./*')
names = []
cutted_dirs = []
for dir in dirs:
    if '_cutted' in dir:
        cutted_dirs.append(dir)
        names.append(re.search(pattern, dir).group())

os.makedirs('./_train', exist_ok = True)
os.makedirs('./_test', exist_ok = True)

for i, name in enumerate(names):
    train_dir = './_train/' + name
    test_dir = './_test/' + name
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    cutted_files = glob.glob(cutted_dirs[i] + '/*')
    random.shuffle(cutted_files)
    train_rate = 0.9
    for i, file in enumerate(cutted_files):
        if i < len(cutted_files) * train_rate:
            shutil.move(file, train_dir)
        else:
            shutil.move(file, test_dir)

os.chdir(project_dir)

print('訓練データ、検証データに分類完了')
cutted_files

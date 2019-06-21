import os
import glob

def get_dir_list(folder):
    dirs = glob.glob('./*')
    return dirs

def generate_textfile(folder):
    project_dir = os.getcwd()
    os.chdir('./data/' + folder)
    f = open('data.txt','w')
    for i, dir in enumerate(get_dir_list(folder)):
        img_files = dirs = glob.glob(dir + '/*')
        for file in img_files:
            path_list = file.split('\\')[1:]
            path = '\\'.join(path_list)
            f.write('.\\data\\' + folder + '\\' + path + ' ' + str(i - 1) + '\n')
    f.close()
    os.chdir(project_dir)

if __name__ == '__main__':
    folders = ['_train', '_test']
    for folder in folders:
        generate_textfile(folder)

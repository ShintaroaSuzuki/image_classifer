import os
import glob

data_folder = 'C:\\Users\\shinb\\OneDrive\\ドキュメント\\Python Scripts\\data_image_classifer'

def get_dir_list(folder):
    dirs = glob.glob(data_folder + './*')
    return dirs

def generate_textfile(folder):
    project_dir = os.getcwd()
    os.chdir(data_folder + '/' + folder)
    f = open('data.txt','w')
    for i, dir in enumerate(get_dir_list(folder)):
        img_files = glob.glob(dir + '/*')
        for file in img_files:
            path = '\\'.join(file.split('\\')[1:])
            f.write(data_folder + '\\' + folder + '\\' + path + ' ' + str(i - 1) + '\n')
    f.close()
    os.chdir(project_dir)

if __name__ == '__main__':
    folders = ['_train', '_test']
    for folder in folders:
        generate_textfile(folder)

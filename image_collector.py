import json
from urllib import parse
import requests
from bs4 import BeautifulSoup
from PIL import Image
import unicodedata
import os

data_folder = 'C:\\Users\\shinb\\OneDrive\\ドキュメント\\Python Scripts\\data_image_classifer'

class Google:
    def __init__(self):
        self.GOOGLE_SEARCH_URL = 'https://www.google.co.jp/search'
        self.session = requests.session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'})

    def Search(self, keyword, type='text', maximum=100):
        '''Google検索'''
        print('Google', type.capitalize(), 'Search :', keyword)
        result, total = [], 0
        query = self.query_gen(keyword, type)
        while True:
            # 検索
            html = self.session.get(next(query)).text
            links = self.get_links(html, type)

            # 検索結果の追加
            if not len(links):
                print('-> No more links')
                break
            elif len(links) > maximum - total:
                result += links[:maximum - total]
                break
            else:
                result += links
                total += len(links)
        return result

    def query_gen(self, keyword, type):
        '''検索クエリジェネレータ'''
        page = 0
        while True:
            if type == 'text':
                params = parse.urlencode({
                    'q': keyword,
                    'num': '100',
                    'filter': '0',
                    'start': str(page * 100)})
            elif type == 'image':
                params = parse.urlencode({
                    'q': keyword,
                    'tbm': 'isch',
                    'filter': '0',
                    'ijn': str(page)})

            yield self.GOOGLE_SEARCH_URL + '?' + params
            page += 1

    def get_links(self, html, type):
        '''リンク取得'''
        soup = BeautifulSoup(html, 'lxml')
        if type == 'text':
            elements = soup.select('.rc > .r > a')
            links = [e['href'] for e in elements]
        elif type == 'image':
            elements = soup.select('.rg_meta.notranslate')
            jsons = [json.loads(e.get_text()) for e in elements]
            links = [js['ou'] for js in jsons]
        return links

def download_img(keyword=None, maximum=None):
    if keyword == None:
        comma_bool = True
        while comma_bool:
            comma_bool = False
            keyword = input('検索ワードを入力してください（半角カンマ区切りで複数ワード指定可）: ')
            if '、' in keyword:
                q_bool = True
                while q_bool:
                    q = input('半角カンマではなく「、」でよろしかったですか？（y/n）: ')
                    if q == 'y':
                        q_bool = False
                    elif q == 'n':
                        q_bool = False
                        comma_bool = True
                    else:
                        print('Error: y/n（半角）で入力してください')
    keywords = keyword.split(',')
    pronunciation_list = []
    for name in keywords:
        not_half_size = False
        for c in name:
            if unicodedata.east_asian_width(c) != 'Na':
                not_half_size = True
        if not_half_size:
            not_half_alphabet = True
            while not_half_alphabet:
                not_half_alphabet = False
                pronunciation = input('「' + name + '」の読み方を半角アルファベットで入力してください : ')
                for c in pronunciation:
                    if unicodedata.east_asian_width(c) != 'Na':
                        not_half_alphabet = True
                        print('Error:半角アルファベットで入力してください')
        else:
            pronunciation = name
        pronunciation = pronunciation.replace(' ', '')
        pronunciation_list.append(pronunciation)
    if maximum == None:
        not_half_size = True
        while not_half_size:
            maximum = input('保存するファイル数の上限を入力してください（半角数字）: ')
            not_half_size = False
            for c in maximum:
                if unicodedata.east_asian_width(c) != 'Na':
                    not_half_size = True
            if not_half_size == True:
                print('Error:半角数字を入力してください')
        maximum = int(maximum)
    google = Google()
    cwd = os.getcwd()
    try:
        os.chdir(data_folder)
    except FileNotFoundError :
        os.makedirs(data_folder)
        os.chdir(data_folder)
    dir = os.getcwd()
    missing_img = 0  # 初期値
    for i, searchword in enumerate(keywords):
        path = './' + pronunciation_list[i]
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        result = google.Search(searchword, type='image', maximum=maximum*2)
        file_number = 0
        for url in result:
            if file_number >= maximum:
                continue
            else:
                file_name = pronunciation_list[i] + '_' + str(file_number) + '.jpg'
                if 'www' in url:  # wwwを含むとうまく取得できないのでは？
                    continue
                elif 'upload' in url:  # uploadを含んでもうまくいかなさそう
                    continue
                else:
                    r = requests.get(url, stream=True)
                    if r.status_code == 200:
                        with open(file_name, 'wb') as f:
                            f.write(r.content)
                    file_number += 1
        os.chdir(dir)
        print('->' + str(file_number) + '枚の画像が保存されました')
    os.chdir(cwd)

if __name__ == '__main__':
    download_img()

"""
確認用
maximum = 200
google = Google()
result = google.Search('与田祐希', type='image', maximum=maximum*2)
for i, url in enumerate(result):
    print(str(i))
    if 'www' in url:  # wwwを含むとうまく取得できないのでは？
        continue
    elif 'upload' in url:
        continue
    else:
        r = requests.get(url, stream=True)
"""

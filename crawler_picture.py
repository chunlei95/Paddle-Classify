import os.path
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

path = '.crop_datasets\\'
pattern = '[\u4e00-\u9fa5]+'


def parseHtml(text, query, num):
    b = BeautifulSoup(text, 'html.parser')
    img = b.find_all('img', class_='main_img img-hover')
    if len(img) > num:
        img = img[:num]
    loop = tqdm(range(len(img)), total=len(img), desc='Total collect {} images'.format(len(img)), colour='green')
    for index, value in enumerate(loop):
        imageUrl = img[index]['data-imgurl']
        imageName = 'image_' + str(index) + '.jpg'
        savePhoto(imageUrl, query, imageName)


def getImageTitle(title, titleIndex):
    if 'title' in title[titleIndex].attrs:
        return titleIndex
    else:
        getImageTitle(title, titleIndex + 1)
        return titleIndex + 1


def savePhoto(imageUrl, query, imageName):
    p = os.path.join(path, query)
    if not os.path.exists(p):
        os.makedirs(p)
    p = os.path.join(p, imageName)
    if os.path.exists(p):
        pass
    else:
        c = requests.get(imageUrl)
        createFile(path)
        with open(p, 'wb') as f:
            f.write(c.content)
        time.sleep(0.2)


def createFile(path):
    file = os.path.exists(path)
    if not file:
        os.makedirs(path)


def slideBrowseWindow(driver, number):
    for i in range(number):
        time.sleep(0.2)
        driver.execute_script('window.scrollBy(0, {})'.format(i * 1000))
        time.sleep(0.3)


if __name__ == '__main__':
    driver = webdriver.Edge()

    query = '芹菜'
    num = 400

    scroll_num = num // 10
    search_url = f"https://image.baidu.com/search/index?tn=baiduimage&word={quote(query)}"
    driver.get(search_url)
    slideBrowseWindow(driver, scroll_num)
    page_source = driver.page_source
    driver.minimize_window()
    parseHtml(page_source, query, num)
    driver.quit()

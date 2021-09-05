import requests
import zipfile
import os
import pandas as pd
import shutil
import urllib
import gdown
from preprocessing import crop_resize

'''
Downloads, crops and resizes to 256x256 images from the following datasets:
>MClass Dermoscopic
>MClass Clinical
>ASAN
Raw images are downloaded temporarily and deleted once resizing has taken place
ISIC and Atlas data should be downloaded separately.
'''

shutil.rmtree('data/images')  # clearing images folder to make sure script runs if executed second time

# Creating directories for images to be downloaded
os.makedirs('data/images/', exist_ok=True)  # remaking images folder
os.makedirs('data/images/atlas_256/', exist_ok=True)
os.makedirs('data/images/MClassD_256/', exist_ok=True)
os.makedirs('data/images/MClassC_256/', exist_ok=True)
os.makedirs('data/images/asan_256/', exist_ok=True)
print('Directories created')

# Downloading ISIC data
url20 = 'https://drive.google.com/uc?id=1BUoRe_0AABhlTOZdu_JONSuCzTgXqem6'
output20 = 'data/raw_images/isic_20_train_256.zip'
url19 = 'https://drive.google.com/uc?id=1fUqSJs_IO1MlmkS6MAp80IGQBZf7gggP'
output19 = 'data/raw_images/isic_19_train_256.zip'
gdown.download(url20, output20, quiet=False)
gdown.download(url19, output19, quiet=False)
print('ISIC images downloaded')

# Downloading zip file of raw MClass dermoscopic images
mclassd = requests.get("https://skinclass.de/MClass/MClass-D.zip")
path_to_mclassd = "./data/raw_images/MClassD_raw.zip"
file = open(path_to_mclassd, "wb")
file.write(mclassd.content)
file.close()
print('Raw MClass Dermoscopic images downloaded')

# Downloading zip file of raw MClass clinical images
mclassc = requests.get("https://skinclass.de/MClass/MClass-ND")
path_to_mclassc = "./data/raw_images/MClassC_raw.zip"
file = open(path_to_mclassc, "wb")
file.write(mclassc.content)
file.close()
print('Raw MClass Clinical images downloaded')

# Downloading zip file of raw ASAN images
ASAN = requests.get("https://ndownloader.figshare.com/files/9328573")
path_to_asan = "./data/raw_images/ASAN_raw.zip"
file = open(path_to_asan, "wb")
file.write(ASAN.content)
file.close()
print('Raw ASAN images downloaded')

# Unzipping Atlas, MClass and ASAN files
with zipfile.ZipFile('data/raw_images/isic_20_train_256.zip', 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/isic_20_train_256')
with zipfile.ZipFile('data/raw_images/isic_19_train_256.zip', 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/isic_19_train_256')
with zipfile.ZipFile('data/raw_images/release_v0.zip', 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/Atlas')
with zipfile.ZipFile(path_to_mclassd, 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/MClassD')
with zipfile.ZipFile(path_to_mclassc, 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/MClassC')
with zipfile.ZipFile(path_to_asan, 'r') as zip_ref:
    zip_ref.extractall('data/raw_images/ASAN')

shutil.move('data/raw_images/isic_20_train_256/isic_20_train_256', 'data/images/')
shutil.move('data/raw_images/isic_19_train_256/isic_19_train_256', 'data/images/')

# Centre cropping and resizing to 256x256
crop_resize('data/raw_images/Atlas/release_v0/images', 'data/images/atlas_256/', 256)
print('Atlas centre cropped and resized to 256x256')
crop_resize('data/raw_images/MClassD/BenchmarkDermoscopic/', 'data/images/MClassD_256/', 256)
print('MClass Dermoscopic centre cropped and resized to 256x256')
crop_resize('data/raw_images/MClassC/BenchmarkClinical/', 'data/images/MClassC_256/', 256)
print('MClass Clinical centre cropped and resized to 256x256')
crop_resize('data/raw_images/ASAN/test-asan test', 'data/images/asan_256/', 256)
print('ASAN centre cropped and resized to 256x256')

# Deleting temporary store of raw images
shutil.rmtree('data/raw_images/')
print('Temporary raw images discarded')

import math
import os
import requests
from tqdm import tqdm
import zipfile

VGG19 = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'

# 2014 COCO dataset
COCO_TRAIN = 'http://images.cocodataset.org/zips/train2014.zip'
COCO_VALIDATE = 'http://images.cocodataset.org/zips/val2014.zip'
COCO_ANNOTATE = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'


file_names = {
        VGG19: 'imagenet-vgg-verydeep-19.mat',
        COCO_TRAIN: 'train2014.zip',
        COCO_VALIDATE: 'val2014.zip',
        COCO_ANNOTATE: 'annotations_trainval2014.zip'
        }


    #####################
    #    Downloading    #
    #####################

def download(URL):
    base_folder = os.path.dirname(__file__)
    download_location = os.path.join(base_folder, 'data')
    file_name = os.path.join(download_location, file_names[URL])
        
    response = requests.get(URL, stream=True)
    response.raise_for_status()
    file_size = int(response.headers.get('content-length', 0))

    if not os.path.isdir(download_location):
        print("Creating 'data/' directory")
        os.mkdir(download_location)
    
    if os.path.exists(file_name) and os.stat(file_name).st_size == file_size:
        print(f'{file_name} is ready.')
    else:
        print(f'Downloading {file_names[URL]}')

        count = 0
        block_size = 32 * 1024

        with open(file_name, 'wb') as data_file:
            with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        data_file.write(chunk)
                        pbar.update(len(chunk))
        print(f'{file_names[URL]} downloaded.')

    return download_location, file_name, file_size


    #####################
    #     Unzipping     #
    #####################

def unzip(URL, download_location, file_name):

    with zipfile.ZipFile(file_name, 'r') as zf:
        uncompressed_size = sum(file.file_size for file in zf.infolist())

        with tqdm(total=uncompressed_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for sub_file in zf.infolist():
                zf.extract(sub_file, download_location)
                pbar.update(sub_file.file_size)

    print(f'{file_names[URL]} has been extracted.')


    ######################
    #  Download + Unzip  #
    ######################

def download_and_unzip(URL):
    download_location, file_name, file_size = download(URL)

    if not os.path.exists(file_name) or not os.stat(file_name).st_size == file_size:
        raise ValueError(f'{file_names[URL]} download was incomplete. Cannot proceed with unzip.')
    unzip(URL, download_location, file_name)


if __name__=="__main__":
    try:
        download(VGG19)
        download_and_unzip(COCO_TRAIN)
        download_and_unzip(COCO_VALIDATE)
        download_and_unzip(COCO_ANNOTATE)
    except requests.exceptions.ConnectionError as ce:
        print(ce)
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(e)


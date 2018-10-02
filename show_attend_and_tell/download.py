import math
import os
import requests
from tqdm import tqdm

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


def download(URL):
    download_location = 'data/'
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

    return file_name


if __name__=="__main__":
    try:
        download(VGG19)
        download(COCO_TRAIN)
        download(COCO_VALIDATE)
        download(COCO_ANNOTATE)
    except requests.exceptions.ConnectionError as ce:
        print(ce)
    except Exception as e:
        print(e)


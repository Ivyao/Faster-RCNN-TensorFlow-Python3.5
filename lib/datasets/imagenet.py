CLASSES = {'synthesizer':'n04376400', 'pipe organ':'n03854065', 'music box': 'n03801353', \
        'electric guitar':'n03272010', 'sax':'n04141076', 'ocarina':'n03840681', 'harmonica':'n03494278',\
        'acoustic guitar':'n02676566', 'trombone':'n04487394','gong':'n03447721',\
        'maraca':'n03720891', 'xylophone':'n03721384', 'pianoforte':'n03928116'}

'''
Given the wnid of a synset, the URLs of its images can be obtained at
http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=[wnid]
'''
IMGS_URL='http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={wnid}'
ANN_URL='http://www.image-net.org/api/download/imagenet.bbox.synset?wnid={wnid}'


TEST = True


#n03447721_39009 http://www.musik-klier.de/prods/Stagg-Gong%2020%20klein.jpg
import requests
from os import linesep
import os.path
import logging
import urllib3
from PIL import Image
from shutil import copyfile
import random

IMAGENET_FOLDER = os.path.join('data', 'imagenet')
IMAGES_FOLDER = os.path.join(IMAGENET_FOLDER, 'JPEGImages')
ANNOTATIONS_FOLDER_SRC = os.path.join(IMAGENET_FOLDER, 'Annotation_imagenet')
ANNOTATIONS_FOLDER_DST = os.path.join(IMAGENET_FOLDER, 'Annotations')
PARTITION_FOLDER = os.path.join(IMAGENET_FOLDER, 'ImageSets', 'Main') 

def is_jpg(filename):
    try:
        i=Image.open(filename)
        return i.format =='JPEG'
    except IOError:
        return False

def load_imagenet_dataset(partition_names):
    index = 0

    # n_partitions = len(partition_names)

    # open_files = {}
    # for partition_name in partition_names:
    #     open_files[partition_name] = open(os.path.join(PARTITION_FOLDER, partition_name + '.txt'))
    #     for class_name in CLASSES.keys():
    #         open(os.path.join(PARTITION_FOLDER, class_name + '_' + partition_name + '.txt'))

    loaded_images = []

    http = urllib3.PoolManager()

    for code in CLASSES.values():
        image_urls_req = requests.get(IMGS_URL.format(wnid=code))

        # print(image_urls_req)

        image_urls_line = image_urls_req.text.split(linesep)

        # iterating over each image in the retrieved text file
        for line in image_urls_line:
            # splitting line over spaces
            line_list = line.split()
            
            # the first element will contain the id of the image, the second one the url        
            if len(line_list) != 2:
                logging.error(len(line_list))
                continue

            image_id = line_list[0]
            image_url = line_list[1]

            logging.error('image_id = '+ image_id)
            logging.error('image_url = '+ image_url)

            # checking if exists the annotation file
            if not os.path.exists(os.path.join(ANNOTATIONS_FOLDER_SRC, image_id + '.xml')):
                logging.error('annotation does not exist')
                continue

            # retrieving image
            try:
                image_request = requests.get(image_url)
            except requests.exceptions.ConnectionError:
                continue
            
            # checking if exists the image in the given url 
            if image_request.status_code != 200:
                logging.error('image problems, code ' + str(image_request.status_code))
                continue

            image_request = http.request('GET', image_url)
            
            # image_request = urllib3.urlopen(image_url)   
            if image_request.status == 404:
                continue

            index_string = str(index).zfill(6)
            image_filename = os.path.join(IMAGES_FOLDER, index_string + '.jpg')
            with open(image_filename, 'wb') as f:
                f.write(image_request.data)

            # checking if the data contains a jpg 
            if not is_jpg(image_filename):
                os.remove(image_filename)
                continue
            
            copyfile(os.path.join(ANNOTATIONS_FOLDER_SRC, image_id + '.xml'), 
                    os.path.join(ANNOTATIONS_FOLDER_DST, index_string + '.xml'))
            index+=1

            loaded_images.append((index_string, code, random.randint(0,1)))
            
            if TEST and index > 0:
               break

    part_0 = open(os.path.join(PARTITION_FOLDER, partition_names[0] + '.txt'), 'w')
    part_1 = open(os.path.join(PARTITION_FOLDER, partition_names[1] + '.txt'), 'w')

    #doMagic
    for elem in loaded_images:
        if elem[2] == 0:
            part_0.write('{}'.format(elem[0]))
        else:    
            part_1.write('{}'.format(elem[0]))

    part_0.close()
    part_1.close()
    
    for class_name in CLASSES.values():
        part_0 = open(os.path.join(PARTITION_FOLDER, class_name + '_' + partition_names[0] + '.txt'), 'w')
        part_1 = open(os.path.join(PARTITION_FOLDER, class_name + '_' + partition_names[1] + '.txt'), 'w')

        for elem in loaded_images:
            if elem[2] == 0:
                part_0.write('{} {}'.format(elem[0], 1 if(elem[1] == class_name) else -1))
            else:    
                part_1.write('{} {}'.format(elem[0], 1 if(elem[1] == class_name) else -1))
        part_0.close()
        part_1.close()


load_imagenet_dataset(["test", "train"])
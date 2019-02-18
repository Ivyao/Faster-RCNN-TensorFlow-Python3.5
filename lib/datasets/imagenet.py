CLASSES = {'synthesizer':'n04376400', 'pipe organ':'n03854065', 'music box': 'n03801353', \
        'electric guitar':'n03272010', 'sax':'n04141076', 'ocarina':'n03840681', 'harmonica':'n03494278',\
        'acoustic guitar':'n02676566', 'trombone':'n04487394','gong':'n03447721',\
        'maraca':'n03720891', 'xylophone':'n03721384', 'pianoforte':'n03928116'}

IMGS_URL='http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={wnid}'
ANN_URL='http://www.image-net.org/api/download/imagenet.bbox.synset?wnid={wnid}'


TEST = False

import requests
from os import linesep
import os.path
import logging
import urllib3
# disabling security warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from PIL import Image
from shutil import copyfile
import random
import xml.etree.ElementTree as ET

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
    index = 1

    loaded_images = []

    http = urllib3.PoolManager()

    for code in CLASSES.values():
        image_urls_req = requests.get(IMGS_URL.format(wnid=code))

        image_urls_line = image_urls_req.text.split(linesep)

        # iterating over each image in the retrieved text file
        for line in image_urls_line:
            # splitting line over spaces
            line_list = line.split()
            
            # the first element will contain the id of the image, the second one the url        
            if len(line_list) != 2:
                #logging.error(len(line_list))
                continue

            image_id = line_list[0]
            image_url = line_list[1]

            # checking if exists the annotation file
            annotation_filename = os.path.join(ANNOTATIONS_FOLDER_SRC, image_id + '.xml')
            if not os.path.exists(annotation_filename):
                # logging.error('annotation does not exist')
                continue
            
            try:
                image_request = http.request('GET', image_url)
            except BaseException:
                continue
              
            if image_request.status != 200:
                continue

            index_string = str(index).zfill(6)
            image_filename = os.path.join(IMAGES_FOLDER, index_string + '.jpg')
            with open(image_filename, 'wb') as f:
                f.write(image_request.data)

            # checking if the data contains a jpg 
            if not is_jpg(image_filename):
                os.remove(image_filename)
                continue

            # discarding images whose width is not the one specified in the xml
            root = ET.parse(annotation_filename).getroot()
            
            # retrieving annotation expected width
            annotation_width = int(root.find('size/width').text)
            # calculating real image width
            real_width = Image.open(image_filename).size[0]

            if real_width != annotation_width:
                # unmatching width
                os.remove(image_filename)
                continue

            copyfile(annotation_filename, 
                    os.path.join(ANNOTATIONS_FOLDER_DST, index_string + '.xml'))
            index+=1

            if index%10 == 0:
                print("Downloaded " + str(index) + " images")

            loaded_images.append((index_string, code, random.randint(0,1)))
            
            # in this case we try to download just an image for each class
            if TEST and index > 0:
               break
    
    # saving indexes in partition txts
    part_0 = open(os.path.join(PARTITION_FOLDER, partition_names[0] + '.txt'), 'w')
    part_1 = open(os.path.join(PARTITION_FOLDER, partition_names[1] + '.txt'), 'w')

    #doMagic
    for elem in loaded_images:
        if elem[2] == 0:
            part_0.write('{}\n'.format(elem[0]))
        else:    
            part_1.write('{}\n'.format(elem[0]))

    part_0.close()
    part_1.close()
    
    # partitioning dataset
    for class_name in CLASSES.values():

        # opening files
        part_0 = open(os.path.join(PARTITION_FOLDER, class_name + '_' + partition_names[0] + '.txt'), 'w')
        part_1 = open(os.path.join(PARTITION_FOLDER, class_name + '_' + partition_names[1] + '.txt'), 'w')

        # iterating over each image in the dataset
        for elem in loaded_images:
            if elem[2] == 0:
                part_0.write('{} {}\n'.format(elem[0], 1 if(elem[1] == class_name) else -1))
            else:    
                part_1.write('{} {}\n'.format(elem[0], 1 if(elem[1] == class_name) else -1))
        
        # closing files
        part_0.close()
        part_1.close()

if not TEST:
    load_imagenet_dataset(["test", "train"])
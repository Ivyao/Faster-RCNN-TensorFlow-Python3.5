	
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

#n03447721_39009 http://www.musik-klier.de/prods/Stagg-Gong%2020%20klein.jpg
import requests
from os import linesep
import os.path
import logging
import urllib3
from PIL import Image
from shutil import copyfile

IMAGENET_FOLDER = os.path.join('data', 'imagenet')
IMAGES_FOLDER = os.path.join(IMAGENET_FOLDER, 'JPEGImages')
ANNOTATIONS_FOLDER_SRC = os.path.join(IMAGENET_FOLDER, 'Annotation_imagenet')
ANNOTATIONS_FOLDER_DST = os.path.join(IMAGENET_FOLDER, 'Annotations')

def is_jpg(filename):
    try:
        i=Image.open(filename)
        return i.format =='JPEG'
    except IOError:
        return False

index = 0
for code in CLASSES.values():
    image_urls_req = requests.get(IMGS_URL.format(wnid=code))

    print(image_urls_req)

    image_urls_line = image_urls_req.text.split(linesep)

    # iterating over each image in the retrieved text file
    for line in image_urls_line:
        # splitting line over spaces
        line_list = line.split()
        
        # the first element will contain the id of the image, the second one the url        
        if len(line_list) != 2:
            logging.error(len(line_list))
            continue

        logging.error('list > 2')

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

        http = urllib3.PoolManager()

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
        else:
            copyfile(os.path.join(ANNOTATIONS_FOLDER_SRC, image_id + '.xml'), 
                     os.path.join(ANNOTATIONS_FOLDER_DST, index_string + '.xml'))
            index+=1

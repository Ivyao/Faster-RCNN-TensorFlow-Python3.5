#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

DEMO_IMAGES_DIR = "demo_images"

CLASSES_DICT = {'synthesizer':'n04376400', 'pipe organ':'n03854065', 'music box': 'n03801353', \
        'electric guitar':'n03272010', 'sax':'n04141076', 'ocarina':'n03840681', 'harmonica':'n03494278',\
        'acoustic guitar':'n02676566', 'trombone':'n04487394','gong':'n03447721',\
        'maraca':'n03720891', 'xylophone':'n03721384', 'pianoforte':'n03928116'}

CLASSES = ('__background__','n04376400','n03854065','n03801353','n03272010', 
                                        'n04141076','n03840681','n03494278','n02676566','n04487394',
                                        'n03447721','n03720891','n03721384','n03928116')


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """ Saves an image with name fig_id"""

    path = os.path.join(DEMO_IMAGES_DIR,"{}.png".format(fig_id))
    print("Saving figure", fig_id)
    
    if tight_layout:
        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)


def my_vis_detections(im, d, thresh=0.5, image_name = "Null"):
    """Draw detected bounding boxes."""

    all_classes = ""

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    for key in d.keys():
        inds = np.where((d[key])[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue
        class_name = ""
        for class_ in CLASSES_DICT.keys():
            if CLASSES_DICT[class_] == key:
                class_name = class_
                break

        all_classes += class_name+"_"
        for i in inds:
            bbox = d[key][i, :4]
            score = d[key][i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(all_classes, all_classes,
                                                  thresh),
                 fontsize=14)

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    save_fig("{}_{}".format(image_name,all_classes))


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    #print(scores)
    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.2
    # d is a dictionary that contains as keys the name of the classes, as value dets
    d = {}
    #print(scores)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        d[cls] = dets


    my_vis_detections(im, d, thresh=CONF_THRESH, image_name = image_name.split(".jpg")[0])

if __name__ == '__main__':
    #args = parse_args()

    # model path
    demonet = 'vgg16'
    tfmodel = os.path.join('default','model.ckpt')
    
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    else:
        raise NotImplementedError

    net.create_architecture(sess, "TEST", 14,
                            tag='default', anchor_scales=[8, 16, 32])
    
    # restoring from snapshot
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # test images
    im_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', "5.jpg"]
    
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)
    
    plt.show()
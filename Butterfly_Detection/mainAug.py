#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:28:08 2018

@author: JieweiLu

Function:
    This script is used to augment the crop image dataset and 
    create the augmented crop image dataset    
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from funcOwn import funcAug


dataset_crop = '/home/JieweiLu/jackie/Program/ButterflyProgram/ButterflyProgram4/saveDataset/datasetKuo2/image/'
savePath = '/home/JieweiLu/jackie/Program/ButterflyProgram/ButterflyProgram4/saveDataset/datasetKuo2Aug/image/'


def main(_):
    # initialization
    photo_filenames = []
    filenameAll = os.listdir(dataset_crop)
    for filename in filenameAll:
        path = os.path.join(dataset_crop, filename)
        photo_filenames.append(path)
    num_image = len(filenameAll)
    for i in range(num_image):
        # read the crop images     
        print(i)
        # get the splited name
        result = filenameAll[i].split('_')
        index = result[0]
        ib = result[1]
        label = result[2]
        name = result[4]

        # perform the data augmentation
        funcAug.data_augmentation(photo_filenames,i,index,ib,label,name,savePath)
        
            
    
    print('OK')
    
if __name__ == '__main__':
    tf.app.run()



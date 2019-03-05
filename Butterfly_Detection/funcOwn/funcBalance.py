#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:00:00 2018

@author: JieweiLu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
# Seed for repeatability
_RANDOM_SEED = 0

def getBalanceFileName(pathDataCleanClaAll):
    filenameAllSum = os.listdir(pathDataCleanClaAll)
    
    # find the max number of picture class
    maxNumAll = []
    for filenameClass in filenameAllSum:
        path = os.path.join(pathDataCleanClaAll, filenameClass)
        picName = os.listdir(path)
        maxNumAll.append(len(picName))
    
    maxNum = max(maxNumAll)
    
    # deal with each class file
    filenameHuiZong = []
    for filenameClass in filenameAllSum:
        path = os.path.join(pathDataCleanClaAll, filenameClass)
        picName = os.listdir(path)
        picNum = len(picName)
    
        runNum = math.ceil(maxNum/picNum)
        extraNum = maxNum%picNum
        if extraNum==0:
            extraNum=picNum
        
        filenameOneAll = []
        for i in range(runNum):
            if i!=(runNum-1):
                filenameOneAll.extend(picName)
            else:            
                filenameOneAll.extend(picName[0:(extraNum)])
        
        # shuffle the dataset
        random.seed(_RANDOM_SEED)
        random.shuffle(filenameOneAll)
        
        filenameHuiZong.extend(filenameOneAll)
    
    # shuffle the dataset
    random.seed(_RANDOM_SEED)
    random.shuffle(filenameHuiZong)
    
    
    
    return filenameHuiZong
#!/usr/bin/python
#-*- coding: utf-8 -*- 

import os
import sys

extractors = ("SIFT", "SURF")
detectors = ("STAR", "SIFT", "SURF", "FAST")
classifiers = ("NormalBayesClassifier", "KNearest")

for detector in detectors:
    for extractor in extractors:
        for classifier in classifiers:
            os.system("./DetectorDeMonumentos %s %s %s >> avaliacao" % (detector, extractor, classifier))

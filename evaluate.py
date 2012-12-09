#!/usr/bin/python
#-*- coding: utf-8 -*- 

import os
import sys

extractors = ("SURF", "SIFT")
detectors = ("SURF", "FAST", "STAR", "SIFT")
classifiers = ("NormalBayesClassifier", "KNearest", "SVM", "DTree")

for detector in detectors:
    for extractor in extractors:
        for classifier in classifiers:
            os.system("./DetectorDeMonumentos %s %s %s >> avaliacao" % (detector, extractor, classifier))

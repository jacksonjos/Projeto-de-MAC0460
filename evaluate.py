#!/usr/bin/python
#-*- coding: utf-8 -*- 

import os
import sys

<<<<<<< HEAD
extractors = ("SIFT", "SURF")
detectors = ("STAR", "SIFT", "SURF", "FAST")
classifiers = ("NormalBayesClassifier", "KNearest")
=======
extractors = ("SURF", "SIFT")
detectors = ("SURF", "FAST", "STAR", "SIFT")
classifiers = ("NormalBayesClassifier", "KNearest", "SVM", "DTree")
>>>>>>> 75252e2614f5c8abe3aa8c25737f22b9ef4d50db

for detector in detectors:
    for extractor in extractors:
        for classifier in classifiers:
            os.system("./DetectorDeMonumentos %s %s %s >> avaliacao" % (detector, extractor, classifier))

#!/usr/bin/python
#-*- coding: utf-8 -*- 

import os
import sys

extractors = ("SURF", "SIFT")
detectors = ("SURF", "FAST", "STAR", "SIFT")
classifiers = ("KNearest", "NormalBayesClassifier")

for classifier in classifiers:
	for detector in detectors:
		for extractor in extractors:
			os.system("./DetectorDeMonumentos %s %s %s >> avaliacao-final2" % (detector, extractor, classifier))

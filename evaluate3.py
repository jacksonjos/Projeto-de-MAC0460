#!/usr/bin/python
#-*- coding: utf-8 -*- 

import os
import sys

extractors = ("SURF", "SIFT")
detectors = ("SURF", "FAST", "STAR", "SIFT")
classifiers = ("NormalBayesClassifier", "KNearest")

for classifier in classifiers:
	for detector in detectors:
		for extractor in extractors:
			os.system("./DetectorDeMonumentos %s %s %s >> avaliacao-final1" % (detector, extractor, classifier))

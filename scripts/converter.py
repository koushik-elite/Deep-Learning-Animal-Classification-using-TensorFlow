import argparse
import sys
import time
import csv
import os
import cv2
from tempfile import NamedTemporaryFile
import shutil

tempfile = NamedTemporaryFile(mode='w', delete=False)
successCount = 0
with open('C:/FPS - Projects/GIT/tensorflow/meta-data/result/result.csv', 'r') as csvfile:
    fields = ['image_id','antelope','bat','beaver','bobcat','buffalo','chihuahua','chimpanzee','collie','dalmatian','german+shepherd','grizzly+bear','hippopotamus','horse','killer+whale','mole','moose','mouse','otter','ox','persian+cat','raccoon','rat','rhinoceros','seal','siamese+cat','spider+monkey','squirrel','walrus','weasel','wolf']
    imagereader = csv.reader(csvfile, delimiter=',', quotechar='|')
    writer = csv.DictWriter(tempfile, delimiter=',', lineterminator='\n', fieldnames=fields)
    for row in imagereader:
        sys.stdout.flush()
        wrow = {}
        # print(row)
        for i, x in enumerate(row):
            successCount = successCount + 1;
            if len(x)< 1:
                x = row[i] = 0
            # print x
            wrow[fields[i]] = row[i]
        writer.writerow(wrow)
        sys.stdout.write('Row Count={} \r'.format(successCount))
    tempfile.close()
    shutil.move(tempfile.name, 'meta-data/result_new.csv')
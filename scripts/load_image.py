import csv
import os
import shutil
import cv2
import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
successCount = 0
failesCount = 0
train_file = "meta-data/train.csv"
train_folder = "train"
parser = argparse.ArgumentParser()
parser.add_argument("--train_folder", help="train image folder")
parser.add_argument("--train_file", help="train image file")
args = parser.parse_args()

if args.train_file:
    train_file = args.train_file
if args.train_folder:
    train_folder = args.train_folder

with open(train_file, 'r') as csvfile:
    imagereader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in imagereader:
        sys.stdout.flush()
        if not os.path.exists('tf_files/animals/'+row[1]):
            os.makedirs('tf_files/animals/'+row[1])
        # adding exception handling
        try:
            # shutil.copy('tf_files/animals/train/'+row[0], 'tf_files/animals/'+row[1])
            img = cv2.imread(train_folder + row[0], cv2.IMREAD_UNCHANGED)
            cv2.imwrite( 'tf_files/animals/'+row[1]+'/'+row[0], image_resize(img, 224));
            successCount = successCount + 1
        except Exception as e:
            print("Unable to copy "+row[0]+" file to "+row[1]+". %s" % e)
            failesCount = failesCount + 1
            pass
        sys.stdout.write('successCount={} | failesCount={} \r'.format(successCount, failesCount))
print('successCount=', successCount, ' | failesCount=', failesCount)
#!/usr/bin/env python3

# Vision finder to find the retro-reflective target around the goal

import sys
import os.path
import cv2
import numpy
import argparse
import csv


def contour_center_width(contour):
    '''Find boundingRect of contour, but return center and width/height'''

    x, y, w, h = cv2.boundingRect(contour)
    return numpy.array((x + int(w / 2), y + int(h / 2))), numpy.array((w, h))


def color_analysis(camera_frame):
    hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
    threshold_frame = cv2.inRange(hsv_frame, low_limit_hsv, high_limit_hsv)
    return threshold_frame


def gray_analysis(camera_frame, threshold):
    gray_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
    _, threshold_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    return threshold_frame


parser = argparse.ArgumentParser(description='Vibration test')
parser.add_argument('--output_dir', required=True, help='Output directory for processed images')
parser.add_argument('--csv', required=True, help='CSV output data')
parser.add_argument('input_files', nargs='+', help='input files')

args = parser.parse_args()

if sys.platform == "win32":
    # windows does not expand the "*" files on the command line
    #  so we have to do it.
    import glob

    infiles = []
    for f in args.input_files:
        infiles.extend(glob.glob(f))
    args.input_files = infiles

# Color threshold values, in HSV space
low_limit_hsv = numpy.array((42, 0, 170), dtype=numpy.uint8)
high_limit_hsv = numpy.array((90, 255, 255), dtype=numpy.uint8)

image_center = numpy.array((720.0/2, 1280.0/2))

csvout = csv.DictWriter(open(args.csv, 'w'), fieldnames=('File', 'NFound', 'ContourArea',
                                                         'ContourWidth', 'ContourHeight', 'ContourX', 'ContourY'))
csvout.writeheader()

for infile in args.input_files:
    camera_frame = cv2.imread(infile)

    # threshold_frame = color_analysis(camera_frame)
    threshold_frame = gray_analysis(camera_frame, threshold=200)

    # OpenCV 3 returns 3 parameters, OpenCV 4 returns 2!
    # Only need the contours variable
    res = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours = res[0]
    else:
        contours = res[1]

    best = None
    nfound = 0
    for c in contours:
        center, widths = contour_center_width(c)

        # For the video: look at the center
        offset = center - image_center
        if abs(offset[0]) > 200 or abs(offset[1]) > 200:
            continue

        nfound += 1
        area = widths[0] * widths[1]
        if best is None or area > best['area']:
            best = {'contour': c, 'area': area, 'widths': widths, 'center': center}

    data = {'File': infile,
            'NFound': nfound}
    if best:
        data.update({'ContourArea': cv2.contourArea(best['contour']),
                     'ContourWidth': best['widths'][0],
                     'ContourHeight': best['widths'][1],
                     'ContourX': best['center'][0],
                     'ContourY': best['center'][1]})
        cv2.drawContours(camera_frame, [best['contour']], -1, (0, 0, 255), 2)

    csvout.writerow(data)
    outfile = os.path.join(args.output_dir, os.path.basename(infile))
    cv2.imwrite(outfile, camera_frame)

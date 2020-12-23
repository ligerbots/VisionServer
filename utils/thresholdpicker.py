#!/usr/bin/env python3

# Simple program to pick the color thresholds for a set of images

import cv2
import numpy
import argparse
import sys


def proceed(x):
    global run
    run = True
    return


def process_files(image_files):
    global run

    # WINDOW_NORMAL allows the size to be changed
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('image')

    cv2.createTrackbar('Hlow', 'image', 65, 255, proceed)
    cv2.createTrackbar('Hhigh', 'image', 100, 255, proceed)
    cv2.createTrackbar('Slow', 'image', 75, 255, proceed)
    cv2.createTrackbar('Shigh', 'image', 255, 255, proceed)
    cv2.createTrackbar('Vlow', 'image', 140, 255, proceed)
    cv2.createTrackbar('Vhigh', 'image', 255, 255, proceed)

    image_id = -1
    cv2.createTrackbar('Image#', 'image', 0, len(image_files)-1, proceed)

    # fixed_height = 450

    # # create switch for ON/OFF functionality
    # switch = '0 : OFF \n1 : ON'
    # cv2.createTrackbar(switch, 'image',0,1,nothing)

    print('Press "q" to exit')

    while True:
        if run:
            if image_id != cv2.getTrackbarPos('Image#', 'image'):
                image_id = cv2.getTrackbarPos('Image#', 'image')
                image_file = image_files[image_id]
                bgr_frame = cv2.imread(image_file)
                hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
                draw_frame = numpy.zeros(shape=bgr_frame.shape, dtype=numpy.uint8)

            # get current positions of four trackbars
            hLow = cv2.getTrackbarPos('Hlow', 'image')
            hHigh = cv2.getTrackbarPos('Hhigh', 'image')
            sLow = cv2.getTrackbarPos('Slow', 'image')
            vLow = cv2.getTrackbarPos('Vlow', 'image')
            sHigh = cv2.getTrackbarPos('Shigh', 'image')
            vHigh = cv2.getTrackbarPos('Vhigh', 'image')

            lowLimitHSV = numpy.array([hLow, sLow, vLow])
            highLimitHSV = numpy.array([hHigh, sHigh, vHigh])
            mask = cv2.inRange(hsv_frame, lowLimitHSV, highLimitHSV)

            # maskedFrame = cv2.bitwise_and(bgr_frame, bgr_frame, mask=mask)

            res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(res) == 2:
                contours = res[0]
            else:
                contours = res[1]

            numpy.copyto(draw_frame, bgr_frame)
            cv2.drawContours(draw_frame, contours, -1, (0, 0, 255), 1)
            run = False

        # this could be done with a factor, like "2.0"
        # height, width, channels = bgr_frame.shape
        # resized = cv2.resize(draw_frame, (int(fixed_height * (width / height)), fixed_height), interpolation=cv2.INTER_AREA)
        cv2.imshow('image', draw_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('hLow =', cv2.getTrackbarPos('Hlow', 'image'))
    print('hHigh =', cv2.getTrackbarPos('Hhigh', 'image'))
    print('sLow =', cv2.getTrackbarPos('Slow', 'image'))
    print('sHigh =', cv2.getTrackbarPos('Shigh', 'image'))
    print('vLow =', cv2.getTrackbarPos('Vlow', 'image'))
    print('vHigh =', cv2.getTrackbarPos('Vhigh', 'image'))

    cv2.destroyAllWindows()
    return


run = True

parser = argparse.ArgumentParser(description='Color threshold utility')
parser.add_argument('input_files', nargs='+', help='Input image files')

args = parser.parse_args()

if sys.platform == "win32":
    # windows does not expand the "*" files on the command line
    #  so we have to do it.
    import glob

    infiles = []
    for f in args.input_files:
        infiles.extend(glob.glob(f))
    args.input_files = infiles

process_files(args.input_files)

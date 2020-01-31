#!/usr/bin/env python3

# Simple program to draw a line on images and tune where it goes

import cv2
import argparse
import sys


def proceed(x):
    global run
    run = True
    return


def process_image(image, x1, y1, x2, y2, rotation=None):
    # WARNING rotation=0 is actually 90deg clockwise (dumb!!)
    if rotation is not None:
        out_frame = cv2.rotate(image, rotation)
    else:
        out_frame = image.copy()
    cv2.line(out_frame, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
    return out_frame


def process_files(image_files, rotation):
    global run

    # process 1st file so we can get the sizes
    out_frame = process_image(cv2.imread(image_files[0]), 0, 0, 0, 0, rotation)
    shape = out_frame.shape
    print('Image shape', shape)

    # Create an output window, allowing it to be resized
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('X1', 'image', shape[1]//2, shape[1], proceed)  # reminder: // means integer division
    cv2.createTrackbar('Y1', 'image', 0, shape[0], proceed)
    cv2.createTrackbar('X2', 'image', shape[1]//2, shape[1], proceed)
    cv2.createTrackbar('Y2', 'image', shape[0], shape[0], proceed)

    image_id = -1
    mn = max(1, len(image_files)-1)
    cv2.createTrackbar('Image#', 'image', 0, mn, proceed)

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

            # get current positions of four trackbars
            x1 = cv2.getTrackbarPos('X1', 'image')
            y1 = cv2.getTrackbarPos('Y1', 'image')
            x2 = cv2.getTrackbarPos('X2', 'image')
            y2 = cv2.getTrackbarPos('Y2', 'image')

            draw_frame = process_image(bgr_frame, x1, y1, x2, y2, rotation)

            run = False

        cv2.imshow('image', draw_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('X1 =', cv2.getTrackbarPos('X1', 'image'))
    print('Y1 =', cv2.getTrackbarPos('Y1', 'image'))
    print('X2 =', cv2.getTrackbarPos('X2', 'image'))
    print('Y2 =', cv2.getTrackbarPos('Y2', 'image'))

    cv2.destroyAllWindows()
    return


run = True

parser = argparse.ArgumentParser(description='Tune line drawing')
parser.add_argument('--rotation', '-r', type=int, help='Rotation (0, 180, 90, -90)')
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

rotation = None
if args.rotation:
    if args.rotation == 90:
        rotation = cv2.ROTATE_90_CLOCKWISE
    elif args.rotation == 180:
        rotation = cv2.ROTATE_180
    elif args.rotation == -90:
        rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
print('rotation =', args.rotation, type(args.rotation), rotation)

process_files(args.input_files, rotation)

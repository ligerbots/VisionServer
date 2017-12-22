#!/usr/bin/env python3

import cv2
import numpy


class CameraCalibration(object):
    """Calibrates a distorted camera"""

    def __init__(self):
        # number of inside corners in the chessboard (height and width)
        self.checkerboard_height = 9
        self.checkerboard_width = 6

        # size of chessboard square in physical units
        self.square_size = 1.0
        return

    def calibrateCamera(self, images):
        '''Calculates the distortion co-efficients'''

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = numpy.zeros((self.checkerboard_height * self.checkerboard_width, 3), numpy.float32)
        objp[:, :2] = numpy.mgrid[0:self.checkerboard_width, 0:self.checkerboard_height].T.reshape(-1, 2)

        # calibrate coordinates to the physical size of the square
        objp *= self.square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []          # 3d point in real world space
        imgpoints = []          # 2d points in image plane.

        for fname in images:
            print('Processing file', fname)
            img = cv2.imread(fname)

            if img is None:
                print("ERROR: Unable to read file", fname)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.checkerboard_width, self.checkerboard_height), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (self.checkerboard_width, self.checkerboard_height), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        if not objpoints:
            print("No useful images. Quitting...")
            return None

        print('Found {} useful images'.format(len(objpoints)))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return (ret, mtx.tolist(), dist.tolist(), rvecs, tvecs)


if __name__ == '__main__':
    import argparse
    from pprint import pprint
    import json

    parser = argparse.ArgumentParser(description='Calibration utility')
    parser.add_argument('--length', '-l', type=int, default=9, help='Length of checkerboard (number of corners)')
    parser.add_argument('--width', '-w', type=int, default=6, help='Width of checkerboard (number of corners)')
    parser.add_argument('--size', '-s', type=float, default=1.0, help='Size of square')
    parser.add_argument('--output', '-o', help="Save the distortion constants to json file")
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    calibrate = CameraCalibration()
    calibrate.checkerboard_width = args.width
    calibrate.checkerboard_height = args.length
    calibrate.square_size = args.size

    ret, mtx, dist, rvecs, tvecs = calibrate.calibrateCamera(args.input_files)
    print('ret =', ret)
    print('mtx =', mtx)
    print('dist =', dist)

    # If the command line argument was specified, save matrices to a file in JSON format
    if args.output:
        # Writing JSON data
        with open(args.output, 'w') as f:
            json.dump({"camera_matrix": mtx, "distortion": dist}, f)

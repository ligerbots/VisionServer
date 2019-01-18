#!/usr/bin/env python3

import cv2
import numpy
import json
import math

class LineFinder2019(object):
    '''Find line following for Deep Space 2019'''

    # CUBE_HEIGHT = 11    #inches
    # CUBE_WIDTH = 13     #inches
    # CUBE_LENGTH = 13    #inches

    HFOV = 64.0                  # horizontal angle of the field of view
    VFOV = 52.0                  # vertical angle of the field of view

    # create imaginary view plane on 3d coords to get height and width
    # place the view place on 3d coordinate plane 1.0 unit away from (0, 0) for simplicity
    VP_HALF_WIDTH = math.tan(math.radians(HFOV)/2.0)  # view plane 1/2 height
    VP_HALF_HEIGHT = math.tan(math.radians(VFOV)/2.0)  # view plane 1/2 width

    def __init__(self, calib_file):
        self.name = "linefinder"
        self.finder_id = 4.0   # id needs to be float! "id" is a reserved word.
        self.desired_camera = "floor"        # string with camera name
        self.exposure = 0

        # Color threshold values, in HSV space -- TODO: in 2019 server (yet to be created) make the low and high hsv limits
        # individual properties
        self.low_limit_hsv = numpy.array((25, 95, 95), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((75, 255, 255), dtype=numpy.uint8)

        # Allowed "error" in the perimeter when fitting using approxPolyDP (in quad_fit)
        self.approx_polydp_error = 0.015

        self.erode_kernel = numpy.ones((3, 3), numpy.uint8)
        self.erode_iterations = 0

        # some variables to save results for drawing
        self.hull_fit = None
        self.biggest_contour = None
        self.min_contour_area = 20
        #TODO: make sure min_contour_area is correct -- what if it doesn't see the whole line...
        #line is 18 inches long and 2 inches wide, area of 36

        self.cameraMatrix = None
        self.distortionMatrix = None
        if calib_file:
            with open(calib_file) as f:
                json_data = json.load(f)
                self.cameraMatrix = numpy.array(json_data["camera_matrix"])
                self.distortionMatrix = numpy.array(json_data["distortion"])

        self.tilt_angle = math.radians(0)  # camera mount angle (radians)
        self.camera_height = 0.0            # height of camera off the ground (inches)
        self.target_height = 0.0             # height of target off the ground (inches)

        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    @staticmethod
    def contour_center_width(contour):
        '''Find boundingRect of contour, but return center, width, and height'''

        x, y, w, h = cv2.boundingRect(contour)
        return (x + int(w / 2), y + int(h / 2)), (w, h)

    @staticmethod
    def quad_fit(contour, approx_dp_error):
        '''Simple polygon fit to contour with error related to perimeter'''

        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, approx_dp_error * peri, True)

    @staticmethod
    def sort_corners(cnrlist, check):
        '''Sort a list of corners -- if check == true then returns x sorted 1st, y sorted 2nd. Otherwise the opposite'''

        # recreate the list of corners to get rid of excess dimensions
        corners = []
        for c in cnrlist:
            corners.append(c[0].tolist())

        # sort the corners by x values (1st column) first and then by y values (2nd column)
        if check:
            return sorted(corners, key=lambda x: (x[0], x[1]))
        # y's first then x's
        else:
            return sorted(corners, key=lambda x: (x[1], x[0]))

    @staticmethod
    def split_xs_ys(corners):
        '''Split a list of corners into sorted lists of x and y values'''
        xs = []
        ys = []

        for i in range(len(corners)):
            xs.append(corners[i][0])
            ys.append(corners[i][1])
        #sort the lists highest to lowest
        xs.sort(reverse=True)
        ys.sort(reverse=True)
        return xs, ys

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        # clear out result variables each iteration
        angle = None
        distance = None
        self.center = None
        self.hull_fit = None
        self.biggest_contour = None

        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        if self.erode_iterations > 0:
            erode_frame = cv2.erode(threshold_frame, self.erode_kernel, iterations=self.erode_iterations)
        else:
            erode_frame = threshold_frame

        # OpenCV 3 returns 3 parameters!
        # Only need the contours variable
        _, contours, _ = cv2.findContours(erode_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for c in contours:
            center, widths = LineFinder2019.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:
                # TODO: use a simple class? Maybe use "attrs" package?
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)


def process_files(rrtarget_finder, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)
        result = rrtarget_finder.process_image(bgr_frame)
        #print(image_file, result[0], result[1], result[2], math.degrees(result[3]), math.degrees(result[4]))

        rrtarget_finder.prepare_output_image(bgr_frame)

        outfile = os.path.join(output_dir, os.path.basename(image_file))
        # print('{} -> {}'.format(image_file, outfile))
        cv2.imwrite(outfile, bgr_frame)

        # cv2.imshow("Window", bgr_frame)
        # q = cv2.waitKey(-1) & 0xFF
        # if q == ord('q'):
        #     break
    return


def time_processing(cube_processor, input_files):
    '''Time the processing of the test files'''

    from codetimer import CodeTimer
    from time import time

    startt = time()

    cnt = 0

    # Loop 100x over the files. This is needed to make it long enough
    #  to get reasonable statistics. If we have 100s of files, we could reduce this.
    # Need the total time to be many seconds so that the timing resolution is good.
    for _ in range(100):
        for image_file in input_files:
            with CodeTimer("Read Image"):
                bgr_frame = cv2.imread(image_file)

            with CodeTimer("Main Processing"):
                cube_processor.process_image(bgr_frame)

            cnt += 1

    deltat = time() - startt

    print("{0} frames in {1:.3f} seconds = {2:.2f} msec/call, {3:.2f} FPS".format(
        cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    CodeTimer.outputTimers()
    return


def main():
    '''Main routine'''
    import argparse

    parser = argparse.ArgumentParser(description='2018 cube finder')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    line_finder = LineFinder2019(args.calib_file)

    if args.output_dir is not None:
        process_files(line_finder, args.input_files, args.output_dir)
    elif args.time:
        time_processing(line_finder, args.input_files)

    return


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main()

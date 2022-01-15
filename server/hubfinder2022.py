#!/usr/bin/env python3

# Vision finder to find the retro-reflective target around the goal

import cv2
import numpy
import math

from genericfinder import GenericFinder, main


class HubFinder2022(GenericFinder):
    '''Find hub ring for 2022 game'''

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('goalfinder', camera='shooter', finder_id=1.0, exposure=1)

        # Color threshold values, in HSV space
        self.low_limit_hsv = numpy.array((65, 75, 75), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((100, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        self.hsv_frame = None
        self.threshold_frame = None

        # DEBUG values
        self.top_contours = None

        # output results
        self.target_contour = None

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.outer_corners = []
        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    @staticmethod
    def get_outer_corners(cnt):
        '''Return the outer four corners of a contour'''

        return GenericFinder.sort_corners(cnt)  # Sort by x value of cnr in increasing value

    def preallocate_arrays(self, shape):
        '''Pre-allocate work arrays to save time'''

        self.hsv_frame = numpy.empty(shape=shape, dtype=numpy.uint8)
        # threshold_fame is grey, so only 2 dimensions
        self.threshold_frame = numpy.empty(shape=shape[:2], dtype=numpy.uint8)
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''
        self.target_contour = None

        # DEBUG values; clear any values from previous image
        self.top_contours = None
        self.outer_corners = None

        shape = camera_frame.shape
        if self.hsv_frame is None or self.hsv_frame.shape != shape:
            self.preallocate_arrays(shape)

        self.hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV, dst=self.hsv_frame)
        self.threshold_frame = cv2.inRange(self.hsv_frame, self.low_limit_hsv, self.high_limit_hsv,
                                           dst=self.threshold_frame)

        # OpenCV 3 returns 3 parameters, OpenCV 4 returns 2!
        # Only need the contours variable
        res = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(res) == 2:
            contours = res[0]
        else:
            contours = res[1]

        # todo: process countours

        return [0.0, self.finder_id, 0.0, 0.0, 0.0, -1.0, -1.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        '''
        output_frame = input_frame.copy()

        if self.target_contour is not None:
            cv2.drawContours(output_frame, [self.target_contour.astype(int)], -1, (0, 0, 255), 2)

        if self.outer_corners is not None:
            for indx, cnr in enumerate(self.outer_corners):
                cv2.drawMarker(output_frame, tuple(cnr.astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)

        return output_frame
        '''

        #for now return the threshold frame
        return self.threshold_frame

# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(HubFinder2022)

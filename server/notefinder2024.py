#!/usr/bin/python3

# Vision finder to find NOTEs on the floor

import cv2
import numpy as np
import math
import logging

from ntprop_wrapper import ntproperty
from genericfinder import GenericFinder, main
from contour import Contour


class NoteFinder2024(GenericFinder):
    '''Find NOTE game piece for 2024'''

    # inches
    CAMERA_HEIGHT = 33
    CAMERA_ANGLE = math.radians(31)
    HUB_HEIGHT = 103

    # values for torture testing
    # self.low_limit_hsv = np.array((55, 5, 30), dtype=np.uint8)
    # self.high_limit_hsv = np.array((160, 255, 255), dtype=np.uint8)

    # Color threshold values, in HSV space
    hue_low_limit = ntproperty('/SmartDashboard/vision/notefinder/hue_low_limit', 50,
                               doc='Hue low limit for thresholding')
    hue_high_limit = ntproperty('/SmartDashboard/vision/notefinder/hue_high_limit', 90,
                                doc='Hue high limit for thresholding')

    saturation_low_limit = ntproperty('/SmartDashboard/vision/notefinder/saturation_low_limit', 110,
                                      doc='Saturation low limit for thresholding')
    saturation_high_limit = ntproperty('/SmartDashboard/vision/notefinder/saturation_high_limit', 255,
                                       doc='Saturation high limit for thresholding')

    value_low_limit = ntproperty('/SmartDashboard/vision/notefinder/value_low_limit', 110,
                                 doc='Value low limit for thresholding')
    value_high_limit = ntproperty('/SmartDashboard/vision/notefinder/value_high_limit', 255,
                                  doc='Value high limit for thresholding')

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('notefinder', camera='intake', finder_id=1.0, exposure=1)

        # Color threshold values, in HSV space
        # Somewhat faster to keep the arrays around and update the values inside
        self.low_limit_hsv = np.zeros((3), dtype=np.uint8)
        self.high_limit_hsv = np.zeros((3), dtype=np.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 40
        self.contour_max_area = 1000
        self.max_2nd_region_dist = 50
        self.contour_min_fill = 0.25

        self.hsv_frame = None
        self.threshold_frame = None

        self.top_contours = None
        self.target_contours = None
        self.filter_box = None
        self.circle = None
        self.test_locations = None

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        return

    @staticmethod
    def get_outer_corners(cnt):
        '''Return the outer four corners of a contour'''

        return GenericFinder.sort_corners(cnt)  # Sort by x value of cnr in increasing value

    def update_color_thresholds(self):
        self.low_limit_hsv[0] = int(self.hue_low_limit)
        self.low_limit_hsv[1] = int(self.saturation_low_limit)
        self.low_limit_hsv[2] = int(self.value_low_limit)
        self.high_limit_hsv[0] = int(self.hue_high_limit)
        self.high_limit_hsv[1] = int(self.saturation_high_limit)
        self.high_limit_hsv[2] = int(self.value_high_limit)
        return

    def preallocate_arrays(self, shape):
        '''Pre-allocate work arrays to save time'''

        self.hsv_frame = np.empty(shape=shape, dtype=np.uint8)
        # threshold_fame is grey, so only 2 dimensions
        self.threshold_frame = np.empty(shape=shape[:2], dtype=np.uint8)

        # translate FOV limits into pixels
        xp = math.tan(math.radians(self.MAX_ANGLE))
        self.minimum_x = self.cameraMatrix[0, 2] - xp * self.cameraMatrix[0, 0]
        self.maximum_x = self.cameraMatrix[0, 2] + xp * self.cameraMatrix[0, 0]

        angley = math.atan((self.HUB_HEIGHT - self.CAMERA_HEIGHT) / self.MIN_DISTANCE) - self.CAMERA_ANGLE
        yp = math.tan(angley)
        self.minimum_y = max(0, self.cameraMatrix[1, 2] - yp * self.cameraMatrix[1, 1])

        angley = math.atan((self.HUB_HEIGHT - self.CAMERA_HEIGHT) / self.MAX_DISTANCE) - self.CAMERA_ANGLE
        yp = math.tan(angley)
        self.maximum_y = self.cameraMatrix[1, 2] - yp * self.cameraMatrix[1, 1]

        # print('fov', self.minimum_x, self.maximum_x, self.minimum_y, self.maximum_y)
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        self.top_contours = None
        self.circle = None
        self.target_contours = None
        self.top_point = None
        self.test_locations = []

        shape = camera_frame.shape
        if self.hsv_frame is None or self.hsv_frame.shape != shape:
            self.preallocate_arrays(shape)

        self.hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV, dst=self.hsv_frame)

        self.update_color_thresholds()
        self.threshold_frame = cv2.inRange(self.hsv_frame, self.low_limit_hsv, self.high_limit_hsv,
                                           dst=self.threshold_frame)

        # OpenCV 3 returns 3 parameters, OpenCV 4 returns 2!
        # Only need the contours variable
        res = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = res[0]

        contour_list = []
        for c in contours:
            c_info = Contour(c)

            bb_area = c_info.bb_area
            if bb_area < self.contour_min_area or bb_area > self.contour_max_area:
                continue

            # ratio = c_info.bb_width / c_info.bb_height
            # # print('c', c_info.bb_center, bb_area, ratio, c_info.contour_area / bb_area)
            # if ratio < 0.8 or ratio > 4.0:
            #     continue

            # if c_info.contour_area / bb_area < self.contour_min_fill:
            #     continue

            # print('center', center, 'area', area)
            contour_list.append(c_info)

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c.contour_area, reverse=True)

        # DEBUG
        self.top_contours = [x.contour for x in contour_list]

        result = np.array([0.0, self.finder_id, 0.0, 0.0, 0.0, 0.0, 0.0])

        logging.info('notefinder2024: %s', result)
        return result

    def calculate_angle_distance(self, center):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the cameraMatrix from the calibration to convert to normalized coordinates'''

        # use the distortion and camera arrays to correct the location of the center point
        # got this from
        #  https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi

        ptlist = np.array([[center]])
        out_pt = cv2.undistortPoints(ptlist, self.cameraMatrix, self.distortionMatrix, P=self.cameraMatrix)
        undist_center = out_pt[0, 0]

        x_prime = (undist_center[0] - self.cameraMatrix[0, 2]) / self.cameraMatrix[0, 0]
        y_prime = -(undist_center[1] - self.cameraMatrix[1, 2]) / self.cameraMatrix[1, 1]

        # now have all pieces to convert to angle:
        ax = math.atan2(x_prime, 1.0)     # horizontal angle

        # naive expression
        # ay = math.atan2(y_prime, 1.0)     # vertical angle

        # corrected expression.
        # As horizontal angle gets larger, real vertical angle gets a little smaller
        ay = math.atan2(y_prime * math.cos(ax), 1.0)     # vertical angle
        # print("ax, ay", math.degrees(ax), math.degrees(ay))

        # now use the x and y angles to calculate the distance to the target:
        d = (self.HUB_HEIGHT - self.CAMERA_HEIGHT) / math.tan(self.CAMERA_ANGLE + ay)    # distance to the target

        return ax, d    # return horizontal angle and distance

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        if self.top_contours is not None:
            cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 1)

        if self.target_contours is not None:
            cv2.drawContours(output_frame, self.target_contours, -1, (255, 0, 255), 2)

        if self.top_point is not None:
            pt = tuple(self.top_point.astype(int))
            if(0 <= pt[0] < output_frame.shape[1] and 0 <= pt[1] < output_frame.shape[0]):
                cv2.drawMarker(output_frame, pt, (0, 0, 255), cv2.MARKER_CROSS, 20, 3)

        return output_frame


# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(NoteFinder2024)

#!/usr/bin/env python3

# Vision finder to find the retro-reflective target on the hub

import cv2
import numpy as np
import math

from genericfinder import GenericFinder, main


def find_circle(x1, y1, x2, y2, x3, y3):
    '''Compute a circle from 3 points
    Based on https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/'''

    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    sx13 = x1**2 - x3**2
    sy13 = y1**2 - y3**2
    sx21 = x2**2 - x1**2
    sy21 = y2**2 - y1**2

    f = (sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) / (2 * (y31 * x12 - y21 * x13))
    g = (sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) / (2 * (x31 * y12 - x21 * y13))

    c = -x1**2 - y1**2 - 2 * g * x1 - 2 * f * y1

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and radius r as r^2 = h^2 + k^2 - c
    r = math.sqrt(g**2 + f**2 - c)

    return (-g, -f), r


class FastFinder2022(GenericFinder):
    '''Find hub ring for 2022 game using a fast circle fit'''

    # inches
    CAMERA_HEIGHT = 33
    CAMERA_ANGLE = math.radians(31)
    HUB_HEIGHT = 103

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('hubfinder', camera='shooter', finder_id=1.0, exposure=1)

        # Color threshold values, in HSV space
        self.low_limit_hsv = np.array((65, 100, 80), dtype=np.uint8)
        self.high_limit_hsv = np.array((100, 255, 255), dtype=np.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        self.hsv_frame = None
        self.threshold_frame = None

        self.top_contours = None
        self.filter_box = None
        self.circle = None

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = np.array((hue_low, sat_low, val_low), dtype=np.uint8)
        self.high_limit_hsv = np.array((hue_high, sat_high, val_high), dtype=np.uint8)
        return

    @staticmethod
    def get_outer_corners(cnt):
        '''Return the outer four corners of a contour'''

        return GenericFinder.sort_corners(cnt)  # Sort by x value of cnr in increasing value

    def preallocate_arrays(self, shape):
        '''Pre-allocate work arrays to save time'''

        self.hsv_frame = np.empty(shape=shape, dtype=np.uint8)
        # threshold_fame is grey, so only 2 dimensions
        self.threshold_frame = np.empty(shape=shape[:2], dtype=np.uint8)
        return

    def process_contours(self, contour_list):
        sumx = 0.0
        sumy = 0.0
        sumw = 0.0
        pts = []
        for c in contour_list:
            img_moments = cv2.moments(c)
            w = img_moments['m00']
            sumx += img_moments['m10']
            sumy += img_moments['m01']
            sumw += w
            pts.append((w, img_moments['m10']/w, img_moments['m01']/w))

        xcentral = sumx/sumw
        if len(pts) < 3:
            return np.array((xcentral, sumy/sumw))

        # fit a circle to the top 3
        pts = sorted(pts, reverse=True)
        center, rad = find_circle(pts[0][1], pts[0][2], pts[1][1], pts[1][2], pts[2][1], pts[2][2])
        self.circle = (center, rad)
        # y axis increases down, so we want the smaller value
        # ycentral = center[1] - math.sqrt(rad**2 - (xcentral - center[0])**2)
        # return (xcentral, ycentral)
        return np.array((center[0], center[1] - rad))

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        self.top_contours = None
        self.filter_box = None
        self.circle = None

        shape = camera_frame.shape
        if self.hsv_frame is None or self.hsv_frame.shape != shape:
            self.preallocate_arrays(shape)

        self.hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV, dst=self.hsv_frame)

        self.threshold_frame = cv2.inRange(self.hsv_frame, self.low_limit_hsv, self.high_limit_hsv,
                                           dst=self.threshold_frame)

        # OpenCV 3 returns 3 parameters, OpenCV 4 returns 2!
        # Only need the contours variable
        res = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(res) == 2:
            contours = res[0]
        else:
            contours = res[1]

        max_area = 0
        filtered_contours = []
        size_cut = shape[0] * shape[1] * 5.0e-5
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            if area < size_cut:
                continue
            if area / (w * h) < 0.4:
                continue
            ratio = w / h
            if ratio < 1.2 or ratio > 4:
                continue

            filtered_contours.append(contour)

            if max_area < area:
                max_area = area
                cx, cy = x + w/2, y + h/2
                dx = 7 * w
                # the biggest contour should be at the top of the set, so use an asymmetric region in y
                # remember that y increases going down in the image
                self.filter_box = [cx - dx, cy - 3*h, cx + dx, cy + 6*h]

        # we defined a large box where all valid regions should be, based on the biggest contour
        # filter against that box
        self.top_contours = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x+w/2, y+h/2
            if (self.filter_box[0] <= cx <= self.filter_box[2]) and (self.filter_box[1] <= cy <= self.filter_box[3]):
                self.top_contours.append(contour)

        if self.top_contours:
            self.top_point = self.process_contours(self.top_contours)
        else:
            self.top_point = None

        if self.top_point is not None:
            angle, distance = self.calculate_angle_distance(self.top_point)

            result = np.array([1.0, self.finder_id, distance, angle, 0.0, 0.0, 0.0])
        else:
            result = np.array([0.0, self.finder_id, 0.0, 0.0, 0.0, 0.0, 0.0])

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
            cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 2)

        if self.top_point is not None:
            pt = tuple(self.top_point.astype(int))
            if(0 <= pt[0] < output_frame.shape[1] and 0 <= pt[1] < output_frame.shape[0]):
                cv2.drawMarker(output_frame, pt, (255, 0, 0), cv2.MARKER_CROSS, 15, 2)

        # if self.filter_box is not None:
        #     cv2.rectangle(output_frame, (int(self.filter_box[0]), int(self.filter_box[1])), (int(self.filter_box[2]), int(self.filter_box[3])),
        #                   (255, 0, 0), 2)

        # if self.circle:
        #     cv2.circle(output_frame, np.array(self.circle[0], dtype=np.int), int(self.circle[1]), (0, 255, 0), 2)

        # cv2.putText(output_frame, f"{round(self.processing_time[0],1)} {round(self.processing_time[1],1)} ms",
        #             (20, output_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), thickness=1)

        return output_frame


# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(FastFinder2022)

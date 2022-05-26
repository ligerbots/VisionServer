#!/usr/bin/env python3

# Vision finder to find the retro-reflective target on the hub

import cv2
import numpy as np
import math
import logging

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


class ContourInfo:
    '''Small class to cache info about a contour, to save repeating the calcs'''

    def __init__(self, contour):
        self.contour = contour
        c, w = GenericFinder.contour_center_width(contour)
        self.bb_center = c
        self.bb_width, self.bb_height = w
        self.bb_area = self.bb_width * self.bb_height

        # delay this calc
        self._moments = None
        return

    def __getattr__(self, attr):
        if attr in ('moments', 'centroid', 'con_area'):
            if self._moments is None:
                self._moments = cv2.moments(self.contour)

            if attr == 'moments':
                return self._moments
            elif attr == 'con_area':
                return self._moments['m00']
            else:
                w = self._moments['m00']
                return (self._moments['m10']/w, self._moments['m01']/w)

        raise AttributeError


class FastFinder2022(GenericFinder):
    '''Find hub ring for 2022 game using a fast circle fit'''

    # inches
    CAMERA_HEIGHT = 33
    CAMERA_ANGLE = math.radians(31)
    HUB_HEIGHT = 103

    # cuts on Field of View
    MIN_DISTANCE = 36.0
    MAX_DISTANCE = 270.0        # far launchpad, plus some
    MAX_ANGLE = 20.0

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('hubfinder', camera='shooter', finder_id=1.0, exposure=1)

        # Color threshold values, in HSV space
        self.low_limit_hsv = np.array((65, 35, 80), dtype=np.uint8)  # s was 100
        self.high_limit_hsv = np.array((100, 255, 255), dtype=np.uint8)

        # values for torture testing
        # self.low_limit_hsv = np.array((55, 5, 30), dtype=np.uint8)
        # self.high_limit_hsv = np.array((160, 255, 255), dtype=np.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 40
        self.contour_max_area = 1000
        self.max_2nd_region_dist = 50
        self.contour_min_fill = 0.25

        # FOV cuts. Do not know the values until we know the image size
        self.minimum_x = 0
        self.maximum_x = 10000
        self.minimum_y = 0
        self.maximum_y = 10000

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

    def process_contours(self, contour_list):
        if len(contour_list) < 3:
            return None

        # fit a circle to the top 3
        pts = [c.centroid for c in sorted(contour_list, key=lambda x: x.con_area, reverse=True)]

        center, rad = find_circle(pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1])
        self.circle = (center, rad)   # for debugging display

        # center of the fit should be *below* the points (y increases down)
        if center[1] < pts[0][1]:
            # print('circle is in wrong direction')
            return None

        if rad < 50.0 or rad > 400.0:
            # print(f'circle radius = {rad} is bad')
            return None

        # test the the found point is reasonably within the main part of the image
        if center[0] < self.minimum_x or center[0] > self.maximum_x:
            return None

        # y axis increases down, so we want the smaller value
        y = center[1] - rad
        # Test that the rough distance is sensible
        if y < self.minimum_y or y > self.maximum_y:
            return None

        return np.array((center[0], y))

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

        self.threshold_frame = cv2.inRange(self.hsv_frame, self.low_limit_hsv, self.high_limit_hsv,
                                           dst=self.threshold_frame)

        # OpenCV 3 returns 3 parameters, OpenCV 4 returns 2!
        # Only need the contours variable
        res = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(res) == 2:
            contours = res[0]
        else:
            contours = res[1]

        contour_list = []
        for c in contours:
            c_info = ContourInfo(c)

            bb_area = c_info.bb_area
            if bb_area < self.contour_min_area or bb_area > self.contour_max_area:
                continue

            ratio = c_info.bb_width / c_info.bb_height
            # print('c', c_info.bb_center, bb_area, ratio, c_info.con_area / bb_area)
            if ratio < 0.8 or ratio > 4.0:
                continue

            if c_info.con_area / bb_area < self.contour_min_fill:
                continue

            # print('center', center, 'area', area)
            contour_list.append(c_info)

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c.con_area, reverse=True)

        # DEBUG
        self.top_contours = [x.contour for x in contour_list]

        # try the 5 biggest contours as the anchor
        for candidate_index in range(min(5, len(contour_list))):
            # shape[0] is height, shape[1] is the width
            matched_contours = self.test_candidate_contour(contour_list[candidate_index:])
            if matched_contours is not None:
                # try to fit it to a circle
                self.top_point = self.process_contours(matched_contours)
                if self.top_point is not None:
                    self.target_contours = [c.contour for c in matched_contours]
                    break

        if self.top_point is not None:
            angle, distance = self.calculate_angle_distance(self.top_point)
            result = np.array([1.0, self.finder_id, distance, angle, 0.0, 0.0, 0.0])
        else:
            result = np.array([0.0, self.finder_id, 0.0, 0.0, 0.0, 0.0, 0.0])

        logging.info('fastfinder2022: %s' % result)
        return result

    def test_candidate_contour(self, contour_list):
        '''Given a contour as the candidate for the closest (unobscured) target region,
        try to find 1 or 2 other regions which makes up the other side of the target (due to the
        possible splitting of the target by cables lifting the elevator)

        In any test over the list of contours, start *after* cand_index. Those above it have
        been rejected.
        '''

        candidate = contour_list[0]
        # list of contour *indices* which might be the hub
        results = [0, ]

        # guess at the offset between regions
        delta_region = (1.8 * candidate.bb_width, 0.5 * candidate.bb_height)
        # print("delta_region =", delta_region)

        # Look for a matching regions to the left
        curr_index = 0
        while True:
            curr_index = self.find_adjacent_contour(contour_list[curr_index], delta_region, contour_list, results, -1)
            if curr_index is None:
                break

            results.append(curr_index)

        # Look for a matching regions to the right
        curr_index = 0
        while True:
            curr_index = self.find_adjacent_contour(contour_list[curr_index], delta_region, contour_list, results, 1)
            if curr_index is None:
                break

            results.append(curr_index)

        if len(results) > 2:
            return [contour_list[x] for x in results]
        return None

    def find_adjacent_contour(self, curr_ref, offset, contour_list, used_contours, direction):
        '''Look for a matching contour to one side of current candidate'''

        center = curr_ref.centroid
        test_loc = (center[0] + direction * offset[0], center[1] + offset[1])
        self.test_locations.append((center, test_loc))
        # print('test_loc', test_loc)

        minD = 100000
        minIndex = None
        for cindex in range(1, len(contour_list)):
            # skip anything already sed
            if cindex in used_contours:
                continue

            # negative is outside. I want the other sign
            dist = -cv2.pointPolygonTest(contour_list[cindex].contour, test_loc, measureDist=True)
            # print('testing', direction, cindex, test_loc, 'against', contour_list[cindex].bb_center, '  --> distance =', dist)
            if dist < self.max_2nd_region_dist and dist < minD:
                # print("  matched. dist =", dist)
                minD = dist
                minIndex = cindex
            if minD < 0:
                break

        # print('minD', minD, ' len(results)', len(results))
        return minIndex

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

        # if self.top_contours is not None:
        #     cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 1)

        if self.target_contours is not None:
            cv2.drawContours(output_frame, self.target_contours, -1, (255, 0, 255), 2)

        if self.top_point is not None:
            pt = tuple(self.top_point.astype(int))
            if(0 <= pt[0] < output_frame.shape[1] and 0 <= pt[1] < output_frame.shape[0]):
                cv2.drawMarker(output_frame, pt, (0, 0, 255), cv2.MARKER_CROSS, 20, 3)

        # if self.test_locations:
        #     for tloc in self.test_locations:
        #         cv2.line(output_frame, (int(tloc[0][0]), int(tloc[0][1])), (int(tloc[1][0]), int(tloc[1][1])), (0, 255, 255), 2)
        #         # cv2.drawMarker(output_frame, tuple(pp), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)

        # if self.filter_box is not None:
        #     cv2.rectangle(output_frame, (int(self.filter_box[0]), int(self.filter_box[1])), (int(self.filter_box[2]), int(self.filter_box[3])),
        #                   (255, 0, 0), 2)

        # if self.circle:
        #     cv2.circle(output_frame, np.array(self.circle[0], dtype=np.int), int(self.circle[1]), (0, 255, 0), 1)

        cv2.line(output_frame, (int(self.minimum_x), 0), (int(self.minimum_x), input_frame.shape[0]), (255, 0, 0), 1)
        cv2.line(output_frame, (int(self.maximum_x), 0), (int(self.maximum_x), input_frame.shape[0]), (255, 0, 0), 1)
        cv2.line(output_frame, (0, int(self.minimum_y)), (input_frame.shape[1], int(self.minimum_y)), (255, 0, 0), 1)
        cv2.line(output_frame, (0, int(self.maximum_y)), (input_frame.shape[1], int(self.maximum_y)), (255, 0, 0), 1)

        return output_frame


# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(FastFinder2022)

#!/usr/bin/env python3

# Vision finder to find the retro-reflective target around the goal

import cv2
import numpy
import json
import math

from genericfinder import GenericFinder, main


class GoalFinder2020(GenericFinder):
    '''Find high goal target for Infinite Recharge 2020'''

    # real world dimensions of the goal target
    # These are the full dimensions around both strips
    TARGET_STRIP_LENGTH = 19.625    # inches
    TARGET_HEIGHT = 17.0            # inches@!
    TARGET_TOP_WIDTH = 39.25        # inches
    TARGET_BOTTOM_WIDTH = TARGET_TOP_WIDTH - 2*TARGET_STRIP_LENGTH*math.cos(math.radians(60))

    # [0, 0] is center of the quadrilateral drawn around the high goal target
    # [top_left, bottom_left, bottom_right, top_right]
    real_world_coordinates = numpy.array([
        [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0],
        [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
        [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
        [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0]
    ])

    def __init__(self, calib_file):
        super().__init__('goalfinder', camera='front', finder_id=1.0, exposure=1)

        # Color threshold values, in HSV space
        self.low_limit_hsv = numpy.array((65, 75, 75), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((100, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        # candidate cut thresholds
        self.min_dim_ratio = 1
        self.max_area_ratio = 0.25

        # camera mount angle (radians)
        # NOTE: not sure if this should be positive or negative
        self.tilt_angle = math.radians(-7.5)

        self.hsv_frame = None
        self.threshold_frame = None

        # DEBUG values
        self.top_contours = None

        # output results
        self.target_contour = None

        if calib_file:
            with open(calib_file) as f:
                json_data = json.load(f)
                self.cameraMatrix = numpy.array(json_data["camera_matrix"])
                self.distortionMatrix = numpy.array(json_data["distortion"])

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

        # OpenCV 3 returns 3 parameters!
        # Only need the contours variable
        _, contours, _ = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for c in contours:
            center, widths = GoalFinder2020.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:        # area cut
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        # DEBUG
        self.top_contours = [x['contour'] for x in contour_list]

        # try only the 5 biggest regions at most
        target_center = None
        for candidate_index in range(min(5, len(contour_list))):
            self.target_contour = self.test_candidate_contour(contour_list[candidate_index], shape)
            if self.target_contour is not None:
                target_center = contour_list[candidate_index]['center']
                break

        if self.target_contour is not None:
            # The target was found. Convert to real world co-ordinates.

            # need the corners in proper sorted order, and as floats
            self.outer_corners = GenericFinder.sort_corners(self.target_contour, target_center).astype(numpy.float)

            # print("Outside corners: ", self.outer_corners)
            # print("Real World target_coords: ", self.real_world_coordinates)

            retval, rvec, tvec = cv2.solvePnP(self.real_world_coordinates, self.outer_corners,
                                              self.cameraMatrix, self.distortionMatrix)
            if retval:
                result = [1.0, self.finder_id, ]
                result.extend(self.compute_output_values(rvec, tvec))
                result.extend((-1.0, -1.0))
                return result

        # no target found. Return "failure"
        return [0.0, self.finder_id, 0.0, 0.0, 0.0, 0.0, 0.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        if self.top_contours:
            cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 1)

        if self.target_contour is not None:
            cv2.drawContours(output_frame, [self.target_contour.astype(int)], -1, (255, 0, 0), 1)

        if self.outer_corners is not None:
            for indx, cnr in enumerate(self.outer_corners):
                cv2.drawMarker(output_frame, tuple(cnr.astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 5, 1)
                # cv2.putText(output_frame, str(indx), tuple(cnr.astype(int)), 0, .5, (255, 255, 255))

        return output_frame

    def test_candidate_contour(self, candidate, shape):
        '''Determine the true target contour out of potential candidates'''

        cand_width = candidate['widths'][0]
        cand_height = candidate['widths'][1]

        cand_dim_ratio = cand_width / cand_height
        if cand_dim_ratio < self.min_dim_ratio:
            return None
        cand_area_ratio = cv2.contourArea(candidate["contour"]) / (cand_width * cand_height)
        if cand_area_ratio > self.max_area_ratio:
            return None

        hull = cv2.convexHull(candidate['contour'])
        contour = self.quad_fit(hull)

        if contour is not None and len(contour) == 4:
            return contour

        return None

    def compute_output_values(self, rvec, tvec):
        '''Compute the necessary output distance and angles'''

        # The tilt angle only affects the distance and angle1 calcs

        x = tvec[0][0]
        z = math.sin(self.tilt_angle) * tvec[1][0] + math.cos(self.tilt_angle) * tvec[2][0]

        # distance in the horizontal plane between camera and target
        distance = math.sqrt(x**2 + z**2)

        # horizontal angle between camera center line and target
        angle1 = math.atan2(x, z)

        rot, _ = cv2.Rodrigues(rvec)
        rot_inv = rot.transpose()
        pzero_world = numpy.matmul(rot_inv, -tvec)
        angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])

        return distance, angle1, angle2


# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(GoalFinder2020)

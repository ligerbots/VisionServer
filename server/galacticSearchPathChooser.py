#!/usr/bin/env python3

import cv2
import numpy
import math
import json
from genericfinder import GenericFinder, main


class GalacticSearchPathChooser(GenericFinder):
    '''Ball finder for Infinite Recharge 2020'''

    BALL_DIAMETER = 7            # inches

    HFOV = 64.0                  # horizontal angle of the field of view
    VFOV = 52.0                  # vertical angle of the field of view

    # create imaginary view plane on 3d coords to get height and width
    # place the view place on 3d coordinate plane 1.0 unit away from (0, 0) for simplicity
    VP_HALF_WIDTH = math.tan(math.radians(HFOV)/2.0)  # view plane 1/2 height
    VP_HALF_HEIGHT = math.tan(math.radians(VFOV)/2.0)  # view plane 1/2 width

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('ballfinder', camera='intake', finder_id=2.0, exposure=0)

        # individual properties
        self.low_limit_hsv = numpy.array((23, 128, 161), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((27, 238, 255), dtype=numpy.uint8)

        self.paths_points={
            "a-blue": numpy.array([[690.6518792359827, 505.81792975970427], [259.26403508771926, 508.4675438596491], [391.60912698412693, 485.3095238095238]]),
            "a-red": numpy.array([[397.7219017654256, 627.8941327273786], [578.3780487804878, 531.2745934959349], [89.57725402169847, 523.7800224466891]]),
            "b-blue": numpy.array([[544.5913218970736, 512.0871173898419], [290.23853211009174, 494.0132517838939], [480.55112474437624, 473.2065439672801]]),
            "b-red": numpy.array([[59.9161622079021, 636.4458113649678], [587.659868729489, 534.20839193624], [269.375166002656, 507.242363877822]]),
        }
        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        # self.erode_kernel = numpy.ones((3, 3), numpy.uint8)
        # self.erode_iterations = 0

        # some variables to save results for drawing
        self.top_contours = []
        self.found_contours = []
        self.center_points = []

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.tilt_angle = math.radians(-20.0)  # camera mount angle (radians)
        self.camera_height = 20.5              # height of camera off the ground (inches)
        self.target_height = 3.5               # height of target off the ground (inches)

        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    def get_ball_values(self, center, shape):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the FOV numbers and the default center to convert to normalized coordinates'''

        # center is in pixel coordinates, 0,0 is the upper-left, positive down and to the right
        # (nx,ny) = normalized pixel coordinates, 0,0 is the center, positive right and up
        # WARNING: shape is (h, w, nbytes) not (w,h,...)
        image_w = shape[1] / 2.0
        image_h = shape[0] / 2.0

        # NOTE: the 0.5 is to place the location in the center of the pixel
        # print("center", center, "shape", shape)
        nx = (center[0] - image_w + 0.5) / image_w
        ny = (image_h - 0.5 - center[1]) / image_h

        # convert normal pixel coords to pixel coords
        x = BallFinder2020.VP_HALF_WIDTH * nx
        y = BallFinder2020.VP_HALF_HEIGHT * ny
        # print("values", center[0], center[1], nx, ny, x, y)

        # now have all pieces to convert to angle:
        ax = math.atan2(x, 1.0)     # horizontal angle

        # naive expression
        # ay = math.atan2(y, 1.0)     # vertical angle

        # corrected expression.
        # As horizontal angle gets larger, real vertical angle gets a little smaller
        ay = math.atan2(y * math.cos(ax), 1.0)     # vertical angle
        # print("ax, ay", math.degrees(ax), math.degrees(ay))

        # now use the x and y angles to calculate the distance to the target:
        d = (self.target_height - self.camera_height) / math.tan(self.tilt_angle + ay)    # distance to the target

        return ax, d    # return horizontal angle and distance

    def get_ball_values_calib(self, center):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the cameraMatrix from the calibration to convert to normalized coordinates'''

        # use the distortion and camera arrays to correct the location of the center point
        # got this from
        #  https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi

        ptlist = numpy.array([[center]])
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
        d = (self.target_height - self.camera_height) / math.tan(self.tilt_angle + ay)    # distance to the target

        return ax, d    # return horizontal angle and distance

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        # clear out member result variables
        self.center_points = []
        self.top_contours = []
        self.found_contours = []

        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        image_height, image_width, _=hsv_frame.shape

        res = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(res) == 2:
            contours = res[0]
        else:
            contours = res[1]

        contour_list = []
        for c in contours:
            center, widths = self.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        self.top_contours = [cnt['contour'] for cnt in contour_list]
        # Sort the list of contours from largest area to smallest area
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        for cnt in contour_list:
            result_cnt = self.test_candidate_contour(cnt)

            if result_cnt is not None:
                self.found_contours.append(result_cnt)

                img_moments = cv2.moments(result_cnt)
                center = numpy.array((img_moments['m10']/img_moments['m00'], img_moments['m01']/img_moments['m00']))

                if(image_height-center[1]<377): # filter out the wheels near the bottom of the robot
                    continue

                self.center_points.append(center)

                if len(self.center_points) >= 3:
                    break

        print("Found points: "+json.dumps([list(itm) for itm in self.center_points]))
        # done with the contours. Pick two centers to return

        # return values: (success, finder_id, distance1, robot_angle1, target_angle1, distance2, robot_angle2)
        if len(self.center_points) < 3:
            return (0.0, self.finder_id, 0.0, 0.0, 0.0, 0.0)

        path_score_min=None
        path_score_min_name=None

        for path_name in self.paths_points:
            target_positions=self.paths_points[path_name]
            diff_squared_total=0
            for point in self.center_points:
                diffs=[numpy.linalg.norm(point-target_position) for target_position in target_positions]
                min_diff=min(diffs)
                diff_squared_total+=min_diff*min_diff
            if(path_score_min_name is None or diff_squared_total<path_score_min):
                path_score_min=diff_squared_total
                path_score_min_name=path_name

        print("Found path type: "+path_score_min_name)
        return (1.0, self.finder_id, 0, 0, 0.0, 0, 0)

    def test_candidate_contour(self, contour_entry):
        cnt = contour_entry['contour']

        # contours not clean, don't filter anything for now

        # real_area = cv2.contourArea(cnt)
        # print('areas:', real_area, contour_entry['area'], real_area / contour_entry['area'])
        # print("ratio"+str(contour_entry['widths'][1] / contour_entry['widths'][0] ))

        # contour_entry['widths'][1] is the height
        # contour_entry['widths'][0] is the width
        # ratio = contour_entry['widths'][1] / contour_entry['widths'][0]
        # TODO if balls cut out at bottom of screen it returns none,
        #   so might want to change the lower value depending on camera location
        #if ratio < 0.8 or ratio > 3.1:
        #    print("not ratio")
        #    return None

        #ratio = cv2.contourArea(cnt) / contour_entry['area']
        #if ratio < (math.pi / 4) - 0.1 or ratio > (math.pi / 4) + 0.1:  # TODO refine the 0.1 error range
        #    print("not ratio2")
        #    return None

        return cnt

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw found contours.'''
        output_frame = input_frame.copy()
        #Draw the contour on the image
        if self.top_contours:
             cv2.drawContours(output_frame, self.top_contours, -1, (255, 0, 0), 1)

        if self.found_contours:
            cv2.drawContours(output_frame, self.found_contours, -1, (0, 0, 255), 2)

        for cnt in self.center_points:
            cv2.drawMarker(output_frame, tuple(cnt.astype(int)), (255, 125, 0), cv2.MARKER_CROSS, 10, 2)

        return output_frame


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main(GalacticSearchPathChooser)

#!/usr/bin/env python3

import cv2
import numpy
from genericfinder import GenericFinder, main


class GalacticSearchPathChooser(GenericFinder):
    '''Path chooser for galactic search skill challenge'''

    RED_CLOSE_Y = 440  # max y value of close ball for path to be red

    def __init__(self, calib_matrix=None, dist_matrix=None, result_ntproperty=""):
        super().__init__('galactic_search_path_chooser', camera='intake', finder_id=6.0, exposure=0)

        # individual properties
        self.low_limit_hsv = numpy.array((20, 125, 170), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((40, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        # some variables to save results for drawing
        self.top_contours = []
        self.found_contours = []
        self.center_points = []

        self.result_ntproperty = result_ntproperty  # color (red or blue)

        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = numpy.array((hue_low, sat_low, val_low), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((hue_high, sat_high, val_high), dtype=numpy.uint8)
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        # clear out member result variables
        self.center_points = []
        self.top_contours = []
        self.found_contours = []

        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        image_height = hsv_frame.shape[0]

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
                img_moments = cv2.moments(result_cnt)
                center = numpy.array((img_moments['m10']/img_moments['m00'], img_moments['m01']/img_moments['m00']))

                # TODO: this could be done faster on the contour center (inside test_candidate_contour)
                #    - saves calcing the moments
                if image_height - center[1] < 237:  # filter out the wheels near the bottom of the robot
                    continue
                if image_height - center[1] > 640:  # filter out the ladder
                    continue

                self.found_contours.append(result_cnt)
                self.center_points.append(center)

                if len(self.center_points) >= 3:
                    break

        if len(self.center_points) < 2:
            self.result_ntproperty = "Fail - Not enough balls"
            return (0.0, self.finder_id, 0, 0, 0.0, 0, 0)

        # code uses coordinates with 0,0 at bottom left
        center_points_mapped = [(x, image_height-y) for (x, y) in self.center_points]
        # sort from bottom to top
        center_points_mapped.sort(key=lambda p: p[1])

        chosen_path = None

        if center_points_mapped[0][1] < self.RED_CLOSE_Y:  # is the path red?
            b1_b2_distance = numpy.linalg.norm(numpy.array(center_points_mapped[0]) - center_points_mapped[1])
            if b1_b2_distance > 160:
                chosen_path = "a-red"
            else:
                chosen_path = "b-red"
        else:  # is the path blue?
            if center_points_mapped[0][0] - 50 > center_points_mapped[-1][0]:
                chosen_path = "a-blue"
            else:
                chosen_path = "b-blue"

        if chosen_path is None:
            return (0.0, self.finder_id, 0, 0, 0.0, 0, 0)

        print("picked ", chosen_path)
        self.result_ntproperty = chosen_path

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
        # if ratio < 0.8 or ratio > 3.1:
        #    print("not ratio")
        #    return None

        # ratio = cv2.contourArea(cnt) / contour_entry['area']
        # if ratio < (math.pi / 4) - 0.1 or ratio > (math.pi / 4) + 0.1:  # TODO refine the 0.1 error range
        #    print("not ratio2")
        #    return None

        return cnt

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw found contours.'''
        output_frame = input_frame.copy()

        # Draw the contour on the image
        if self.top_contours:
            cv2.drawContours(output_frame, self.top_contours, -1, (255, 0, 0), 1)

        if self.found_contours:
            cv2.drawContours(output_frame, self.found_contours, -1, (0, 0, 255), 2)

        for cnt in self.center_points:
            cv2.drawMarker(output_frame, tuple(cnt.astype(int)), (255, 125, 0), cv2.MARKER_CROSS, 10, 2)

        cv2.putText(output_frame, self.result_ntproperty, (60, input_frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

        return output_frame


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main(GalacticSearchPathChooser)

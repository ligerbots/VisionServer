#!/usr/bin/env python3

# Vision finder to find the retro-reflective target around the goal

import cv2
import numpy as np
import math

from genericfinder import GenericFinder, main
import matplotlib.pyplot as plt



class HubFinder2022(GenericFinder):
    '''Find hub ring for 2022 game'''

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('goalfinder', camera='shooter', finder_id=1.0, exposure=1)

        # Color threshold values, in HSV space
        self.low_limit_hsv = np.array((75, 210, 210), dtype=np.uint8)
        self.high_limit_hsv = np.array((90, 255, 255), dtype=np.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        self.hsv_frame = None
        self.threshold_frame = None
        self.ellipse_coeffs = None

        # DEBUG values
        self.top_contours = None
        self.upper_points = None
        # output results
        self.target_contour = None

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.outer_corners = []
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

    def fit_ABCDE(self, points):
        '''
        solving Ax^2 + Bxy + Cy^2 + Dx + Ey = 1
        given: points (x1, y1), (x2, y2), .. (xn, yn)
        NM = P
        '''
        N = np.empty((len(points),5))
        for (i, (x,y)) in enumerate(points):
            N[i] = [x**2, x*y, y**2, x, y]
        M = np.linalg.pinv(N) @ np.ones((len(points),))
        return M

    def ransac_fit_ABCDE(self, points, sample_size=6, attempts = 20):
        best_error = None
        best_coeffs = None
        N = np.empty((len(points),5))
        for (i, (x,y)) in enumerate(points):
            N[i] = [x**2, x*y, y**2, x, y]

        for _ in range(attempts):
            selected_index = np.random.choice(len(points), sample_size, replace=False)
            selected_points = points[selected_index]
            coeffs = self.fit_ABCDE(self.upper_points)
            error = N@coeffs-1
            total_error = np.sum(np.abs(error))
            if(best_error is None or total_error<best_error):
                best_error = total_error
                best_coeffs = coeffs
        print(best_coeffs)
        return(best_coeffs)

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
        #plt.imshow(self.hsv_frame)
        #plt.show()

        self.threshold_frame = cv2.inRange(self.hsv_frame, self.low_limit_hsv, self.high_limit_hsv,
                                           dst=self.threshold_frame)


        # OpenCV 3 returns 3 parameters, OpenCV 4 returns 2!
        # Only need the contours variable
        res = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(res) == 2:
            contours = res[0]
        else:
            contours = res[1]

        self.top_contours  = contours


        contour_centers = []

        self.upper_points = []

        leftvec = [-1,0]
        for contour in contours:
            prev = contour[-1][0]
            for [point] in contour:
                vec = point - prev

                if(np.dot(vec,leftvec)>0):
                    self.upper_points.append(point)
                prev = point
        self.ellipse_coeffs = self.ransac_fit_ABCDE(np.array(self.upper_points))
        '''
        # todo: process countours
        '''
        return [0.0, self.finder_id, 0.0, 0.0, 0.0, -1.0, -1.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''


        output_frame = input_frame.copy()

        if self.top_contours is not None:
            cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 2)


        '''
        if self.target_contour is not None:
            cv2.drawContours(output_frame, [self.target_contour.astype(int)], -1, (0, 0, 255), 2)

        if self.outer_corners is not None:
            for indx, cnr in enumerate(self.outer_corners):
                cv2.drawMarker(output_frame, tuple(cnr.astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)

        return output_frame
        '''

        plt.imshow(output_frame)
        plt.scatter(*np.array(self.upper_points).T)

        x = np.linspace(0, output_frame.shape[1], 400)
        y = np.linspace(0, output_frame.shape[0], 400)
        x, y = np.meshgrid(x, y)

        a,b,c,d,e=self.ellipse_coeffs
        f = -1
        plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0])
        plt.show()

        #for now return the threshold frame
        return output_frame

# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(HubFinder2022)

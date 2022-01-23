#!/usr/bin/env python3

# Vision finder to find the retro-reflective target around the goal

import cv2
import numpy as np
import scipy # for numba blas
import numba as nb
from timeit import default_timer as timer

import math

from genericfinder import GenericFinder, main
import matplotlib.pyplot as plt

@nb.njit(nb.float64[:](nb.float64[:,:]))
def fit_ABCDE(points):
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

@nb.njit(nb.float64[:](nb.float64[:]))
def canonical(M):
    A, B, C, D, E = M
    F = -1
    #https://en.wikipedia.org/wiki/Ellipse
    deter = B**2 - 4 * A * C
    if(B != 0):
        theta = np.arctan(1/B*(C-A-np.sqrt((A-C)**2 + B**2)))
    elif(A<C):
        theta = 0
    else:
        theta = math.pi/2
    x_c = (2*C*D - B*E) / deter
    y_c = (2*A*E - B*D) / deter

    a = - np.sqrt(2*(A*E*E + C*D*D - B*D*E + deter * F)*((A+C)+np.sqrt((A-C)**2+B**2))) / deter
    b = - np.sqrt(2*(A*E*E + C*D*D - B*D*E + deter * F)*((A+C)-np.sqrt((A-C)**2+B**2))) / deter
    return(np.array([a ,b, x_c, y_c, theta], dtype = np.float64))


@nb.njit(nb.float64(nb.float64[:],nb.float64[:], nb.float64, nb.float64))
def check_point(point, canonical, max_distance, error):
    a, b, x_c, y_c, theta = canonical
    x, y = point

    if(a>b):
        up_side = y - y_c < np.tan(theta) * (x - x_c)
    else:
        up_side = y - y_c < np.tan(theta + math.pi/2) * (x - x_c)
    if(not up_side):
        return error
    tx =  (x - x_c)*np.cos(theta) + (y - y_c)*np.sin(theta)
    ty = -(x - x_c)*np.sin(theta) + (y - y_c)*np.cos(theta)

    m = tx/a
    n = ty/b
    mn_d = np.sqrt(m**2 + n**2)

    stx = a * m / mn_d
    sty = b * n / mn_d
    d = np.sqrt((tx - stx)**2 + (ty - sty)**2)
    if(d > max_distance):
        return error
    else:
        return 0

@nb.njit(nb.float64[:](nb.float64[:,:], nb.int64, nb.int64, nb.float64, nb.float64))
def ransac_fit_ABCDE(points, sample_size, attempts, max_distance, error):
    best_error = -1
    best_coeffs = np.zeros((5,))
    N = np.empty((len(points),5))
    for (i, (x,y)) in enumerate(points):
        N[i] = [x**2, x*y, y**2, x, y]

    for _ in range(attempts):
        selected_index = np.random.choice(len(points), sample_size, replace=False)
        selected_points = points[selected_index]

        M = fit_ABCDE(selected_points)

        deter = M[1]**2 - 4 * M[0] * M[2]
        if(deter>=0):
            continue

        canonical_M = canonical(M)
        if(np.abs(canonical_M[0])<0.0001 or np.abs(canonical_M[2])<0.0001):
            continue
        total_error = 0
        for point in points:
            total_error += check_point(point, canonical_M, max_distance, error)

        if(best_error == -1 or total_error < best_error):
            best_error = total_error
            best_coeffs = M

    return(best_coeffs)

class HubFinder2022(GenericFinder):
    '''Find hub ring for 2022 game'''

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('goalfinder', camera='shooter', finder_id=1.0, exposure=1)

        # Color threshold values, in HSV space
        self.low_limit_hsv = np.array((75, 120, 170), dtype=np.uint8)
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
        if(len(self.upper_points)< 10):
                return [0.0, self.finder_id, 0.0, 0.0, 0.0, -1.0, -1.0]

        start = timer()
        self.ellipse_coeffs = ransac_fit_ABCDE(np.array(self.upper_points, dtype=np.float64), 5, 50, 5.0, 1)
        end = timer()

        print((end - start)*1000)

        return [0.0, self.finder_id, 0.0, 0.0, 0.0, -1.0, -1.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''


        #output_frame = self.hsv_frame.copy()
        output_frame = input_frame.copy()

        if self.top_contours is not None:
            cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 2)

        a, b, x_c, y_c, theta = canonical(self.ellipse_coeffs)

        cv2.ellipse(output_frame, (int(x_c), int(y_c)), (int(a),int(b)),
           math.degrees(theta), 0, 360, (255,0,0), 3)

        '''
        if self.target_contour is not None:
            cv2.drawContours(output_frame, [self.target_contour.astype(int)], -1, (0, 0, 255), 2)

        if self.outer_corners is not None:
            for indx, cnr in enumerate(self.outer_corners):
                cv2.drawMarker(output_frame, tuple(cnr.astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)

        return output_frame
        '''
        '''
        plt.imshow(output_frame)
        plt.scatter(*np.array(self.upper_points).T)
        print(self.ellipse_coeffs)
        canonical_M = canonical(self.ellipse_coeffs)
        for point in self.upper_points:
            plt.annotate(str(check_point(point.astype(np.float64), canonical_M, 5, 1)), point, color = "red")

        x = np.linspace(0, output_frame.shape[1], 400)
        y = np.linspace(0, output_frame.shape[0], 400)
        x, y = np.meshgrid(x, y)

        a,b,c,d,e=self.ellipse_coeffs
        f = -1
        plt.contour(x, y,(a*x**2 + b*x*y + c*y**2 + d*x + e*y + f), [0])
        plt.show()
        '''

        #for now return the threshold frame
        return output_frame

# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(HubFinder2022)

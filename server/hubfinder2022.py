#!/usr/bin/env python3

# Vision finder to find the retro-reflective target on the hub

import cv2
import numpy as np
import numba as nb
from timeit import default_timer as timer
import math
from genericfinder import GenericFinder, main
# import matplotlib.pyplot as plt
from scipy import optimize

@nb.njit(nb.float32[:](nb.float32, nb.float32, nb.float32, nb.float32, nb.float32))
def transform_hubplane(x_prime, y_prime, sin_theta, cos_theta, h):
    '''
    Transform a point in camera space to the hub plane (tilt of camera = theta, height of hub = h)
    Sine and cosine of theta cashed
    '''
    return(np.array([x_prime*h/(sin_theta + y_prime * cos_theta), (cos_theta - y_prime * sin_theta) * h / (sin_theta + y_prime * cos_theta)], dtype = np.float32))

@nb.njit(nb.float32[:](nb.float32, nb.float32, nb.float32, nb.float32, nb.float32))
def transform_camera(x, y, sin_theta, cos_theta, h):
    '''
    Transform a point in the hub plane to camera space (tilt of camera = theta, height of hub = h)
    Sine and cosine of theta cashed
    '''
    return(np.array([x/(y*cos_theta + h*sin_theta), (-y*sin_theta + h*cos_theta) / (y*cos_theta + h*sin_theta)], dtype = np.float32))


@nb.njit(nb.float32[:,:](nb.float32[:,:], nb.float32, nb.float32))
def transform_hubplane_many(pts, theta, h):
    '''
    Transform many points with transform_hubplane
    '''
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    res = np.empty_like(pts)
    for i in range(len(pts)):
        res[i] = transform_hubplane(pts[i][0],pts[i][1],sin_theta, cos_theta, h)
    return(res)

@nb.njit(nb.float32[:,:](nb.float32[:,:], nb.float32, nb.float32))
def transform_camera_many(pts, theta, h):
    '''
    Transform many points with transform_camera
    '''
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    res = np.empty_like(pts)
    for i in range(len(pts)):
        res[i] = transform_camera(pts[i][0],pts[i][1],sin_theta, cos_theta, h)
    return(res)

@nb.njit(nb.float32[:](nb.float32[:], nb.float32[:]))
def redistort(pt, dist_matrix):
    '''
    Redistort a point that was undistorted with cv2.undistortPoints()
    '''
    x, y = pt
    k1, k2, p1, p2, k3 = dist_matrix
    r2 = x*x + y*y

    x_distorted = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    y_distorted =  y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    x_distorted += (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distorted += (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);
    return(np.array([x_distorted, y_distorted], np.float32))

@nb.njit(nb.float32[:,:](nb.float32[:,:], nb.float32[:]))
def redistort_many(pts, dist_matrix):
    '''
    Redistort many points with redistort
    '''
    res = np.empty_like(pts)
    for i in range(len(pts)):
        res[i] = redistort(pts[i],dist_matrix)

    return(res)

def test():
    '''
    Ensure transform_hubplane_many is the inverse of transform_camera_many
    Also forces numba to compile the functions
    '''
    x = np.array([[1.,2.],[3.,4.]], dtype = np.float32)
    camera = transform_hubplane_many(x, 1., 1.)
    hub = transform_camera_many(camera, 1., 1.)
    if(not np.allclose(x, hub)):
        print("transform error:",x,hub)
test()

@nb.njit(nb.types.Array(nb.float32, 2, "C")(nb.types.ListType(nb.types.Array(nb.int32, 3, "C")), nb.int32, nb.int32))
def compute_midline(contours, width, height):
    """
    Takes an array of contours and computes a midline:
            **************
     *******              *********
    *                     _________**
    * ----____________----         -*
    *                     ***********
    ******            *****
          ************
    Midline will have at most 5 points
    contour must be computed with CHAIN_APPROX_NONE
    """
    top_y = np.empty((width,), dtype=np.int32)
    bottom_y = np.empty((width,), dtype=np.int32)

    midline_points = nb.typed.List()
    for contour in contours:
        min_x = np.amin(contour[:, 0, 0])
        max_x = np.amax(contour[:, 0, 0])
        top_y[min_x:max_x+1] = -1
        bottom_y[min_x:max_x+1] = -1

        leftvec = np.array([-1.0, 0.0])
        for next_i in range(len(contour)):
            prev_p, this_p, next_p = contour[0, next_i-2], contour[0, next_i-1], contour[0, next_i]
            vec_p, vec_n = this_p - prev_p + 0.0, next_p - this_p + 0.0
            d_p, d_n = np.dot(vec_p, leftvec), np.dot(vec_n, leftvec)
            if(d_p > 0 and d_n > 0):
                top_y[this_p[0]] = this_p[1]
            elif(d_p < 0 and d_n < 0):
                bottom_y[this_p[0]] = this_p[1]

        for x in np.linspace(min_x, max_x, 5).astype(np.int32):
            if(top_y[x] != -1 and bottom_y[x] != -1):
                midline_points.append(np.array([x, (top_y[x] + bottom_y[x])//2], dtype=np.int32))


    midline_points_np = np.empty((len(midline_points), 2), dtype=np.float32)
    for i in range(len(midline_points)):
        midline_points_np[i] = midline_points[i]
    return midline_points_np

test_vals = np.array([[[1,1]], [[1,2]], [[2,2]]], dtype = np.int32)
compute_midline(nb.typed.List([test_vals]), 10, 10)

def solve_circle(pts, known_radius, guess, max_error_accepted):
    """
    Fit a circle to a set of points (max error = merror)
    """

    if len(pts) < 2:
        return None

    x, y = pts.T
    def error(c):
        xc, yc = c
        return np.sqrt((x-xc)**2 + (y-yc)**2)-known_radius

    center, ier = optimize.leastsq(error, guess, maxfev=50) # maxfev limits the number of iterations
    if ier in [1,2,3,4]:
        errs = error(center)
        max_err = np.amax(np.abs(errs))
        if max_err > max_error_accepted:
            return None
        return center
    return None

@nb.njit(nb.int32[:](nb.float32, nb.float32[:]))
def slide_window(window_width, sorted_xs):
    '''
    maximizes the number of points that can fit in a 1d window
    paramater: sorted list of xs and the size of the window
    returns: the first index and the last index of sorted_xs in the window
    '''
    max_is = 0
    max_start = 0
    max_end = 0
    window_end_i = 0
    for window_start_i in range(len(sorted_xs)):
        window_start = sorted_xs[window_start_i]
        window_end = window_start + window_width
        while window_end_i < len(sorted_xs) and sorted_xs[window_end_i] < window_end:
            window_end_i+=1
        if(window_end_i - window_start_i > max_is):
            max_is = window_end_i - window_start_i
            max_start = window_start_i
            max_end = window_end_i
    return(np.array([max_start, max_end], dtype=np.int32))

@nb.njit(nb.float32[:, :](nb.float32[:, :], nb.float32, nb.float32))
def yboundxbound(points, y_b, x_b):
    '''
    slides window in the y direction, then the x direction
    returns a list of points in the window
    '''
    y_sorted_is = np.argsort(points[:,1])
    y_sorted = points[y_sorted_is]
    y_inrange_start, y_inrange_end = slide_window(y_b, y_sorted[:,1])
    y_inrange = y_sorted[y_inrange_start : y_inrange_end]

    x_sorted_is = np.argsort(y_inrange[:,0])
    x_sorted = y_inrange[x_sorted_is]
    x_inrange_start, x_inrange_end = slide_window(x_b, x_sorted[:,0])
    x_inrange = x_sorted[x_inrange_start : x_inrange_end]
    return(x_inrange)

yboundxbound(np.array([[1,1],[3,3], [4,4]], dtype=np.float32),2,2)

class HubFinder2022(GenericFinder):
    '''Find hub ring for 2022 game'''
    # inches
    CAMERA_HEIGHT = 33
    CAMERA_ANGLE = math.radians(31)
    HUB_HEIGHT = 103
    HUB_RADIUS = 26.6875
    HUB_CIRCLE = np.array([(np.cos(theta) * 26.6875, np.sin(theta) * 26.6875) for theta in np.linspace(0,2*math.pi,20,endpoint=False)])
    MAX_FIT_ERROR = 1.5 #
    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('hubfinder', camera='shooter', finder_id=1.0, exposure=1)

        # Color threshold values, in HSV space
        self.low_limit_hsv = np.array((65, 35, 80), dtype=np.uint8)  # s was 100
        self.high_limit_hsv = np.array((100, 255, 255), dtype=np.uint8)

        # torture testing
        # self.low_limit_hsv = np.array((55, 5, 30), dtype=np.uint8)
        # self.high_limit_hsv = np.array((160, 255, 255), dtype=np.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 80

        self.hsv_frame = None
        self.threshold_frame = None
        self.ellipse_coeffs = None

        # DEBUG values
        self.midlines = None
        self.top_contours = None
        self.upper_points = None
        # output results
        self.target_contour = None

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.filter_box = None
        return

    def set_color_thresholds(self, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
        self.low_limit_hsv = np.array((hue_low, sat_low, val_low), dtype=np.uint8)
        self.high_limit_hsv = np.array((hue_high, sat_high, val_high), dtype=np.uint8)
        return

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

        shape = camera_frame.shape
        if self.hsv_frame is None or self.hsv_frame.shape != shape:
            self.preallocate_arrays(shape)

        self.hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV, dst=self.hsv_frame)
        self.threshold_frame = cv2.inRange(self.hsv_frame, self.low_limit_hsv, self.high_limit_hsv,
                                           dst=self.threshold_frame)
        # OpenCV 3 returns 3 parameters, OpenCV 4 returns 2!
        # Only need the contours variable
        start = timer()
        res = cv2.findContours(self.threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(res) == 2:
            contours = res[0]
        else:
            contours = res[1]


        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            if area < shape[0] * shape[1] * 2.5e-5:
                continue

            if area / (w*h) < 0.4:
                continue

            if w/h < 1.2:
                continue

            if w/h > 4:
                continue

            filtered_contours.append(contour)


        self.top_contours = filtered_contours
        if len(self.top_contours) > 1:
            self.midline_points = compute_midline(
                nb.typed.List(self.top_contours), self.threshold_frame.shape[1], self.threshold_frame.shape[0])
            midline_cv2 = np.array(self.midline_points, dtype=np.float32)[:,np.newaxis,:]
            midline_undistort_cv2 = cv2.undistortPoints(midline_cv2, self.cameraMatrix, self.distortionMatrix, P=self.cameraMatrix)
            midline_undistort = midline_undistort_cv2[:, 0, :]

            midline_prime_x = (midline_undistort[:, 0]  - self.cameraMatrix[0, 2]) / self.cameraMatrix[0, 0]
            midline_prime_y = -(midline_undistort[:, 1]  - self.cameraMatrix[1, 2]) / self.cameraMatrix[1, 1]
            hubplane_points_all = transform_hubplane_many(np.vstack([midline_prime_x, midline_prime_y]).T, HubFinder2022.CAMERA_ANGLE, HubFinder2022.HUB_HEIGHT - HubFinder2022.CAMERA_HEIGHT)

            hubplane_points = yboundxbound(hubplane_points_all, HubFinder2022.HUB_RADIUS*0.8,  HubFinder2022.HUB_RADIUS*0.8 * 2)

            middle_guess = np.mean(hubplane_points, axis = 0)
            middle_guess[1] += HubFinder2022.HUB_RADIUS

            middle = solve_circle(hubplane_points,HubFinder2022.HUB_RADIUS, middle_guess, HubFinder2022.MAX_FIT_ERROR)

            if middle is None:
                self.ellipse, self.top_point, self.hub_point = None, None, None
                #print("failed circle fit")

            else:
                top_point_hubplane = np.array(middle)
                top_point_hubplane[1] -= HubFinder2022.HUB_RADIUS

                '''
                plt.plot(*middle_guess, marker="o")
                plt.plot(*middle, marker="o")
                plt.plot(0,0, marker="o")
                plt.axline((0, 0), middle, c="green")
                plt.axline((0, 0), top_point_hubplane, c="red")
                plt.axline((0, 0), (0,1))

                plt.plot(*top_point_hubplane, marker="o")
                plt.scatter(*hubplane_points_all.T)
                plt.axis('equal')
                plt.gca().add_patch(plt.Circle(middle, HubFinder2022.HUB_RADIUS, color='b', fill=False))

                plt.show()
                '''

                to_transform = np.array([top_point_hubplane, *(HubFinder2022.HUB_CIRCLE + middle)], dtype=np.float32)
                transformed_camera = transform_camera_many(to_transform, HubFinder2022.CAMERA_ANGLE, HubFinder2022.HUB_HEIGHT - HubFinder2022.CAMERA_HEIGHT)
                transformed_camera = redistort_many(transformed_camera, self.distortionMatrix.reshape((5,)).astype(np.float32))
                transformed_image_x = transformed_camera[:,0] * self.cameraMatrix[0, 0] + self.cameraMatrix[0, 2]
                transformed_image_y = - transformed_camera[:,1] * self.cameraMatrix[1, 1] + self.cameraMatrix[1, 2]
                transformed_image = np.array([transformed_image_x, transformed_image_y]).T

                self.top_point = transformed_image[0]
                self.ellipse = transformed_image[1:]
                self.hub_point = middle
                #print("success")
        else:
            #print("failed no. contours")

            self.midline_points, self.ellipse, self.top_point, self.hub_point = None, None, None, None


        if self.hub_point is not None:
            angle = np.arctan2(self.hub_point[0], self.hub_point[1])
            distance = np.hypot(self.hub_point[1], self.hub_point[0]) - HubFinder2022.HUB_RADIUS

            result = np.array([1.0, self.finder_id, distance, angle, 0.0, 0.0, 0.0])
        else:
            result = np.array([0.0, self.finder_id, 0.0, 0.0, 0.0, 0.0, 0.0])

        end = timer()

        self.processing_time = (end - start)*1000
        #print(self.processing_time)
        return result

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        if self.top_contours is not None:
            cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 2)

        if self.ellipse is not None:
            cv2.drawContours(output_frame, self.ellipse[:,np.newaxis,:].astype(np.int32), -1, (0, 0, 255), 2)

        if self.midline_points is not None:
            for pt in self.midline_points:
                cv2.drawMarker(output_frame, tuple(pt.astype(int)),
                               (0, 255, 255), cv2.MARKER_CROSS, 5, 2)
        if(self.ellipse_coeffs is not None and not np.any(np.isnan(self.ellipse_coeffs))):
            cv2.ellipse(output_frame, (int(self.ellipse_coeffs[0]), int(self.ellipse_coeffs[1])), (int(self.ellipse_coeffs[2]), int(self.ellipse_coeffs[3])),
                        0, 0, 360, (255, 0, 0), 3)

        if(self.top_point is not None):
            pt = tuple(self.top_point.astype(int))
            if(0 <= pt[0] < output_frame.shape[1] and 0 <= pt[1] < output_frame.shape[0]):
                cv2.drawMarker(output_frame, pt, (255, 255, 255), cv2.MARKER_CROSS, 15, 2)

        if self.filter_box is not None:
            cv2.rectangle(output_frame, (int(self.filter_box[0]), int(self.filter_box[1])), (int(self.filter_box[2]), int(self.filter_box[3])), (255,0,0), 2)

        cv2.putText(output_frame, f"{int(self.processing_time*10)/10}ms", (20,
                    output_frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=2)

        return output_frame

# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(HubFinder2022)

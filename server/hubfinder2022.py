#!/usr/bin/env python3

# Vision finder to find the retro-reflective target on the hub

import cv2
import numpy as np
import numba as nb
from timeit import default_timer as timer
import math
from genericfinder import GenericFinder, main


@nb.njit(nb.float32[:](nb.float32[:, :]))
def fit_ACDE(points):

    # solving Ax^2 + Cy^2 + Dx + Ey = 1
    # given: points (x1, y1), (x2, y2), .. (xn, yn)
    # NM = P

    N = np.empty((len(points), 4))
    for (i, (x, y)) in enumerate(points):
        N[i] = [x**2, y**2, x, y]
    M = np.linalg.pinv(N) @ np.ones((len(points),))
    return M.astype(np.float32)


@nb.njit(nb.float32[:](nb.float32[:]))
def canonical(M):
    # compute canoical form of ellipse, given ACDE
    A, C, D, E = M
    h = -D/(2*A)
    k = -E/(2*C)
    m = D*D/(4*A) + E*E/(4*C) + 1
    a = np.sqrt(m/A)
    b = np.sqrt(m/C)

    return(np.array([h, k, a, b], dtype=np.float32))

'''
@nb.njit(nb.float32(nb.float32, nb.float32, nb.float32, nb.float32))
def ellipse_distance(a, b, px, py):
    # credit: https://stackoverflow.com/a/46007540/5771000

    px, py = abs(px), abs(py)

    tx = 0.707
    ty = 0.707

    for _ in range(0, 3):
        x = a * tx
        y = b * ty

        ex = (a*a - b*b) * tx**3 / a
        ey = (b*b - a*a) * ty**3 / b

        rx = x - ex
        ry = y - ey

        qx = px - ex
        qy = py - ey

        r = math.hypot(ry, rx)
        q = math.hypot(qy, qx)

        tx = min(1, max(0, (qx * r / q + ex) / a))
        ty = min(1, max(0, (qy * r / q + ey) / b))
        t = math.hypot(ty, tx)
        tx /= t
        ty /= t

    return(math.hypot(a * tx, b * ty))

'''
@nb.njit(nb.types.Tuple([nb.types.ListType(nb.types.Array(nb.int32, 1, "C")), nb.optional(nb.types.Array(nb.float32, 1, "A")), nb.optional(nb.types.Array(nb.float32, 1, "A"))])(nb.types.ListType(nb.types.Array(nb.int32, 3, "C")), nb.int32, nb.int32))
def process_contours(contours, width, height):
    # first compute "midline": center lines of each contour
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

        for x in range(min_x, max_x+1):
            if(top_y[x] != -1 and bottom_y[x] != -1):
                midline_points.append(np.array([x, (top_y[x] + bottom_y[x])//2]))
    if len(midline_points) == 0:
        return (midline_points, None, None)

    midline_points_np = np.empty((len(midline_points), 2), dtype=np.float32)
    for i in range(len(midline_points)):
        midline_points_np[i] = midline_points[i]

    # fit to ellipse
    M = fit_ACDE(midline_points_np)

    std = canonical(M)

    if(np.any(np.isnan(std))):
        return midline_points, None, None

    toppoint = np.array([std[0], std[1] - std[3]])
    return (midline_points, std, toppoint)

test_vals = np.array([[[1,1]], [[1,2]], [[2,2]]], dtype = np.int32)
process_contours(nb.typed.List([test_vals]), 10, 10)


class HubFinder2022(GenericFinder):
    '''Find hub ring for 2022 game'''
    # inches
    CAMERA_HEIGHT = 32.5
    CAMERA_ANGLE = math.radians(35)
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
        self.ellipse_coeffs = None

        # DEBUG values
        self.midlines = None
        self.top_contours = None
        self.upper_points = None
        # output results
        self.target_contour = None

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix

        self.outer_corners = []

        self.filter_box = None
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

        max_area = 0

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

            if(max_area < area):
                max_area = area
                cx, cy = x+w/2 , y+h/2
                dx, dy = w*10, h*5
                self.filter_box = [cx-dx, cy-dy, cx+dx, cy+dy]
            filtered_contours.append(contour)

        position_filtered_contours = []
        if len(filtered_contours):
            for contour in filtered_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x+w/2, y+h/2
                if (self.filter_box[0] <= cx <= self.filter_box[2]) and (self.filter_box[1] <= cy <= self.filter_box[3]):
                    position_filtered_contours.append(contour)
        self.top_contours = position_filtered_contours
        if len(position_filtered_contours):

            self.midline_points, self.ellipse_coeffs, self.top_point = process_contours(
                nb.typed.List(position_filtered_contours), self.threshold_frame.shape[1], self.threshold_frame.shape[0])
        else:
            self.midline_points, self.ellipse_coeffs, self.top_point = None, None, None

        if self.top_point is not None:
            angle, distance = self.calculate_angle_distance(self.top_point)

            result = np.array([1.0, self.finder_id, distance, angle, 0.0, 0.0, 0.0])
        else:
            result = np.array([0.0, self.finder_id, 0.0, 0.0, 0.0, 0.0, 0.0])
        end = timer()

        self.processing_time = (end - start)*1000
        return result

    def calculate_angle_distance(self, center):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the cameraMatrix from the calibration to convert to normalized coordinates'''

        # use the distortion and camera arrays to correct the location of the center point
        # got this from
        #  https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi

        ptlist = np.array([[center]])
        out_pt = cv2.undistortPoints(ptlist, self.cameraMatrix,
                                     self.distortionMatrix, P=self.cameraMatrix)
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
        d = (HubFinder2022.HUB_HEIGHT - HubFinder2022.CAMERA_HEIGHT) / \
            math.tan(HubFinder2022.CAMERA_ANGLE + ay)    # distance to the target

        return ax, d    # return horizontal angle and distance

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()

        if self.top_contours is not None:
            cv2.drawContours(output_frame, self.top_contours, -1, (0, 0, 255), 2)

        if self.midline_points is not None:
            for pt in self.midline_points:
                cv2.drawMarker(output_frame, tuple(pt.astype(int)),
                               (0, 255, 255), cv2.MARKER_CROSS, 15, 2)

        if(self.ellipse_coeffs is not None and not np.any(np.isnan(self.ellipse_coeffs))):
            cv2.ellipse(output_frame, (int(self.ellipse_coeffs[0]), int(self.ellipse_coeffs[1])), (int(self.ellipse_coeffs[2]), int(self.ellipse_coeffs[3])),
                        0, 0, 360, (255, 0, 0), 3)

        if(self.top_point is not None):
            pt = tuple(self.top_point.astype(int))
            if(0 <= pt[0] < output_frame.shape[1] and 0 <= pt[1] < output_frame.shape[0]):
                cv2.drawMarker(output_frame, pt, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

        if self.filter_box is not None:
            cv2.rectangle(output_frame, (int(self.filter_box[0]), int(self.filter_box[1])), (int(self.filter_box[2]), int(self.filter_box[3])), (255,0,0), 2)

        cv2.putText(output_frame, f"{int(self.processing_time*10)/10}ms", (20,
                    output_frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=2)

        return output_frame

# Main routine
# This is for development/testing without running the whole server
if __name__ == '__main__':
    main(HubFinder2022)

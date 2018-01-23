#!/usr/bin/env python3

import cv2
import numpy
import json
from skimage.feature import corner
from skimage.feature.corner import corner_fast


class CubeFinder2018(object):
    '''Find power cube for PowerUp 2018'''

    CUBE_HEIGHT = 11    #inches
    CUBE_WIDTH = 13     #inches
    CUBE_LENGTH = 13    #inches

    def __init__(self, calib_file):
        # Color threshold values, in HSV space -- TODO: in 2018 server (yet to be created) make the low and high hsv limits
        # individual properties
        self.low_limit_hsv = numpy.array((25, 95, 110), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((75, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 100

        self.erode_kernel = numpy.ones((3, 3), numpy.uint8)
        self.erode_iterations = 2

        with open(calib_file) as f:
            json_data = json.load(f)
            self.cameraMatrix = numpy.array(json_data["camera_matrix"])
            self.distortionMatrix = numpy.array(json_data["distortion"])

        self.target_coords = numpy.array([[-CubeFinder2018.CUBE_LENGTH/2.0, -CubeFinder2018.CUBE_HEIGHT/2.0, 0.0],
                                          [-CubeFinder2018.CUBE_LENGTH/2.0,  CubeFinder2018.CUBE_HEIGHT/2.0, 0.0],
                                          [ CubeFinder2018.CUBE_LENGTH/2.0,  CubeFinder2018.CUBE_HEIGHT/2.0, 0.0],
                                          [ CubeFinder2018.CUBE_LENGTH/2.0, -CubeFinder2018.CUBE_HEIGHT/2.0, 0.0]])
        return

    @staticmethod
    def contour_center_width(contour):
        '''Find boundingRect of contour, but return center, width, and height'''

        x, y, w, h = cv2.boundingRect(contour)
        return (x + int(w / 2), y + int(h / 2)), (w, h)

    @staticmethod
    def quad_fit(contour, approx_dp_error):
        '''Simple polygon fit to contour with error related to perimeter'''

        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, approx_dp_error * peri, True)

    @staticmethod
    def sort_corners(cnrlist, check):
        '''Sort a list of corners -- if check == true then returns x sorted 1st, y sorted 2nd. Otherwise the opposite'''
        
        #recreate the list of corners to get rid of excess dimensions
        corners = []
        for i in range(int((cnrlist.size + 1) / 2)):
            corners.append(cnrlist[i][0])
            
        #sort the corners by x values (1st column) first and then by y values (2nd column)
        if check:
            return sorted(corners, key=lambda x: (x[0], x[1]))
        #y's first then x's
        else:
            return sorted(corners, key=lambda x: (x[1], x[0]))
    
    @staticmethod
    def split_xs_ys(corners):
        '''Split a list of corners into sorted lists of x and y values'''
        xs = []
        ys = []
        
        for i in range(int(len(corners) / 2)):
            xs.append(corners[i][0])
            ys.append(corners[i][1])
        #sort the lists lowest to highest
        xs.sort()
        ys.sort()
        return xs, ys
    
    @staticmethod
    def choose_corners_sides(cnrlist):
        '''Sort a list of corners and returns the 2 left most and 2 right most corners in order'''
        corners = CubeFinder2018.sort_corners(cnrlist, True)
        return [corners[0], corners[1], corners[len(corners) - 2], corners[len(corners) - 1]]
    
    @staticmethod
    def choose_corners_frontface(cnrlist):
        '''Sort a list of corners and return the bottom and side corners (one side -- 3 in total - .: or :.)
        of front face'''
        corners = CubeFinder2018.sort_corners(cnrlist, True)
        xs, _ = CubeFinder2018.split_xs_ys(corners)
        x_min1 = xs[0]
        x_min2 = xs[1]
        
        # if above point is within ___ pixels to the right and left
        error_margin = 40   #pixels
        possible_points = []   #stores possible points for 3rd point of front face
        #TODO: should only be 2 points each time -- if not, adjust error_margin -- maybe adjust in proportion with cube area?
        for corner in corners:
            y = corner[1]
            if y >= (x_min1 - error_margin) and y <= (x_min1 + error_margin):
                possible_points.append(corner)
            if y >= (x_min2 - error_margin) and y <= (x_min2 + error_margin):
                possible_points.append(corner)
        
        _, ys = CubeFinder2018.split_xs_ys(possible_points) #used to get y min
        #find the actual top point with the found y value
        for corner in corners:
            if corner[1] == ys[0]:
                top_corner = [corner[0], corner[1]]
        #find the other 2 points with the x_min1 and x_min2:
        for corner in corners:
            x = corner[0]
            y = corner[1]
            if (x == x_min1 or x == x_min2) and not (y >= (x_min1 - error_margin) and (y <= x_min1 + error_margin)):
                lonely_corner = [corner[0], corner[1]]
            if (x == x_min1 or x == x_min2) and (y >= (x_min1 - error_margin) and (y <= x_min1 + error_margin)):
                happy_corner = [corner[0], corner[1]]
                
        #return order: [bottom lonely corner, bottom happy corner, top corner]
        #TODO: fix this to get the right values - not work
        return [lonely_corner, happy_corner, top_corner]
    
    @staticmethod
    def get_cube_facecenter(cnrlist):
        '''Compute the center of a cube face from a list of the three face corners'''
        #get the three corners of the front face
        front_corners = CubeFinder2018.choose_corners_frontface(cnrlist)
        #average of x, y values of opposite corners of front face of cube
        x = (front_corners[0][0] + front_corners[2][0]) / 2
        y = (front_corners[0][1] + front_corners[2][1]) / 2
        return [x, y]       #return center point of cube front face
        
    
    @staticmethod
    def get_cube_center(img, cnrlist):
        '''return the center of the cube'''
        #sort just to format correctly -- get rid of extra dimensions
        corners = CubeFinder2018.sort_corners(cnrlist, True)
        #xs and ys only needed for drawing the point on the image
        xs, ys = CubeFinder2018.split_xs_ys(corners)
        
        sum_x = 0
        sum_y = 0
        for corner in corners:
            sum_x += corner[0]
            sum_y += corner[1]
        center = numpy.array([ int(sum_x / (len(corners) / 2)), int(sum_y / (len(corners) / 2)) ])
        cv2.circle(img, (center[0], center[1]), 5, (255, 0, 0), thickness=50, lineType=8, shift=0)
        #cv2.line(img, (xs.min, center[1]), (xs.max, center[1]), (255,0,0), 5)
        #cv2.line(img, (center[0], ys.min), (center[0], ys.max), (255,0,0), 5)
        return sum_x / (len(corners) / 2), sum_y / (len(corners) / 2)

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)

        if self.erode_iterations > 0:
            erode_frame = cv2.erode(threshold_frame, self.erode_kernel, iterations=self.erode_iterations)
        else:
            erode_frame = threshold_frame

        # OpenCV 3 returns 3 parameters!
        # Only need the contours variable
        _, contours, _ = cv2.findContours(erode_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for c in contours:
            center, widths = CubeFinder2018.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:
                # TODO: use a simple class? Maybe use "attrs" package?
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        if len(contour_list) != 0:
            biggest_contour = contour_list[0]['contour']
            top2_contour = [contour_list[0]['contour'], ]
            if len(contour_list) > 1:
                top2_contour.append(contour_list[1]['contour'])

            cv2.drawContours(camera_frame, top2_contour, -1, (0, 0, 255), 2)

            # # lets see what the bounding rectangle looks like
            # center = contour_list[0]['center']
            # widths = contour_list[0]['widths']
            # cv2.rectangle(camera_frame, (center[0] - widths[0] // 2, center[1] - widths[1] // 2),
            #               (center[0] + widths[0] // 2, center[1] + widths[1] // 2), (255, 255, 0))

            # # poly_fit = CubeFinder.quad_fit(contour_list[0]['contour'], 0.01)
            # # cv2.drawContours(camera_frame, [poly_fit], -1, (100, 255, 255), 1)

            hull = cv2.convexHull(biggest_contour)
            # hull_fit contains the corners for the contour
            hull_fit = CubeFinder2018.quad_fit(hull, 0.01)
            # cv2.drawContours(camera_frame, [hull], -1, (0, 255, 0), 1)
            # Draw the contour on the image
            cv2.drawContours(camera_frame, [hull_fit], -1, (255, 0, 0), 2)

            corners = numpy.array(hull_fit)
            # divide by 2 since there are 2 elements per coordinate and .size takes into account both of them
            if (corners.size / 2) >= 4 and (corners.size / 2) <= 6:
                #to find center of whole cube:
                #image_corners = CubeFinder2018.choose_corners_lr(corners)
                #center = CubeFinder2018.get_cube_center(camera_frame, corners)
                
                #to find center of front cube face:
                center = CubeFinder2018.get_cube_facecenter(corners)
                pass
            
            #DON'T WANT TO USE SOLVEPNP() --> THIS IS 3D TARGET, NOT 2D
            #retval, rvec, tvec = cv2.solvePnP(self.target_coords, image_corners,
            #                                  self.cameraMatrix, self.distortionMatrix) #TODO: fix this
            #if retval:
            #    # Return values are 3x1 matrices. Convert to Python lists
            #    return rvec.flatten().tolist(), tvec.flatten().tolist()

        # print('contour peri =', cv2.arcLength(contour_list[0]['contour'], True),
        #       ' hull peri =', cv2.arcLength(hull, True))
        # min_rect = cv2.minAreaRect(biggest_contour)
        # box = numpy.int0(cv2.boxPoints(min_rect))
        # cv2.drawContours(camera_frame, [box], -1, (255, 0, 255), 1)
        # minRectArea = cv2.contourArea(box)
        # print('contour area =', cv2.contourArea(biggest_contour),
        #       ' contour bounding area =', contour_list[0]['area'],
        #       ' contour min area rect =', minRectArea,
        #       ' hull area =', cv2.contourArea(hull),
        #       ' hull fit area =', cv2.contourArea(hull_fit))

        cv2.imshow("Window", camera_frame)

        # Probably can distinguish a cross by the ratio of perimeters and/or areas
        # That is, it is not universally true, but probably true from what we would see on the field

        return None, None


def process_files(cube_processor, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)
        cube_processor.process_image(bgr_frame)
        # print("rvec: " + rvec)
        # print("tvec: " + tvec)

        outfile = os.path.join(output_dir, os.path.basename(image_file))
        # print('{} -> {}'.format(image_file, outfile))
        cv2.imwrite(outfile, bgr_frame)

        # cv2.imshow("Window", bgr_frame)
        q = cv2.waitKey(-1) & 0xFF
        if q == ord('q'):
            break
    return


def time_processing(cube_processor, input_files):
    '''Time the processing of the test files'''

    from codetimer import CodeTimer
    from time import time

    startt = time()

    cnt = 0

    # Loop 100x over the files. This is needed to make it long enough
    #  to get reasonable statistics. If we have 100s of files, we could reduce this.
    # Need the total time to be many seconds so that the timing resolution is good.
    for _ in range(100):
        for image_file in input_files:
            with CodeTimer("Read Image"):
                bgr_frame = cv2.imread(image_file)

            with CodeTimer("Main Processing"):
                cube_processor.process_image(bgr_frame)

            cnt += 1

    deltat = time() - startt

    print("{0} frames in {1:.3f} seconds = {2:.2f} msec/call, {3:.2f} FPS".format(
        cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    CodeTimer.outputTimers()
    return


def main():
    '''Main routine'''
    import argparse

    parser = argparse.ArgumentParser(description='2018 cube finder')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    cube_processor = CubeFinder2018(args.calib_file)

    if args.output_dir is not None:
        process_files(cube_processor, args.input_files, args.output_dir)
    elif args.time:
        time_processing(cube_processor, args.input_files)

    return


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main()

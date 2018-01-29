#!/usr/bin/env python3

import cv2
import numpy
import json
from math import tan
from math import atan
from math import atan2


class CubeFinder2018(object):
    '''Find power cube for PowerUp 2018'''

    CUBE_HEIGHT = 11    #inches
    CUBE_WIDTH = 13     #inches
    CUBE_LENGTH = 13    #inches

    def __init__(self, calib_file):
        # Color threshold values, in HSV space -- TODO: in 2018 server (yet to be created) make the low and high hsv limits
        # individual properties
        self.low_limit_hsv = numpy.array((25, 95, 95), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((75, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 100

        self.erode_kernel = numpy.ones((3, 3), numpy.uint8)
        self.erode_iterations = 2

		# save the found center for drawing on the image
        self.center = None

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
        for i in range(int(cnrlist.size / 2)):
            corners.append(cnrlist[i][0].tolist())
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

        for i in range(len(corners)):
            xs.append(corners[i][0])
            ys.append(corners[i][1])
        #sort the lists highest to lowest
        xs.sort(reverse=True)
        ys.sort(reverse=True)
        return xs, ys

    @staticmethod
    def choose_corners_frontface(img, cnrlist):
        '''Sort a list of corners and return the bottom and side corners (one side -- 3 in total - .: or :.)
        of front face'''
        corners = CubeFinder2018.sort_corners(cnrlist, False)    #get rid of extra dimensions
        
        happy_corner = corners[len(corners) - 1]
        lonely_corner = corners[len(corners) - 2]
        
        xs, ys = CubeFinder2018.split_xs_ys(corners)

        #lonely corner is green and happy corner is red
        cv2.circle(img, (lonely_corner[0], lonely_corner[1]), 5, (0, 255, 0), thickness=10, lineType=8, shift=0)
        cv2.circle(img, (happy_corner[0], happy_corner[1]), 5, (0, 0, 255), thickness=10, lineType=8, shift=0)
        
        corners = CubeFinder2018.sort_corners(cnrlist, True)
        
        if happy_corner[0] > lonely_corner[0]:
            top_corner = corners[len(corners) - 1]
        else:
            top_corner = corners[0]
        #top corner is in blue
        cv2.circle(img, (top_corner[0], top_corner[1]), 5, (255, 0, 0), thickness=10, lineType=8, shift=0)
        return ([lonely_corner, happy_corner, top_corner])

    @staticmethod
    def get_cube_facecenter(img, cnrlist):
        '''Compute the center of a cube face from a list of the three face corners'''
        #get the three corners of the front face
        front_corners = CubeFinder2018.choose_corners_frontface(img, cnrlist)
        #average of x, y values of opposite corners of front face of cube
        x = int((front_corners[0][0] + front_corners[2][0]) / 2)
        y = int((front_corners[0][1] + front_corners[2][1]) / 2)

        #middle point in white
        #cv2.circle(img, (x, y), 5, (255, 255, 255), thickness=10, lineType=8, shift=0)
        return [x, y]       #return center point of cube front face'''


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
        return sum_x / (len(corners) / 2), sum_y / (len(corners) / 2)

    @staticmethod
    def get_cube_bottomcenter(img, cnrlist):
        '''return the center of the bottom-front side of the cube'''
        corners = CubeFinder2018.sort_corners(cnrlist, False)
        
        bottom_corner_a = corners[len(corners) - 1]
        bottom_corner_b = corners[len(corners) - 2]



        center = [ int((bottom_corner_a[0] + bottom_corner_b[0]) / 2), int((bottom_corner_a[1] + bottom_corner_b[1]) / 2) ]
        cv2.circle(img, (center[0], center[1]), 5, (0, 255, 0), thickness=10, lineType=8, shift=0)
        return center

    @staticmethod
    def get_cube_values(center):
        '''Calculate the angle and distance from the camera to the center point of the robot'''

        #(px,py) = pixel coordinates, 0,0 is the upper-left, positive down and to the right
        #(nx,ny) = normalized pixel coordinates, 0,0 is the center, positive right and up
        px = center[0]
        py = center[1]
        nx = (1/160) * (px - 159.5)  #TODO: Change values for size of our camera
        ny = (1/120) * (119.5 - py)

        horizontal_fov = 54     #horizontal angle of the field of view
        vertical_fov = 41       #vertical angle of the field of view

        #create imaginary view plane on 3d coords to get height and width
        #place the view place on 3d coordinate plane 1.0 unit away from (0, 0) for simplicity
        vpw = 2.0*tan(horizontal_fov/2)     #view plane height
        vph = 2.0*tan(vertical_fov/2)       #view plane width

        #convert normal pixel coords to pixel coords
        x = vpw/2 * nx
        y = vph/2 * ny

        #now have all pieces to convert to angle:
        ax = atan2(1,x)     #horizontal angle
        ay = atan2(1,y)     #vertical angle

        #now use the x and y angles to calculate the distance to the target:
        a1 = 15   #camera mount angle (degrees)
        #TODO: change h1? ask cad people (the people over there)
        h1 = 13   #height of camera off the ground (inches)
        h2 = 0    #height of target off the ground (inches)
        d = (h2-h1) / tan(a1+ay)    #distance to the target

        return ax, d    #return horizontal angle and distance

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        hsv_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
        threshold_frame = cv2.inRange(hsv_frame, self.low_limit_hsv, self.high_limit_hsv)
        rvec = None
        tvec = None #TODO: Should end up being vector of 3 numbers -- ??
        self.center = None
        
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

        # NOTE: testing a list returns true if there is something in the list
        if contour_list:
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
            if (corners.size / 2) >= 4 and (corners.size / 2) <= 12:    #12 = max # of corners if all corners are flat
                #to find center of whole cube:
                #image_corners = CubeFinder2018.choose_corners_lr(corners)
                #center = CubeFinder2018.get_cube_center(camera_frame, corners)

                #to find center of front cube face:
                #self.center = CubeFinder2018.get_cube_facecenter(camera_frame, corners)

                self.center = CubeFinder2018.get_cube_bottomcenter(camera_frame, corners)
                rvec, tvec = CubeFinder2018.get_cube_values(self.center)

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

        if rvec is None or tvec is None:
            return None, None
        return rvec, tvec

    def prepare_output_image(self, output_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        if self.center is not None:
            cv2.circle(output_frame, (self.center[0],self.center[1]), 5, (255, 0, 0), thickness=10, lineType=8, shift=0)

        # TODO: draw the found elements on the image for the Driver Station
        # if self.target_contour is not None:
        #    cv2.drawContours(output_frame, [self.target_contour], -1, (255, 255, 255), 1)

        return


def process_files(cube_processor, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path

    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)
        rvec, tvec = cube_processor.process_image(bgr_frame)
        print("rvec: " + str(rvec))
        print("tvec: " + str(tvec))

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

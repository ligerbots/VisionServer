#!/usr/bin/env python3

"""Generic finder used when not searching for any targets.
name and finder_id are instance variables so that we can create multiple
GenericFinders for different purposes.

finder_id may not need to be changed, depending on circumstances."""

import sys
import math
import cv2
import numpy
import hough_fit

two_pi = 2 * math.pi
pi_by_2 = math.pi / 2


class GenericFinder:
    def __init__(self, name, camera, finder_id=1.0, exposure=0, line_coords=None):
        self.name = name
        self.finder_id = float(finder_id)   # id needs to be float! "id" is a reserved word.
        self.camera = camera                # string with camera name
        self.stream_camera = None           # None means same camera
        self.exposure = exposure
        self.line_coords = line_coords      # coordinates to draw a line on the image
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''

        # for 2020, standard result includes position, angle of a 2nd ball
        return (1.0, self.finder_id, 0.0, 0.0, 0.0, -1.0, -1.0)

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station.
        Add a guide line if that is called for.'''

        output_frame = input_frame.copy()
        if self.line_coords is not None:
            cv2.line(output_frame, self.line_coords[0], self.line_coords[1], (255, 255, 255), 2)

        return output_frame

    # ----------------------------------------------------------------------
    # the routines below are not needed here, but are used by lots of Finders,
    #  so keep them in one place

    @staticmethod
    def contour_center_width(contour):
        '''Find boundingRect of contour, but return center and width/height'''

        x, y, w, h = cv2.boundingRect(contour)
        return (x + int(w / 2), y + int(h / 2)), (w, h)

    @staticmethod
    def quad_fit(contour, image_frame=None):
        '''Best fit of a quadrilateral to the contour.
        Pass in image_frame to get some debugging info.'''

        approx = hough_fit.approxPolyDP_adaptive(contour, nsides=4)
        return hough_fit.hough_fit(contour, nsides=4, approx_fit=approx, image_frame=image_frame)

    @staticmethod
    def sort_corners(contour, center=None):
        '''Sort the contour in our standard order, starting upper-left and going counter-clockwise'''

        # Note: the inputs are all numpy arrays, so it is fast to operate on the whole array at once

        if center is None:
            center = contour.mean(axis=0)

        d = contour - center
        # remember that y-axis increases down, so flip the sign
        angle = (numpy.arctan2(-d[:, 1], d[:, 0]) - pi_by_2) % two_pi
        return contour[numpy.argsort(angle)]

    @staticmethod
    def center_cross(points):
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """

        rho1, theta1 = GenericFinder._hesse_form(points[0], points[2])
        rho2, theta2 = GenericFinder._hesse_form(points[1], points[3])
        if abs(theta1 - theta2) < 1e-6:
            # parallel
            return None

        cos1 = math.cos(theta1)
        sin1 = math.sin(theta1)
        cos2 = math.cos(theta2)
        sin2 = math.sin(theta2)

        denom = cos1*sin2 - sin1*cos2
        x = (sin2*rho1 - sin1*rho2) / denom
        y = (cos1*rho2 - cos2*rho1) / denom
        res = numpy.array((x, y))
        return res

    @staticmethod
    def _hesse_form(pt1, pt2):
        '''Compute the Hesse form for the line through the points'''

        delta = pt2 - pt1
        mag2 = delta.dot(delta)
        vec = pt2 - pt2.dot(delta) * delta / mag2

        rho = math.sqrt(vec.dot(vec))
        if abs(rho) < 1e-6:
            # through 0. Need to compute theta differently
            theta = math.atan2(delta[1], delta[0]) + pi_by_2
            if theta > two_pi:
                theta -= two_pi
        else:
            theta = math.atan2(vec[1], vec[0])

        return rho, theta

    @staticmethod
    def major_minor_axes(moments):
        '''Compute the major/minor axes and orientation of an object from the moments'''

        # See https://en.wikipedia.org/wiki/Image_moment
        # Be careful, different sites define the normalized central moments differently
        # See also http://raphael.candelier.fr/?blog=Image%20Moments

        m00 = moments['m00']
        mu20 = moments['mu20'] / m00
        mu02 = moments['mu02'] / m00
        mu11 = moments['mu11'] / m00

        descr = math.sqrt(4.0 * mu11*mu11 + (mu20 - mu02)**2)

        major = math.sqrt(2.0 * (mu20 + mu02 + descr))
        minor = math.sqrt(2.0 * (mu20 + mu02 - descr))

        # note this does not use atan2.
        angle = 0.5 * math.atan(2*mu11 / (mu20-mu02))
        if mu20 < mu02:
            angle += pi_by_2

        return major, minor, angle

    @staticmethod
    def distance_angle_from_point(point, height_diff, camera_matrix, distortion_matrix, tilt_angle=0.0):
        '''Calculate the angle and distance from the camera to the center point of the robot
        This routine uses the cameraMatrix from the calibration to convert to normalized coordinates'''

        # use the distortion and camera arrays to correct the location of the center point
        # got this from
        #  https://stackoverflow.com/questions/8499984/how-to-undistort-points-in-camera-shot-coordinates-and-obtain-corresponding-undi

        ptlist = numpy.array([[point]])
        out_pt = cv2.undistortPoints(ptlist, camera_matrix, distortion_matrix, P=camera_matrix)
        undist_center = out_pt[0, 0]

        x_prime = (undist_center[0] - camera_matrix[0, 2]) / camera_matrix[0, 0]
        y_prime = -(undist_center[1] - camera_matrix[1, 2]) / camera_matrix[1, 1]

        # now have all pieces to convert to angle:
        ax = math.atan2(x_prime, 1.0)     # horizontal angle

        # naive expression
        # ay = math.atan2(y_prime, 1.0)     # vertical angle

        # corrected expression.
        # As horizontal angle gets larger, real vertical angle gets a little smaller
        ay = math.atan2(y_prime * math.cos(ax), 1.0)     # vertical angle
        # print("ax, ay", math.degrees(ax), math.degrees(ay))

        # now use the x and y angles to calculate the distance to the target:
        d = height_diff / math.tan(tilt_angle + ay)    # distance to the target

        return d, ax    # return distance and horizontal angle


# --------------------------------------------------------------------------------
# Main routines, used for running the finder by itself for debugging and timing

def process_files(line_finder, input_files, output_dir):
    '''Process the files and output the marked up image'''
    import os.path
    import re

    print('File,Success,Mode,Distance1,RobotAngle1,TargetAngle1,Distance2,RobotAngle2')
    for image_file in input_files:
        # print()
        # print(image_file)
        bgr_frame = cv2.imread(image_file)

        result = line_finder.process_image(bgr_frame)
        print(image_file, result[0], result[1], round(result[2], 1),
              round(math.degrees(result[3]), 1), round(math.degrees(result[4]), 1),
              round(result[5], 1), round(math.degrees(result[6]), 1), sep=',')

        bgr_frame = line_finder.prepare_output_image(bgr_frame)

        outfile = os.path.join(output_dir, os.path.basename(image_file))
        # output as PNG, because that does not do compression
        outfile = re.sub(r'\.jpg$', '.png', outfile, re.IGNORECASE)
        # print('{} -> {}'.format(image_file, outfile))
        cv2.imwrite(outfile, bgr_frame)

        # cv2.imshow("Window", bgr_frame)
        # q = cv2.waitKey(-1) & 0xFF
        # if q == ord('q'):
        #     break
    return


def time_processing(processor, input_files):
    '''Time the processing of the test files'''

    from codetimer import CodeTimer
    from time import time

    startt = time()

    cnt = 0

    # Loop 100x over the files. This is needed to make it long enough
    #  to get reasonable statistics. If we have 100s of files, we could reduce this.
    # Need the total time to be many seconds so that the timing resolution is good.
    for _ in range(10):
        for image_file in input_files:
            with CodeTimer("Read Image"):
                bgr_frame = cv2.imread(image_file)

            with CodeTimer("Main Processing"):
                processor.process_image(bgr_frame)
                processor.prepare_output_image(bgr_frame)

            cnt += 1

    deltat = time() - startt

    print("{0} frames in {1:.3f} seconds = {2:.2f} ms/call, {3:.2f} FPS".format(
        cnt, deltat, 1000.0 * deltat / cnt, cnt / deltat))
    CodeTimer.output_timers()
    return


def main(finder_type):
    '''Main routine for testing a Finder'''
    import argparse
    import camerautil

    parser = argparse.ArgumentParser(description='finder test routine')
    parser.add_argument('--output_dir', help='Output directory for processed images')
    parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    parser.add_argument('--calib_file', help='Calibration file')
    parser.add_argument('--rotate_calib', action='store_true', help='Rotate the calibration file upon load')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()
    calib_matrix = None
    dist_matrix = None

    rot = 90 if args.rotate_calib else 0
    if args.calib_file:
        calib_matrix, dist_matrix = camerautil.load_calibration_file(args.calib_file, rotation=rot)
        # print('calib', calib_matrix)
        # print('dist', dist_matrix)

    finder = finder_type(calib_matrix, dist_matrix)

    if sys.platform == "win32":
        # windows does not expand the "*" files on the command line
        #  so we have to do it.
        import glob

        infiles = []
        for f in args.input_files:
            infiles.extend(glob.glob(f))
        args.input_files = infiles

    if args.output_dir is not None:
        process_files(finder, args.input_files, args.output_dir)
    elif args.time:
        time_processing(finder, args.input_files)

    return

def testutil(finders, annotated_files, metrics, output_dir):
    from tabulate import tabulate
    from termcolor import colored
    import numpy as np
    import traceback
    from timeit import default_timer as timer
    import os

    for file in annotated_files:
        file["results"] = {}


    for (finder_name, finder) in finders.items():
        print("Testing",finder_name)
        try:
            os.makedirs(os.path.join(output_dir, finder_name))
        except FileExistsError:
            pass
        for file in annotated_files:
            try:
                start = timer()
                result_arr = finder.process_image(file['frame'])
                end = timer()

                out_frame = finder.prepare_output_image(file['frame'])
                outfile = os.path.join(output_dir, finder_name, f"{file['time']}.png")
                cv2.imwrite(outfile, out_frame)

                result = {"success": result_arr[0] > 0.5, "time": (end - start)*1000, "error": False}
                for metric in metrics:
                    result[metric["name"]] = result_arr[metric["array_index"]]
            except Exception as e:
                print(traceback.format_exc())
                result = {"error": True}
            file["results"][finder_name] = result


    # print test cases (usaco moment)
    headers = ["file"]

    for finder_name in finders:
        headers.append(f"{finder_name}")

    '''
    for finder_name in finders:
        for metric in metrics:
            headers.append(f"{finder_name} {metric}")
    '''

    rows = []
    for file in annotated_files:
        row = [colored(file["time"], "grey")]
        for finder_name in finders:
            result = file["results"][finder_name]

            if result["error"]:
                row.append(colored('!', 'red'))
            else:
                time = f"{result['time']:0.2f}ms"

                if not result["success"]:
                    row.append(f"{colored('x', 'red')} {time}")
                else:
                    row.append(f"{colored('✓', 'green')} {time}")

        rows.append(row)

    print()
    print(tabulate(rows, headers=headers))
    headers = ["finder", "errors", "failures", "successes"]
    rows = []
    for finder_name in finders:
        errors = 0
        successes = 0
        failures = 0
        for file in annotated_files:
            result = file["results"][finder_name]
            if result["error"]:
                errors += 1
            elif not result["success"]:
                failures += 1
            else:
                successes += 1
        rows.append([finder_name, errors, failures, successes])
    print()
    print(tabulate(rows, headers=headers))

    headers = ["finder", "median", "95th", "99th", "max"]
    rows = []
    for finder_name in finders:
        times = []
        for file in annotated_files:
            result = file["results"][finder_name]
            if not result["error"]:
                times.append(result["time"])
        median = f"{np.percentile(times, 50):0.2f}ms"
        th95 = f"{np.percentile(times, 95):0.2f}ms"
        th99 = f"{np.percentile(times, 99):0.2f}ms"
        max = f"{np.amax(times):0.2f}ms"
        rows.append([finder_name, median, th95, th99, max])
    print()
    print(tabulate(rows, headers=headers))
    merr = 0
    for metric in metrics:
        headers = ["file", "annotated"]
        rows = []
        for finder_name in finders:
            headers.append(finder_name)
            headers.append(finder_name+" error")


        for file in annotated_files:
            row = [colored(file["time"], "grey")]
            if file["annotation"] is None:
                row.append(colored("N/A", "yellow"))
            else:
                row.append(colored(f"{file['annotation'][metric['name']]:0.2f}{metric['unit']}", "magenta"))

            for finder_name in finders:
                if metric['name'] in file['results'][finder_name] and file['results'][finder_name]["success"]:
                    finder_result = file['results'][finder_name][metric['name']]
                    if "convert" in metric:
                        finder_result = metric['convert'](finder_result)
                    row.append(f"{finder_result:0.2f}{metric['unit']}")
                    if file["annotation"] is None:
                        row.append(colored(f"N/A", "yellow"))
                    else:
                        row.append(colored(f"Δ {np.abs(finder_result - file['annotation'][metric['name']]):0.2f}{metric['unit']}", "magenta"))
                else:
                    row.append(colored(f"FAIL", "red"))
                    row.append(colored(f"FAIL", "red"))
            rows.append(row)
        print()
        print(tabulate(rows, headers=headers))
def testutil_main(finder_types, *, output_dir = None, calib_file, rotate_calib, input_files, annotations = {}, metrics = {}):
    '''Main routine for testing multiple Finders'''
    import re
    import camerautil
    import os

    calib_matrix = None
    dist_matrix = None

    rot = 90 if rotate_calib else 0
    if calib_file:
        calib_matrix, dist_matrix = camerautil.load_calibration_file(calib_file, rotation=rot)

    finders = {finder_name: finder_type(calib_matrix, dist_matrix) for (finder_name, finder_type) in finder_types.items()}

    # expand files because no shell
    import glob

    infiles = []
    for f in input_files:
        infiles.extend(glob.glob(f))
    input_files = infiles

    print(f"Reading {len(input_files)} files...")
    time_matcher = re.compile("(.*)\.[\w]+")
    annotated_files = []
    for file in input_files:
        match = time_matcher.fullmatch(os.path.basename(file))
        annotated_files.append({
            "filepath": file,
            "frame":cv2.imread(file),
            "time": None if match is None else match.group(1),
            "annotation": None
        })

    annotated_files.sort(key = lambda x:x["time"])

    for (filter, annotation) in annotations.items():
        # format of annotation:
        # time(str) : {"distance": ..., "angle": ...}
        # [start_time, end_time] : {"distance": ..., "angle": ...}
        # ^: None
        used = False
        if isinstance(filter, str):
            for file in annotated_files:
                if file["time"] == filter:
                    file["annotation"] = annotation
                    used = True
        else:
            for file in annotated_files:
                if filter[0] <= file["time"] <= filter[1]:
                    file["annotation"] = annotation
                    used = True
        if not used:
            print("Warning: annotation", f"{filter} -> {annotation}", "matched no files")


    testutil(finders, annotated_files, metrics, output_dir)

    return

#!/usr/bin/env python3

# Play around with thresholding using GREEN - RED and adaptive OTSU thresholding

import cv2
import numpy


class CubeFinder(object):
    '''Find a cross target'''

    def __init__(self):
        # Color threshold values, in HSV space
        self.low_limit_hsv = numpy.array((25, 95, 110), dtype=numpy.uint8)
        self.high_limit_hsv = numpy.array((75, 255, 255), dtype=numpy.uint8)

        # pixel area of the bounding rectangle - just used to remove stupidly small regions
        self.contour_min_area = 100

        self.erode_kernel = numpy.ones((3, 3), numpy.uint8)
        self.erode_iterations = 0
        return

    @staticmethod
    def contour_center_width(contour):
        '''Find boundingRect of contour, but return center and width/height'''

        x, y, w, h = cv2.boundingRect(contour)
        return (x + int(w / 2), y + int(h / 2)), (w, h)

    @staticmethod
    def quad_fit(contour, approx_dp_error):
        '''Simple polygon fit to contour with error related to perimeter'''

        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, approx_dp_error * peri, True)

    def process_image(self, camera_frame):
        blue_frame, green_frame, red_frame = cv2.split(camera_frame)

        diff = green_frame - red_frame

        # See what the subtracted (grey) frame looks like
        # cv2.imshow("Window", diff)
        # return

        ret2, threshold_frame = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        if self.erode_iterations > 0:
            erode_frame = cv2.erode(threshold_frame, self.erode_kernel, iterations=self.erode_iterations)
        else:
            erode_frame = threshold_frame

        # OpenCV 3 returns 3 parameters!
        # Only need the contours variable
        _, contours, _ = cv2.findContours(erode_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_list = []
        for c in contours:
            center, widths = CubeFinder.contour_center_width(c)
            area = widths[0] * widths[1]
            if area > self.contour_min_area:
                # TODO: use a simple class? Maybe use "attrs" package?
                contour_list.append({'contour': c, 'center': center, 'widths': widths, 'area': area})

        # Sort the list of contours from biggest area to smallest
        contour_list.sort(key=lambda c: c['area'], reverse=True)

        if len(contour_list):
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
            hull_fit = CubeFinder.quad_fit(hull, 0.01)
            # cv2.drawContours(camera_frame, [hull], -1, (0, 255, 0), 1)
            cv2.drawContours(camera_frame, [hull_fit], -1, (255, 0, 0), 2)

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

        return


def main():
    '''Main routine, for testing'''
    import argparse

    parser = argparse.ArgumentParser(description='2017 peg target')
    # parser.add_argument('--output_dir', help='Output directory for processed images')
    # parser.add_argument('--time', action='store_true', help='Loop over files and time it')
    # parser.add_argument('--calib', help='Calibration file')
    parser.add_argument('input_files', nargs='+', help='input files')

    args = parser.parse_args()

    cv2.namedWindow("Window", cv2.WINDOW_NORMAL)

    processor = CubeFinder()
    for image_file in args.input_files:
        bgr_frame = cv2.imread(image_file)
        processor.process_image(bgr_frame)

        q = cv2.waitKey(-1) & 0xFF
        if q == ord('q'):
            break

    cv2.destroyAllWindows()

    return


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main()

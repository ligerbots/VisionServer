#!/usr/bin/env python3

import cv2
import numpy
import json
import math

from genericfinder import GenericFinder, main
import hough_fit


class FaceFinder2021 (GenericFinder):
    '''Find face to shoot at'''

    def __init__(self, calib_matrix=None, dist_matrix=None):
        super().__init__('facefinder', camera='intake', finder_id=13, exposure=1)

        self.cameraMatrix = calib_matrix
        self.distortionMatrix = dist_matrix
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


        return


    def process_image(self, camera_frame):
        '''Main image processing routine'''
        # Convert into grayscale
        gray = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
        print(gray)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces
        self.face = faces[0]
        (x,y,w,h) = self.face
        centerx = x+w/2
        centery = y+h/2
        image_width = camera_frame.shape[1]
        image_middlex = image_width/2

        if abs(centerx - image_middlex) < .1*image_width:
            result = 0
        elif centerx - image_middlex > 0:
            result = 1
        else:
            result = -1
        return [0.0, self.finder_id, result, 0.0, 0.0, 0.0, 0.0, 0.0]

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''

        output_frame = input_frame.copy()
        (x,y,w,h) = self.face
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return output_frame


# Main routine
# This is for development/testing
if __name__ == '__main__':
    main(FaceFinder2021)

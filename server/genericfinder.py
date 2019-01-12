#!/usr/bin/env python3

"""Generic finder used when not searching for any targets.
name and finder_id are instance variables so that we can create multiple
GenericFinders for different purposes.

finder_id may not need to be changed, depending on circumstances."""

import cv2

class GenericFinder(object):
    def __init__(self, name, camera, finder_id=1.0, exposure=0, rotation=None):
        self.name = name
        self.finder_id = float(finder_id)   # id needs to be float! "id" is a reserved word.
        self.desired_camera = camera        # string with camera name
        self.exposure = exposure
        self.rotation = rotation            # cv2.ROTATE_90_CLOCKWISE = 0, cv2.ROTATE_180 = 1, cv2.ROTATE_90_COUNTERCLOCKWISE = 2
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''
        return (1.0, self.finder_id, 0.0, 0.0, 0.0)

    def prepare_output_image(self, output_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''
        if self.rotation is not None:
            cv2.rotate(output_frame, self.rotation)
        return

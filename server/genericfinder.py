#!/usr/bin/env python3

"""Generic finder used when not searching for any targets.
name and finder_id are instance variables so that we can create multiple
GenericFinders for different purposes.

finder_id may not need to be changed, depending on circumstances."""

import cv2


class GenericFinder(object):
    def __init__(self, name, camera, finder_id=1.0, exposure=0, rotation=None, line_coords = None):
        self.name = name
        self.finder_id = float(finder_id)   # id needs to be float! "id" is a reserved word.
        self.camera = camera                # string with camera name
        self.stream_camera = None           # None means same camera
        self.exposure = exposure
        self.rotation = rotation            # cv2.ROTATE_90_CLOCKWISE = 0, cv2.ROTATE_180 = 1, cv2.ROTATE_90_COUNTERCLOCKWISE = 2
        self.line_coords = line_coords      # coordinates to draw a line on the image
        return

    def process_image(self, camera_frame):
        '''Main image processing routine'''
        return (1.0, self.finder_id, 0.0, 0.0, 0.0)

    def prepare_output_image(self, input_frame):
        '''Prepare output image for drive station. Rotate image if needed, otherwise nothing to do.'''

        # WARNING rotation=0 is actually 90deg clockwise (dumb!!)
        if self.rotation is not None:
            # rotate function makes a copy, so no need to do that ahead.
            output_frame = cv2.rotate(input_frame, self.rotation)
        else:
            output_frame = input_frame.copy()
        if self.line_coords is not None:
            cv2.line(output_frame, self.line_coords[0], self.line_coords[1], (255, 255, 255), 2)

        return output_frame

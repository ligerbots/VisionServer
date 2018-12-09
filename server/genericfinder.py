#!/usr/bin/env python3

"""Generic finder used when not searching for any targets"""

class GenericFinder(object):
    
    GENERIC_MODE = 3.0
    READABLE_ID = "generic"
    
    def __init__(self, camera, exposure, rotation=None):
        self.desired_camera = camera        #string with camera name
        self.exposure = exposure
        self.rotation = rotation            # stored as enum: cv2.ROTATE_90_CLOCKWISE = 0, cv2.ROTATE_180 = 1, cv2.ROTATE_90_COUNTERCLOCKWISE = 2
    
    def process_image(self, camera_frame):
        '''Main image processing routine'''
        return (1.0, VisionServer2018.GENERIC_MODE, 0.0, 0.0, 0.0)
    
    def prepare_output_image(self, output_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''
        if self.rotation is not None:
            cv2.rotate(self.output_frame, self.rotation)
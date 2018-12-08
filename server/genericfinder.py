#!/usr/bin/env python3

"""Generic finder used when not searching for any targets"""

class genericfinder:
    
    DRIVER_MODE = 3.0
    
    def process_image(self, camera_frame):
        '''Main image processing routine'''
        #must return 1.0 at zero index since the program DID work
        return (1.0, VisionServer2018.DRIVER_MODE, 0.0, 0.0, 0.0)
    
    #only need this as a placeholder method (does nothing to the image) so we can implement polymorphism
    def prepare_output_image(self, output_frame):
        '''Prepare output image for drive station. Draw the found target contour.'''
        return
#!/usr/bin/env python3

'''Vision server for 2018 Power Up -- updated to conform with VisionServer superclass'''

#TODO are you required to import each library into file if superclass file already imported them?... if so, is there an easier way than simply restating each of the imports?

import time
import cv2
import numpy
import logging

import cscore
from cscore.imagewriter import ImageWriter
from networktables.util import ntproperty
from networktables import NetworkTables

from visionserver import VisionServer
from switchtarget2018 import SwitchTarget2018
from cubefinder2018 import CubeFinder2018

class VisionServer2018_new(VisionServer):

    INITIAL_MODE = 'switch'     #TODO not in init, so are these ok to reset before init is called?
    
    # Cube finding parameters

    # Color threshold values, in HSV space
    cube_hue_low_limit = ntproperty('/SmartDashboard/vision/cube/hue_low_limit', 25,
                                    doc='Hue low limit for thresholding (cube mode)')
    cube_hue_high_limit = ntproperty('/SmartDashboard/vision/cube/hue_high_limit', 75,
                                     doc='Hue high limit for thresholding (cube mode)')

    cube_saturation_low_limit = ntproperty('/SmartDashboard/vision/cube/saturation_low_limit', 95,
                                           doc='Saturation low limit for thresholding (cube mode)')
    cube_saturation_high_limit = ntproperty('/SmartDashboard/vision/cube/saturation_high_limit', 255,
                                            doc='Saturation high limit for thresholding (cube mode)')

    cube_value_low_limit = ntproperty('/SmartDashboard/vision/cube/value_low_limit', 95,
                                      doc='Value low limit for thresholding (cube mode)')
    cube_value_high_limit = ntproperty('/SmartDashboard/vision/cube/value_high_limit', 255,
                                       doc='Value high limit for thresholding (cube mode)')

    cube_exposure = ntproperty('/SmartDashboard/vision/cube/exposure', 0, doc='Camera exposure for cube (0=auto)')

    # Switch target parameters

    switch_hue_low_limit = ntproperty('/SmartDashboard/vision/switch/hue_low_limit', 70,
                                      doc='Hue low limit for thresholding (switch mode)')
    switch_hue_high_limit = ntproperty('/SmartDashboard/vision/switch/hue_high_limit', 100,
                                       doc='Hue high limit for thresholding (switch mode)')

    switch_saturation_low_limit = ntproperty('/SmartDashboard/vision/switch/saturation_low_limit', 100,
                                             doc='Saturation low limit for thresholding (switch mode)')
    switch_saturation_high_limit = ntproperty('/SmartDashboard/vision/switch/saturation_high_limit', 255,
                                              doc='Saturation high limit for thresholding (switch mode)')

    switch_value_low_limit = ntproperty('/SmartDashboard/vision/switch/value_low_limit', 130,
                                        doc='Value low limit for thresholding (switch mode)')
    switch_value_high_limit = ntproperty('/SmartDashboard/vision/switch/value_high_limit', 255,
                                         doc='Value high limit for thresholding (switch mode)')

    switch_exposure = ntproperty('/SmartDashboard/vision/switch/exposure', 6, doc='Camera exposure for switch (0=auto)')
    
    nt_active_mode = ntproperty('/SmartDashboard/vision/active_mode', INITIAL_MODE, doc='Active mode')      #needed to reset b/c we changed the value of INITIAL_MODE afterwards

    def __init__(self, calib_file):
        super.__init__()
        
        self.switch_finder = SwitchTarget2018(calib_file)
        self.cube_finder = CubeFinder2018(calib_file)
        
        self.update_parameters()
        
        # Start in cube mode, then then switch to INITIAL_MODE after camera is fully initialized
        self.switch_mode('cube')
        
    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable

        self.switch_finder.set_color_thresholds(self.switch_hue_low_limit, self.switch_hue_high_limit,
                                                self.switch_saturation_low_limit, self.switch_saturation_high_limit,
                                                self.switch_value_low_limit, self.switch_value_high_limit)
        self.cube_finder.set_color_thresholds(self.cube_hue_low_limit, self.cube_hue_high_limit,
                                              self.cube_saturation_low_limit, self.cube_saturation_high_limit,
                                              self.cube_value_low_limit, self.cube_value_high_limit)

        self.cube_finder.camera_height = self.camera_height
        return

    def add_cameras(self):
        '''add a single camera at /dev/videoN, N=camera_device'''

        self.add_camera('intake', self.camera_device_vision, True)
        self.add_camera('driver', self.camera_device_driver, False)
        return
    
    def process_image(self):
        '''Run the processor on the image to find the target'''

        # make sure to catch any except from processing the image
        try:
            # rvec, tvec return as None if no target found
            if self.curr_processor is not None:
                result = self.curr_processor.process_image(self.camera_frame)
            else:
                result = (1.0, VisionServer2018.DRIVER_MODE, 0.0, 0.0, 0.0)
        except Exception as e:
            logging.error('Caught processing exception: %s', e)
            result = (0.0, 0.0, 0.0, 0.0, 0.0)

        return result

    def prepare_output_image(self):
        '''Prepare an image to send to the drivers station'''

        if self.active_mode == 'driver':
            # stored as enum: ROTATE_90_CLOCKWISE = 0, ROTATE_180 = 1, ROTATE_90_COUNTERCLOCKWISE = 2
            self.output_frame = cv2.rotate(self.camera_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            self.output_frame = self.camera_frame.copy()
            if self.curr_processor is not None:
                self.curr_processor.prepare_output_image(self.output_frame)

        # If saving images, add a little red "Recording" dot in upper left
        if self.image_writer_state:
            cv2.circle(self.output_frame, (20, 20), 5, (0, 0, 255), thickness=10, lineType=8, shift=0)

        return

# -----------------------------------------------------------------------------


# syntax checkers don't like global variables, so use a simple function
def main():
    '''Main routine'''

    import argparse
    parser = argparse.ArgumentParser(description='2018 Vision Server')
    parser.add_argument('--test', action='store_true', help='Run in local test mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose. Turn up debug messages')
    parser.add_argument('--files', action='store_true', help='Process input files instead of camera')
    parser.add_argument('--calib', required=True, help='Calibration file for camera')
    parser.add_argument('input_files', nargs='*', help='input files')

    args = parser.parse_args()

    # To see messages from networktables, you must setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.test:
        # FOR TESTING, set this box as the server
        NetworkTables.enableVerboseLogging()
        NetworkTables.initialize()
    else:
        NetworkTables.initialize(server='10.28.77.2')

    server = VisionServer2018_new(args.calib)

    if args.files:
        if not args.input_files:
            parser.usage()

        server.run_files(args.input_files)
    else:
        server.run()
    return


# Main routine
if __name__ == '__main__':
    main()

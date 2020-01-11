#!/usr/bin/env python3

'''Vision server for 2019 Deep Space'''

import cv2
import logging

from networktables.util import ntproperty
from networktables import NetworkTables

from visionserver import VisionServer
from genericfinder import GenericFinder
from rrtargetfinder2019 import RRTargetFinder2019
from hatchfinder2019 import HatchFinder2019
from linefinder2019 import LineFinder2019

class VisionServer2019(VisionServer):

    # Retro-reflective targer finding parameters

    # Color threshold values, in HSV space
    rrtarget_hue_low_limit = ntproperty('/SmartDashboard/vision/rrtarget/hue_low_limit', 25,
                                    doc='Hue low limit for thresholding (rrtarget mode)')
    rrtarget_hue_high_limit = ntproperty('/SmartDashboard/vision/rrtarget/hue_high_limit', 75,
                                     doc='Hue high limit for thresholding (rrtarget mode)')

    rrtarget_saturation_low_limit = ntproperty('/SmartDashboard/vision/rrtarget/saturation_low_limit', 95,
                                           doc='Saturation low limit for thresholding (rrtarget mode)')
    rrtarget_saturation_high_limit = ntproperty('/SmartDashboard/vision/rrtarget/saturation_high_limit', 255,
                                            doc='Saturation high limit for thresholding (rrtarget mode)')

    rrtarget_value_low_limit = ntproperty('/SmartDashboard/vision/rrtarget/value_low_limit', 95,
                                      doc='Value low limit for thresholding (rrtarget mode)')
    rrtarget_value_high_limit = ntproperty('/SmartDashboard/vision/rrtarget/value_high_limit', 255,
                                       doc='Value high limit for thresholding (rrtarget mode)')

    rrtarget_exposure = ntproperty('/SmartDashboard/vision/rrtarget/exposure', 0, doc='Camera exposure for rrtarget (0=auto)')

    @Override
    def __init__(self, calib_file):
        super.__init__()

        #  Initial mode for start of match.
        #  VisionServer switches to this mode after a second, to get the cameras initialized
        self.initial_mode = 'rrtarget'
        self.switch_mode(self.initial_mode)

        #self.switch_finder = SwitchTarget2018(calib_file)
        #self.cube_finder = CubeFinder2018(calib_file)

        self.camera_device_front = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_DF7AF0BE-video-index0'    #for driver and rrtarget processing
        self.camera_device_floor = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_70E19A9E-video-index0'    #for line and hatch processing

        self.generic_finder = GenericFinder("driver", "front")
        self.add_target_finder(self.generic_finder)

        self.rrtarget_finder = RRTargetFinder2019(calib_file)    #TODO change to the real finders once they are created
        self.add_target_finder(self.rrtarget_finder)

        self.hatch_finder = HatchFinder2019(calib_file)
        self.add_target_finder(self.hatch_finder)

        self.line_finder = LineFinder2019(calib_file)
        self.add_target_finder(self.line_finder)

        self.update_parameters()

    @Override
    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable

        self.rrtarget_finder.set_color_thresholds(self.rrtarget_hue_low_limit, self.rrtarget_hue_high_limit,
                                                self.rrtarget_saturation_low_limit, self.rrtarget_saturation_high_limit,
                                                self.rrtarget_value_low_limit, self.rrtarget_value_high_limit)
        return
    
    @Override
    def add_cameras(self):
        '''add a single camera at /dev/videoN, N=camera_device'''

        self.add_camera('front', self.camera_device_front, True)
        self.add_camera('floor', self.camera_device_floor, False)
        return

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

    server = VisionServer2019(args.calib)

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

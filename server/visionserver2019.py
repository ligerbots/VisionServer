#!/usr/bin/env python3

'''Vision server for 2019 Deep Space'''

from networktables.util import ntproperty

from visionserver import VisionServer, main
from genericfinder import GenericFinder
from rrtargetfinder2019 import RRTargetFinder2019
# from hatchfinder2019 import HatchFinder2019
# from linefinder2019 import LineFinder2019


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

    def __init__(self, calib_file, test_mode=False):
        super().__init__(initial_mode='driver', test_mode=test_mode)

        self.camera_device_front = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_DF7AF0BE-video-index0'    # for driver and rrtarget processing
        self.camera_device_side = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_70E19A9E-video-index0'    # for line and hatch processing

        self.add_cameras()

        self.generic_finder_front = GenericFinder("driver_front", "front")  `   #finder_id=1.0
        self.add_target_finder(self.generic_finder_front)                       #TODO make it a seperate finder to draw a line down the screen where 
                                                                                #the line at the bottom of the rocket will be

        self.generic_finder_side = GenericFinder("driver_side", "side", finder_id=2.0, rotation=cv2.ROTATE_90_CLOCKWISE)
        self.add_target_finder(self.generic_finder_side)

        self.rrtarget_finder = RRTargetFinder2019(calib_file)       #finder_id=3.0
        self.add_target_finder(self.rrtarget_finder)

        # self.hatch_finder = HatchFinder2019(calib_file)
        # self.add_target_finder(self.hatch_finder)

        # self.line_finder = LineFinder2019(calib_file)
        # self.add_target_finder(self.line_finder)

        self.update_parameters()

        # start in rrtarget mode to get cameras going. Will switch to 'driver' after 1 sec.
        self.switch_mode('rrtarget')
        return

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable

        self.rrtarget_finder.set_color_thresholds(self.rrtarget_hue_low_limit, self.rrtarget_hue_high_limit,
                                                  self.rrtarget_saturation_low_limit, self.rrtarget_saturation_high_limit,
                                                  self.rrtarget_value_low_limit, self.rrtarget_value_high_limit)
        return

    def add_cameras(self):
        '''add a single camera at /dev/videoN, N=camera_device'''

        self.add_camera('front', self.camera_device_front, True)
        self.add_camera('side', self.camera_device_side, False)
        return

    def mode_after_error(self):
        return 'driver'


# Main routine
if __name__ == '__main__':
    main(VisionServer2019)
